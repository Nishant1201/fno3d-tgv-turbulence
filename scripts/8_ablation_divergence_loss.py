import sys
import json
import time
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fno_model import FNO3d

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9.5,
})


class HDF5TGVDataset(Dataset):
    def __init__(self, h5_filepath):
        self.filepath = str(h5_filepath)
        with h5py.File(self.filepath, "r") as f:
            self.samples = f["velocity"].shape[0]
            self.timesteps = f["velocity"].shape[1]
            self.sequences_per_sample = self.timesteps - 2

    def __len__(self):
        return self.samples * self.sequences_per_sample

    def __getitem__(self, idx):
        s = idx // self.sequences_per_sample
        t = idx % self.sequences_per_sample
        with h5py.File(self.filepath, "r") as f:
            x = f["velocity"][s, t]
            y1 = f["velocity"][s, t + 1]
            y2 = f["velocity"][s, t + 2]
        return torch.tensor(x), torch.tensor(y1), torch.tensor(y2)


def spectral_div_mse(u_grid, kx, ky, kz):
    """
    Computes mean(|div(u)|^2) in float32 for u_grid of shape (B, N, N, N, 3).
    """
    N = u_grid.shape[1]
    u = u_grid.permute(0, 4, 1, 2, 3).float()  # (B, 3, N, N, N)
    u_hat = torch.fft.rfftn(u, dim=(2, 3, 4))  # (B, 3, N, N, N//2 + 1)

    div_hat = 1j * (
        kx * u_hat[:, 0]
        + ky * u_hat[:, 1]
        + kz * u_hat[:, 2]
    )  # (B, N, N, N//2 + 1)
    div_u = torch.fft.irfftn(div_hat, s=(N, N, N), dim=(1, 2, 3))  # (B, N, N, N)
    return torch.mean(div_u ** 2)

def log_line(msg, log_path):
    print(msg, flush=True)
    with open(log_path, "a") as lf:
        lf.write(msg + "\n")


def train_div_fno(
    train_h5,
    ckpt_path,
    weights_path,
    log_path,
    device,
    lambda_div=1e-2,
    epochs=150,
    modes=24,
    width=20,
):
    dataset = HDF5TGVDataset(train_h5)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)

    N = dataset[0][0].shape[0]
    k_1d = torch.fft.fftfreq(N, d=1.0 / N, device=device)
    kz_1d = torch.fft.rfftfreq(N, d=1.0 / N, device=device)
    kx, ky, kz = torch.meshgrid(k_1d, k_1d, kz_1d, indexing="ij")

    model = FNO3d(modes=modes, width=width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    start_epoch = 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"]
        log_line(f"[div_lam={lambda_div}] Resumed from epoch {start_epoch}/{epochs}", log_path)
    else:
        log_line(
            f"[div_lam={lambda_div}] Starting training from scratch for {epochs} epochs...",
            log_path,
        )

    if start_epoch >= epochs:
        torch.save(model.state_dict(), weights_path)
        return

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    model.train()
    for ep in range(start_epoch, epochs):
        running_loss = 0.0
        for x, y1, y2 in loader:
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred1 = x + model(x)
                pred2 = pred1 + model(pred1)

            loss_mse = F.mse_loss(pred1.float(), y1.float()) + F.mse_loss(pred2.float(), y2.float())
            loss_div = spectral_div_mse(pred1, kx, ky, kz) + spectral_div_mse(pred2, kx, ky, kz)
            loss = loss_mse + lambda_div * loss_div

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(loader)
        log_line(f"[div_lam={lambda_div}] Epoch {ep + 1}/{epochs} | Loss: {avg_loss:.6e}", log_path)

        if (ep + 1) % 50 == 0 or (ep + 1) == epochs:
            torch.save(
                {
                    "epoch": ep + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                },
                ckpt_path,
            )
            torch.save(model.state_dict(), weights_path)

    if device.type == "cuda":
        torch.cuda.synchronize()
    log_line(f"[div_lam={lambda_div}] Completed in {time.perf_counter() - t0:.2f} s", log_path)


def evaluate_rollout(weights_path, fdns_traj, device, modes=24, width=20):
    model = FNO3d(modes=modes, width=width).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    num_steps = fdns_traj.shape[0] - 1
    pred_traj = np.empty_like(fdns_traj)
    pred_traj[0] = fdns_traj[0]

    with torch.no_grad():
        u_curr = torch.tensor(fdns_traj[0], dtype=torch.float32, device=device).unsqueeze(0)
        for step in range(1, num_steps + 1):
            u_curr = u_curr + model(u_curr)
            pred_traj[step] = u_curr.squeeze(0).cpu().numpy()

    N = fdns_traj.shape[1]
    k_1d = torch.fft.fftfreq(N, d=1.0 / N, device=device)
    kz_1d = torch.fft.rfftfreq(N, d=1.0 / N, device=device)
    kx, ky, kz = torch.meshgrid(k_1d, k_1d, kz_1d, indexing="ij")

    ke_p, enst_p, ke_r, enst_r, div_rms_series = [], [], [], [], []

    with torch.no_grad():
        for step in range(fdns_traj.shape[0]):
            up_grid = torch.tensor(pred_traj[step], dtype=torch.float32, device=device).unsqueeze(0)
            ur_grid = torch.tensor(fdns_traj[step], dtype=torch.float32, device=device).unsqueeze(0)

            up = up_grid.permute(0, 4, 1, 2, 3)
            ur = ur_grid.permute(0, 4, 1, 2, 3)

            ke_p.append(0.5 * torch.mean(torch.sum(up * up, dim=1)).item())
            ke_r.append(0.5 * torch.mean(torch.sum(ur * ur, dim=1)).item())

            up_hat = torch.fft.rfftn(up, dim=(2, 3, 4))
            ur_hat = torch.fft.rfftn(ur, dim=(2, 3, 4))

            cx_p = torch.fft.irfftn(1j * (ky * up_hat[:, 2] - kz * up_hat[:, 1]), s=(N, N, N))
            cy_p = torch.fft.irfftn(1j * (kz * up_hat[:, 0] - kx * up_hat[:, 2]), s=(N, N, N))
            cz_p = torch.fft.irfftn(1j * (kx * up_hat[:, 1] - ky * up_hat[:, 0]), s=(N, N, N))
            enst_p.append((0.5 * torch.mean(cx_p**2 + cy_p**2 + cz_p**2)).item())

            cx_r = torch.fft.irfftn(1j * (ky * ur_hat[:, 2] - kz * ur_hat[:, 1]), s=(N, N, N))
            cy_r = torch.fft.irfftn(1j * (kz * ur_hat[:, 0] - kx * ur_hat[:, 2]), s=(N, N, N))
            cz_r = torch.fft.irfftn(1j * (kx * ur_hat[:, 1] - ky * ur_hat[:, 0]), s=(N, N, N))
            enst_r.append((0.5 * torch.mean(cx_r**2 + cy_r**2 + cz_r**2)).item())

            div_rms = torch.sqrt(spectral_div_mse(up_grid, kx, ky, kz)).item()
            div_rms_series.append(div_rms)

    ke_p, ke_r = np.array(ke_p), np.array(ke_r)
    enst_p, enst_r = np.array(enst_p), np.array(enst_r)
    div_rms_arr = np.array(div_rms_series)

    metrics = {
        "u_rel_l2_pct": round(float(np.linalg.norm(pred_traj - fdns_traj) / np.linalg.norm(fdns_traj)) * 100, 3),
        "ke_rel_l2_pct": round(float(np.linalg.norm(ke_p - ke_r) / np.linalg.norm(ke_r)) * 100, 3),
        "enst_rel_l2_pct": round(float(np.linalg.norm(enst_p - enst_r) / np.linalg.norm(enst_r)) * 100, 3),
        "enst_peak_err_pct": round(float(abs(enst_p.max() - enst_r.max()) / abs(enst_r.max())) * 100, 3),
        "mean_div_rms": float(np.mean(div_rms_arr)),
        "final_div_rms": float(div_rms_arr[-1]),
    }
    return ke_p, enst_p, ke_r, enst_r, div_rms_arr, metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lambda_div = 1e-2

    train_h5 = PROJECT_ROOT / "data" / "train_data.h5"
    test_h5 = PROJECT_ROOT / "data" / "test_data_64.h5"
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    log_dir = PROJECT_ROOT / "logs"
    fig_dir = PROJECT_ROOT / "results" / "figures"
    bench_dir = PROJECT_ROOT / "results" / "benchmarks"

    for d in (ckpt_dir, log_dir, fig_dir, bench_dir):
        d.mkdir(parents=True, exist_ok=True)

    ckpt_div = ckpt_dir / "fno3d_checkpoint_div1e-2.pth"
    weights_div = ckpt_dir / "fno3d_weights_div1e-2.pth"
    weights_base = ckpt_dir / "fno3d_weights_latest.pth"
    log_div = log_dir / "ablation_div_loss_train.txt"

    train_div_fno(
        train_h5=train_h5,
        ckpt_path=ckpt_div,
        weights_path=weights_div,
        log_path=log_div,
        device=device,
        lambda_div=lambda_div,
        epochs=150,
        modes=24,
        width=20,
    )

    with h5py.File(test_h5, "r") as f:
        fdns_traj = f["velocity"][0]

    _, _, ke_ref, enst_ref, div_ref, _ = evaluate_rollout(weights_base, fdns_traj, device, modes=24, width=20)
    ke_base, enst_base, _, _, div_base, m_base = evaluate_rollout(weights_base, fdns_traj, device, modes=24, width=20)
    ke_div, enst_div, _, _, div_div, m_div = evaluate_rollout(weights_div, fdns_traj, device, modes=24, width=20)

    rows = [
        {"config": "MSE Only (lambda_div = 0)", **m_base},
        {"config": f"MSE + Div Loss (lambda_div = {lambda_div:.0e})", **m_div},
    ]

    with open(bench_dir / "ablation_div_loss.json", "w") as jf:
        json.dump(rows, jf, indent=2)

    md_lines = [
        "| Configuration | Rel. $L_2$ $\\mathbf{u}$ (%) | Rel. $L_2$ $E_k$ (%) | Rel. $L_2$ $\\mathcal{E}$ (%) | $\\mathcal{E}_{\\max}$ Err (%) | Mean $\\|\\nabla \\cdot \\mathbf{u}\\|_{\\mathrm{rms}}$ | Final $\\|\\nabla \\cdot \\mathbf{u}\\|_{\\mathrm{rms}}$ |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['config']} | {r['u_rel_l2_pct']} | {r['ke_rel_l2_pct']} | {r['enst_rel_l2_pct']} | "
            f"{r['enst_peak_err_pct']} | {r['mean_div_rms']:.3e} | {r['final_div_rms']:.3e} |"
        )
    with open(bench_dir / "ablation_div_loss.md", "w") as mf:
        mf.write("\n".join(md_lines) + "\n")

    print("\n" + "\n".join(md_lines))

    times = np.linspace(0.0, 10.0, fdns_traj.shape[0])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(times, enst_ref, "k-", lw=2.0, label=r"Filtered DNS ($256^3 \rightarrow 64^3$)")
    ax1.plot(times, enst_base, "r--", lw=2.0, label=r"3D FNO MSE-Only ($\lambda_{\mathrm{div}}=0$)")
    ax1.plot(times, enst_div, color="tab:blue", linestyle="-.", lw=1.8, label=rf"3D FNO + Div Loss ($\lambda_{{\mathrm{{div}}}}={lambda_div:.0e}$)")
    ax1.set_title("Enstrophy Evolution: MSE vs. Divergence-Regularized")
    ax1.set_xlabel("Time $t$ (s)")
    ax1.set_ylabel(r"Enstrophy $\mathcal{E}(t)$")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.semilogy(times, np.maximum(div_base, 1e-12), "r--", lw=2.0, label=r"MSE-Only ($\lambda_{\mathrm{div}}=0$)")
    ax2.semilogy(times, np.maximum(div_div, 1e-12), color="tab:blue", linestyle="-.", lw=1.8, label=rf"+ Div Loss ($\lambda_{{\mathrm{{div}}}}={lambda_div:.0e}$)")
    ax2.semilogy(times, np.maximum(div_ref, 1e-12), "k:", lw=1.5, label=r"Filtered DNS Reference")
    ax2.set_title(r"Divergence Residual $\|\nabla \cdot \mathbf{u}\|_{\mathrm{rms}}$ vs. Time")
    ax2.set_xlabel("Time $t$ (s)")
    ax2.set_ylabel(r"RMS Divergence $\|\nabla \cdot \mathbf{u}\|_{\mathrm{rms}}$")
    ax2.legend()
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig_out = fig_dir / "fig8_ablation_div_loss.png"
    plt.savefig(fig_out, dpi=300)
    plt.close(fig)
    print(f"Saved -> {fig_out}")


if __name__ == "__main__":
    main()
