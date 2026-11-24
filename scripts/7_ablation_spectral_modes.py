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


def log_line(msg, log_path):
    print(msg, flush=True)
    with open(log_path, "a") as lf:
        lf.write(msg + "\n")


def train_modes16(train_h5, ckpt_path, weights_path, log_path, device, epochs=150, width=20):
    dataset = HDF5TGVDataset(train_h5)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)

    model = FNO3d(modes=16, width=width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    start_epoch = 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"]
        log_line(f"[modes=16] Resumed from epoch {start_epoch}/{epochs}", log_path)
    else:
        log_line(f"[modes=16] Starting training from scratch for {epochs} epochs...", log_path)

    if start_epoch >= epochs:
        torch.save(model.state_dict(), weights_path)
        return 0.0

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    model.train()
    for ep in range(start_epoch, epochs):
        running_loss = 0.0
        for x, y1, y2 in loader:
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            x.requires_grad_(True)
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred1 = x + model(x)
                loss1 = F.mse_loss(pred1, y1)
                pred2 = pred1 + model(pred1)
                loss2 = F.mse_loss(pred2, y2)
                loss = loss1 + loss2

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(loader)
        log_line(f"[modes=16] Epoch {ep + 1}/{epochs} | Loss: {avg_loss:.6f}", log_path)

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
    train_elapsed = time.perf_counter() - t0
    log_line(f"[modes=16] Training finished in {train_elapsed:.2f} s", log_path)
    return train_elapsed


def rollout_and_metrics(weights_path, modes, width, fdns_traj, device):
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
    kx = torch.fft.fftfreq(N, d=1.0 / N, device=device)
    ky = kx
    kz = torch.fft.rfftfreq(N, d=1.0 / N, device=device)
    Kx, Ky, Kz = torch.meshgrid(kx, ky, kz, indexing="ij")

    ke_p, enst_p, ke_r, enst_r = [], [], [], []
    with torch.no_grad():
        for step in range(fdns_traj.shape[0]):
            up = torch.tensor(pred_traj[step], dtype=torch.float32, device=device).unsqueeze(0).permute(0, 4, 1, 2, 3)
            ur = torch.tensor(fdns_traj[step], dtype=torch.float32, device=device).unsqueeze(0).permute(0, 4, 1, 2, 3)

            ke_p.append(0.5 * torch.mean(torch.sum(up * up, dim=1)).item())
            ke_r.append(0.5 * torch.mean(torch.sum(ur * ur, dim=1)).item())

            up_hat = torch.fft.rfftn(up, dim=(2, 3, 4))
            ur_hat = torch.fft.rfftn(ur, dim=(2, 3, 4))

            cx_p = torch.fft.irfftn(1j * (Ky * up_hat[:, 2] - Kz * up_hat[:, 1]), s=(N, N, N))
            cy_p = torch.fft.irfftn(1j * (Kz * up_hat[:, 0] - Kx * up_hat[:, 2]), s=(N, N, N))
            cz_p = torch.fft.irfftn(1j * (Kx * up_hat[:, 1] - Ky * up_hat[:, 0]), s=(N, N, N))
            enst_p.append((0.5 * torch.mean(cx_p**2 + cy_p**2 + cz_p**2)).item())

            cx_r = torch.fft.irfftn(1j * (Ky * ur_hat[:, 2] - Kz * ur_hat[:, 1]), s=(N, N, N))
            cy_r = torch.fft.irfftn(1j * (Kz * ur_hat[:, 0] - Kx * ur_hat[:, 2]), s=(N, N, N))
            cz_r = torch.fft.irfftn(1j * (Kx * ur_hat[:, 1] - Ky * ur_hat[:, 0]), s=(N, N, N))
            enst_r.append((0.5 * torch.mean(cx_r**2 + cy_r**2 + cz_r**2)).item())

    ke_p, ke_r = np.array(ke_p), np.array(ke_r)
    enst_p, enst_r = np.array(enst_p), np.array(enst_r)

    metrics = {
        "u_rel_l2_pct": round(float(np.linalg.norm(pred_traj - fdns_traj) / np.linalg.norm(fdns_traj)) * 100, 3),
        "ke_rel_l2_pct": round(float(np.linalg.norm(ke_p - ke_r) / np.linalg.norm(ke_r)) * 100, 3),
        "enst_rel_l2_pct": round(float(np.linalg.norm(enst_p - enst_r) / np.linalg.norm(enst_r)) * 100, 3),
        "ke_final_err_pct": round(float(abs(ke_p[-1] - ke_r[-1]) / abs(ke_r[-1])) * 100, 3),
        "enst_peak_err_pct": round(float(abs(enst_p.max() - enst_r.max()) / abs(enst_r.max())) * 100, 3),
    }
    return ke_p, enst_p, ke_r, enst_r, metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_h5 = PROJECT_ROOT / "data" / "train_data.h5"
    test_h5 = PROJECT_ROOT / "data" / "test_data_64.h5"
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    log_dir = PROJECT_ROOT / "logs"
    fig_dir = PROJECT_ROOT / "results" / "figures"
    bench_dir = PROJECT_ROOT / "results" / "benchmarks"

    for d in (ckpt_dir, log_dir, fig_dir, bench_dir):
        d.mkdir(parents=True, exist_ok=True)

    ckpt16 = ckpt_dir / "fno3d_checkpoint_modes16.pth"
    weights16 = ckpt_dir / "fno3d_weights_modes16.pth"
    weights24 = ckpt_dir / "fno3d_weights_latest.pth"
    log16 = log_dir / "ablation_modes16_train.txt"

    train_modes16(train_h5, ckpt16, weights16, log16, device, epochs=150, width=20)

    with h5py.File(test_h5, "r") as f:
        fdns_traj = f["velocity"][0]

    ke16, enst16, ke_ref, enst_ref, m16 = rollout_and_metrics(weights16, 16, 20, fdns_traj, device)
    ke24, enst24, _, _, m24 = rollout_and_metrics(weights24, 24, 20, fdns_traj, device)

    rows = [
        {"config": "modes=16, width=20 (50% Nyquist)", **m16},
        {"config": "modes=24, width=20 (75% Nyquist)", **m24},
    ]

    with open(bench_dir / "ablation_modes16_vs_24.json", "w") as jf:
        json.dump(rows, jf, indent=2)

    md_lines = [
        "| Configuration | Rel. $L_2$ $\\mathbf{u}$ (%) | Rel. $L_2$ $E_k$ (%) | Rel. $L_2$ $\\mathcal{E}$ (%) | $E_k(T)$ Err (%) | $\\mathcal{E}_{\\max}$ Err (%) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['config']} | {r['u_rel_l2_pct']} | {r['ke_rel_l2_pct']} | "
            f"{r['enst_rel_l2_pct']} | {r['ke_final_err_pct']} | {r['enst_peak_err_pct']} |"
        )
    with open(bench_dir / "ablation_modes16_vs_24.md", "w") as mf:
        mf.write("\n".join(md_lines) + "\n")

    print("\n" + "\n".join(md_lines))

    times = np.linspace(0.0, 10.0, fdns_traj.shape[0])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(times, ke_ref, "k-", lw=2.0, label=r"Filtered DNS ($256^3 \rightarrow 64^3$)")
    ax1.plot(times, ke24, "r--", lw=2.0, label=r"3D FNO ($M=24$, $64^3$)")
    ax1.plot(times, ke16, color="tab:blue", linestyle="-.", lw=1.8, label=r"3D FNO ($M=16$, $64^3$)")
    ax1.set_title("Spectral Bandwidth Ablation: Kinetic Energy")
    ax1.set_xlabel("Time $t$ (s)")
    ax1.set_ylabel(r"Kinetic Energy $E_k(t)$")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(times, enst_ref, "k-", lw=2.0, label=r"Filtered DNS ($256^3 \rightarrow 64^3$)")
    ax2.plot(times, enst24, "r--", lw=2.0, label=r"3D FNO ($M=24$, $64^3$)")
    ax2.plot(times, enst16, color="tab:blue", linestyle="-.", lw=1.8, label=r"3D FNO ($M=16$, $64^3$)")
    ax2.set_title("Spectral Bandwidth Ablation: Enstrophy")
    ax2.set_xlabel("Time $t$ (s)")
    ax2.set_ylabel(r"Enstrophy $\mathcal{E}(t)$")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig_out = fig_dir / "fig7_ablation_modes16_vs_24.png"
    plt.savefig(fig_out, dpi=300)
    plt.close(fig)
    print(f"Saved -> {fig_out}")


if __name__ == "__main__":
    main()
