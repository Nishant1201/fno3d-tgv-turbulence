import sys
import json
import time
import math
from pathlib import Path
import h5py
import numpy as np
import cupy as cp
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fno_model import FNO3d

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})


def spectral_truncate_cupy(u_high, N_low, N_sim):
    if u_high.shape[1] == N_low:
        return u_high.astype(cp.float32)

    u_hat_high = cp.fft.fftn(u_high.astype(cp.complex64), axes=(1, 2, 3))
    u_hat_low = cp.zeros((3, N_low, N_low, N_low), dtype=cp.complex64)

    half = N_low // 2
    u_hat_low[:, :half, :half, :half] = u_hat_high[:, :half, :half, :half]
    u_hat_low[:, -half:, :half, :half] = u_hat_high[:, -half:, :half, :half]
    u_hat_low[:, :half, -half:, :half] = u_hat_high[:, :half, -half:, :half]
    u_hat_low[:, -half:, -half:, :half] = u_hat_high[:, -half:, -half:, :half]
    u_hat_low[:, :half, :half, -half:] = u_hat_high[:, :half, :half, -half:]
    u_hat_low[:, -half:, :half, -half:] = u_hat_high[:, -half:, :half, -half:]
    u_hat_low[:, :half, -half:, -half:] = u_hat_high[:, :half, -half:, -half:]
    u_hat_low[:, -half:, -half:, -half:] = u_hat_high[:, -half:, -half:, -half:]

    u_low = cp.real(cp.fft.ifftn(u_hat_low, axes=(1, 2, 3)))
    return (u_low * (N_low / N_sim) ** 3).astype(cp.float32)


def run_pseudospectral_rk4_single(
    N_sim,
    N_save=None,
    u0_init=None,
    phase=(0.0, 0.0, 0.0),
    T=10.0,
    dt=0.01,
    snapshot_dt=0.1,
    nu=0.000625,
    record_snapshots=True,
):
    """
    Executes a single pseudo-spectral RK4 trajectory at grid resolution N_sim^3.
    Returns (snapshots_np or None, elapsed_sec, peak_vram_mb).
    """
    if N_save is None:
        N_save = N_sim

    timesteps = int(round(T / snapshot_dt)) + 1
    snap_interval = int(round(snapshot_dt / dt))

    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()

    x = cp.linspace(0, 2 * math.pi, N_sim, endpoint=False, dtype=cp.float64)
    X = cp.meshgrid(x, x, x, indexing="ij")

    kx = cp.fft.fftfreq(N_sim, 1.0 / N_sim).astype(cp.float64)
    ky = kx
    kz = cp.fft.rfftfreq(N_sim, 1.0 / N_sim).astype(cp.float64)
    K = cp.array(cp.meshgrid(kx, ky, kz, indexing="ij"))

    K2 = cp.sum(K * K, axis=0)
    K2_safe = cp.where(K2 == 0, 1.0, K2)
    K_over_K2 = K / K2_safe

    kmax_dealias = 2.0 / 3.0 * (N_sim / 2 + 1)
    dealias = (
        (cp.abs(K[0]) < kmax_dealias)
        & (cp.abs(K[1]) < kmax_dealias)
        & (cp.abs(K[2]) < kmax_dealias)
    )

    U = cp.empty((3, N_sim, N_sim, N_sim), dtype=cp.float64)
    if u0_init is not None:
        U[:] = cp.asarray(u0_init.transpose(3, 0, 1, 2), dtype=cp.float64)
    else:
        px, py, pz = phase
        U[0] = cp.sin(X[0] + px) * cp.cos(X[1] + py) * cp.cos(X[2] + pz)
        U[1] = -cp.cos(X[0] + px) * cp.sin(X[1] + py) * cp.cos(X[2] + pz)
        U[2] = 0.0

    U_hat = cp.empty((3, N_sim, N_sim, N_sim // 2 + 1), dtype=cp.complex128)
    for i in range(3):
        U_hat[i] = cp.fft.rfftn(U[i], axes=(0, 1, 2))

    U_hat0 = cp.empty_like(U_hat)
    U_hat1 = cp.empty_like(U_hat)
    dU = cp.empty_like(U_hat)
    curl = cp.empty_like(U)
    P_hat = cp.empty((N_sim, N_sim, N_sim // 2 + 1), dtype=cp.complex128)

    snapshots = None
    if record_snapshots:
        snapshots = np.empty((timesteps, N_save, N_save, N_save, 3), dtype=np.float32)
        snapshots[0] = cp.asnumpy(spectral_truncate_cupy(U, N_save, N_sim)).transpose(1, 2, 3, 0)

    b_rk = [0.5, 0.5, 1.0, 0.0]
    a_rk = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]

    t = 0.0
    tstep = 0
    snap_idx = 1

    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()

    while t < T - 1e-5:
        t += dt
        tstep += 1
        U_hat0[:] = U_hat
        U_hat1[:] = U_hat

        for rk in range(4):
            if rk > 0:
                for i in range(3):
                    U[i] = cp.fft.irfftn(U_hat[i], axes=(0, 1, 2))

            curl[0] = cp.fft.irfftn(1j * (K[1] * U_hat[2] - K[2] * U_hat[1]), axes=(0, 1, 2))
            curl[1] = cp.fft.irfftn(1j * (K[2] * U_hat[0] - K[0] * U_hat[2]), axes=(0, 1, 2))
            curl[2] = cp.fft.irfftn(1j * (K[0] * U_hat[1] - K[1] * U_hat[0]), axes=(0, 1, 2))

            dU[0] = cp.fft.rfftn(U[1] * curl[2] - U[2] * curl[1], axes=(0, 1, 2))
            dU[1] = cp.fft.rfftn(U[2] * curl[0] - U[0] * curl[2], axes=(0, 1, 2))
            dU[2] = cp.fft.rfftn(U[0] * curl[1] - U[1] * curl[0], axes=(0, 1, 2))

            dU *= dealias
            P_hat[:] = cp.sum(dU * K_over_K2, axis=0)
            dU -= P_hat * K
            dU -= nu * K2 * U_hat

            if rk < 3:
                U_hat[:] = U_hat0 + b_rk[rk] * dt * dU
            U_hat1 += a_rk[rk] * dt * dU

        U_hat[:] = U_hat1
        for i in range(3):
            U[i] = cp.fft.irfftn(U_hat[i], axes=(0, 1, 2))

        if record_snapshots and (tstep % snap_interval == 0) and snap_idx < timesteps:
            snapshots[snap_idx] = cp.asnumpy(spectral_truncate_cupy(U, N_save, N_sim)).transpose(
                1, 2, 3, 0
            )
            snap_idx += 1

    cp.cuda.Device().synchronize()
    elapsed_sec = time.perf_counter() - t0
    peak_vram_mb = mempool.total_bytes() / (1024**2)

    del U, U_hat, U_hat0, U_hat1, dU, curl, P_hat, X, K, K2, K2_safe, K_over_K2, dealias
    mempool.free_all_blocks()

    return snapshots, elapsed_sec, peak_vram_mb


def run_fno_rollout(weights_path, u0_64, num_steps, device, modes=24, width=20):
    """
    Runs autoregressive FNO rollout in float32 and returns
    (pred_traj [num_steps+1, N, N, N, 3], rollout_sec, peak_vram_mb).
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model = FNO3d(modes=modes, width=width).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    pred_traj = np.empty((num_steps + 1, *u0_64.shape), dtype=np.float32)
    pred_traj[0] = u0_64

    with torch.no_grad():
        u_curr = torch.tensor(u0_64, dtype=torch.float32, device=device).unsqueeze(0)
        # Warmup forward pass to exclude one-time cuFFT plan initialization
        _ = model(u_curr)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for step in range(1, num_steps + 1):
            u_curr = u_curr + model(u_curr)
            pred_traj[step] = u_curr.squeeze(0).cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
        rollout_sec = time.perf_counter() - t0

    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    )
    return pred_traj, rollout_sec, peak_vram_mb


def compute_series_and_metrics(traj, ref_traj, device):
    """
    Computes KE and Enstrophy time series plus relative L2 error metrics
    against ref_traj (Filtered DNS 256^3 -> 64^3).
    """
    N = traj.shape[1]
    kx = torch.fft.fftfreq(N, d=1.0 / N, device=device)
    ky = kx
    kz = torch.fft.rfftfreq(N, d=1.0 / N, device=device)
    Kx, Ky, Kz = torch.meshgrid(kx, ky, kz, indexing="ij")

    ke_p, enst_p = [], []
    ke_r, enst_r = [], []

    with torch.no_grad():
        for step in range(traj.shape[0]):
            up = torch.tensor(traj[step], dtype=torch.float32, device=device).unsqueeze(0).permute(0, 4, 1, 2, 3)
            ur = torch.tensor(ref_traj[step], dtype=torch.float32, device=device).unsqueeze(0).permute(0, 4, 1, 2, 3)

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
        "u_rel_l2_pct": round(float(np.linalg.norm(traj - ref_traj) / np.linalg.norm(ref_traj)) * 100, 3),
        "ke_rel_l2_pct": round(float(np.linalg.norm(ke_p - ke_r) / np.linalg.norm(ke_r)) * 100, 3),
        "enst_rel_l2_pct": round(float(np.linalg.norm(enst_p - enst_r) / np.linalg.norm(enst_r)) * 100, 3),
        "ke_final_err_pct": round(float(abs(ke_p[-1] - ke_r[-1]) / abs(ke_r[-1])) * 100, 3),
        "enst_peak_err_pct": round(float(abs(enst_p.max() - enst_r.max()) / abs(enst_r.max())) * 100, 3),
    }
    return ke_p, enst_p, ke_r, enst_r, metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_h5 = PROJECT_ROOT / "data" / "test_data_64.h5"
    weights_path = PROJECT_ROOT / "checkpoints" / "fno3d_weights_latest.pth"

    fig_dir = PROJECT_ROOT / "results" / "figures"
    bench_dir = PROJECT_ROOT / "results" / "benchmarks"
    fig_dir.mkdir(parents=True, exist_ok=True)
    bench_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(test_h5, "r") as f:
        fdns_traj = f["velocity"][0]  # (101, 64, 64, 64, 3)

    u0_64 = fdns_traj[0]
    num_steps = fdns_traj.shape[0] - 1

    # 1. Benchmark single-trajectory Filtered DNS (256^3 -> 64^3) runtime
    print("Benchmarking Case 1: Filtered DNS (256^3 -> 64^3, 1 trajectory)...")
    _, fdns_time_1traj, fdns_vram_mb = run_pseudospectral_rk4_single(
        N_sim=256,
        N_save=64,
        phase=(0.0, 0.0, 0.0),
        T=10.0,
        dt=0.01,
        snapshot_dt=0.1,
        nu=0.000625,
        record_snapshots=True,
    )

    # 2. Benchmark & evaluate 3D FNO Surrogate (64^3)
    print("Benchmarking Case 2: 3D FNO Surrogate (64^3, modes=24, width=20)...")
    fno_traj, fno_time, fno_vram_mb = run_fno_rollout(
        weights_path, u0_64, num_steps=num_steps, device=device, modes=24, width=20
    )
    fno_ke, fno_enst, fdns_ke, fdns_enst, fno_metrics = compute_series_and_metrics(
        fno_traj, fdns_traj, device
    )

    # 3. Benchmark & evaluate Native Coarse Solver (64^3)
    print("Benchmarking Case 3: Native Pseudo-Spectral Solver (64^3)...")
    nat64_traj, nat64_time, nat64_vram_mb = run_pseudospectral_rk4_single(
        N_sim=64,
        N_save=64,
        u0_init=u0_64,
        T=10.0,
        dt=0.01,
        snapshot_dt=0.1,
        nu=0.000625,
        record_snapshots=True,
    )
    nat64_ke, nat64_enst, _, _, nat64_metrics = compute_series_and_metrics(
        nat64_traj, fdns_traj, device
    )

    # 4. Save JSON & Markdown Benchmark Tables
    records = [
        {
            "case": "Filtered DNS (256^3 -> 64^3)",
            "wall_time_s": round(fdns_time_1traj, 3),
            "ms_per_snap": round((fdns_time_1traj / num_steps) * 1000, 2),
            "speedup_vs_256_dns": 1.0,
            "peak_mem_mb": round(fdns_vram_mb, 1),
            "u_rel_l2_pct": 0.0,
            "ke_rel_l2_pct": 0.0,
            "enst_rel_l2_pct": 0.0,
            "ke_final_err_pct": 0.0,
            "enst_peak_err_pct": 0.0,
        },
        {
            "case": "3D FNO Surrogate (64^3)",
            "wall_time_s": round(fno_time, 3),
            "ms_per_snap": round((fno_time / num_steps) * 1000, 2),
            "speedup_vs_256_dns": round(fdns_time_1traj / fno_time, 1),
            "peak_mem_mb": round(fno_vram_mb, 1),
            **fno_metrics,
        },
        {
            "case": "Native Solver (64^3)",
            "wall_time_s": round(nat64_time, 3),
            "ms_per_snap": round((nat64_time / num_steps) * 1000, 2),
            "speedup_vs_256_dns": round(fdns_time_1traj / nat64_time, 1),
            "peak_mem_mb": round(nat64_vram_mb, 1),
            **nat64_metrics,
        },
    ]

    json_path = bench_dir / "benchmark_summary.json"
    md_path = bench_dir / "benchmark_summary.md"

    with open(json_path, "w") as jf:
        json.dump(records, jf, indent=2)

    lines = [
        "| Case | Wall Time (s) | ms / Snapshot | Speedup vs. $256^3$ DNS | Peak Mem (MB) | Rel. $L_2$ $\\mathbf{u}$ (%) | Rel. $L_2$ $E_k$ (%) | Rel. $L_2$ $\\mathcal{E}$ (%) | $E_k(T)$ Err (%) | $\\mathcal{E}_{\\max}$ Err (%) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in records:
        lines.append(
            f"| {r['case']} | {r['wall_time_s']} | {r['ms_per_snap']} | {r['speedup_vs_256_dns']}x | "
            f"{r['peak_mem_mb']} | {r['u_rel_l2_pct']} | {r['ke_rel_l2_pct']} | {r['enst_rel_l2_pct']} | "
            f"{r['ke_final_err_pct']} | {r['enst_peak_err_pct']} |"
        )

    with open(md_path, "w") as mf:
        mf.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))

    # 5. Generate Three-Way Comparison Plot (fig4_native64_vs_fno_fdns.png)
    times = np.linspace(0.0, 10.0, fdns_traj.shape[0])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(times, fdns_ke, "k-", lw=2.0, label=r"Filtered DNS ($256^3 \rightarrow 64^3$)")
    ax1.plot(times, fno_ke, "r--", lw=2.0, label=r"3D FNO Surrogate ($64^3$)")
    ax1.plot(times, nat64_ke, color="tab:blue", linestyle="-.", lw=1.8, label=r"Native Solver ($64^3$)")
    ax1.set_title("Kinetic Energy Decay vs. Time")
    ax1.set_xlabel("Time $t$ (s)")
    ax1.set_ylabel(r"Kinetic Energy $E_k(t)$")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(times, fdns_enst, "k-", lw=2.0, label=r"Filtered DNS ($256^3 \rightarrow 64^3$)")
    ax2.plot(times, fno_enst, "r--", lw=2.0, label=r"3D FNO Surrogate ($64^3$)")
    ax2.plot(times, nat64_enst, color="tab:blue", linestyle="-.", lw=1.8, label=r"Native Solver ($64^3$)")
    ax2.set_title("Enstrophy Evolution vs. Time")
    ax2.set_xlabel("Time $t$ (s)")
    ax2.set_ylabel(r"Enstrophy $\mathcal{E}(t)$")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig_path = fig_dir / "fig4_native64_vs_fno_fdns.png"
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)

    print(f"\nSaved Figure   -> {fig_path}")
    print(f"Saved JSON     -> {json_path}")
    print(f"Saved Markdown -> {md_path}")


if __name__ == "__main__":
    main()
