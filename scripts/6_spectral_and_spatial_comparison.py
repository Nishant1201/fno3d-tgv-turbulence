import sys
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
    "legend.fontsize": 9.5,
})


def run_native_64_solver(u0_64_np, T=10.0, dt=0.01, snapshot_dt=0.1, nu=0.000625):
    """
    Runs the pseudo-spectral RK4 solver natively at N = 64 (float64/complex128)
    starting from the given (64, 64, 64, 3) velocity field at t = 0.
    """
    N = u0_64_np.shape[0]
    timesteps = int(round(T / snapshot_dt)) + 1
    snap_interval = int(round(snapshot_dt / dt))

    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()

    kx = cp.fft.fftfreq(N, 1.0 / N).astype(cp.float64)
    ky = kx
    kz = cp.fft.rfftfreq(N, 1.0 / N).astype(cp.float64)
    K = cp.array(cp.meshgrid(kx, ky, kz, indexing="ij"))

    K2 = cp.sum(K * K, axis=0)
    K2_safe = cp.where(K2 == 0, 1.0, K2)
    K_over_K2 = K / K2_safe

    kmax_dealias = 2.0 / 3.0 * (N / 2 + 1)
    dealias = (
        (cp.abs(K[0]) < kmax_dealias)
        & (cp.abs(K[1]) < kmax_dealias)
        & (cp.abs(K[2]) < kmax_dealias)
    )

    U = cp.asarray(u0_64_np.transpose(3, 0, 1, 2), dtype=cp.float64)
    U_hat = cp.empty((3, N, N, N // 2 + 1), dtype=cp.complex128)
    for i in range(3):
        U_hat[i] = cp.fft.rfftn(U[i], axes=(0, 1, 2))

    U_hat0 = cp.empty_like(U_hat)
    U_hat1 = cp.empty_like(U_hat)
    dU = cp.empty_like(U_hat)
    curl = cp.empty_like(U)
    P_hat = cp.empty((N, N, N // 2 + 1), dtype=cp.complex128)

    snapshots = np.empty((timesteps, N, N, N, 3), dtype=np.float32)
    snapshots[0] = cp.asnumpy(U).transpose(1, 2, 3, 0).astype(np.float32)

    b_rk = [0.5, 0.5, 1.0, 0.0]
    a_rk = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]

    t = 0.0
    tstep = 0
    snap_idx = 1

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

        if tstep % snap_interval == 0 and snap_idx < timesteps:
            snapshots[snap_idx] = cp.asnumpy(U).transpose(1, 2, 3, 0).astype(np.float32)
            snap_idx += 1

    cp.cuda.Device().synchronize()
    del U, U_hat, U_hat0, U_hat1, dU, curl, P_hat, K, K2, K2_safe, K_over_K2, dealias
    mempool.free_all_blocks()
    return snapshots


def run_fno_rollout(weights_path, u0_64, num_steps, device, modes=24, width=20):
    model = FNO3d(modes=modes, width=width).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    pred_traj = np.empty((num_steps + 1, *u0_64.shape), dtype=np.float32)
    pred_traj[0] = u0_64

    with torch.no_grad():
        u_curr = torch.tensor(u0_64, dtype=torch.float32, device=device).unsqueeze(0)
        for step in range(1, num_steps + 1):
            u_curr = u_curr + model(u_curr)
            pred_traj[step] = u_curr.squeeze(0).cpu().numpy()

    return pred_traj


def compute_isotropic_energy_spectrum(u_frame_np, device):
    """
    Computes 3D shell-averaged isotropic kinetic energy spectrum E(k)
    for a single snapshot of shape (N, N, N, 3).
    """
    N = u_frame_np.shape[0]
    u = torch.tensor(u_frame_np, dtype=torch.float32, device=device).permute(3, 0, 1, 2)

    u_hat = torch.fft.fftn(u, dim=(1, 2, 3)) / (N**3)
    E_modal = 0.5 * torch.sum(torch.abs(u_hat) ** 2, dim=0)

    kx = torch.fft.fftfreq(N, d=1.0 / N, device=device)
    Kx, Ky, Kz = torch.meshgrid(kx, kx, kx, indexing="ij")
    K_mag = torch.sqrt(Kx**2 + Ky**2 + Kz**2)

    k_max = N // 2
    k_bins = np.arange(1, k_max + 1)
    E_k = np.zeros(k_max, dtype=np.float64)

    with torch.no_grad():
        for idx, k_val in enumerate(k_bins):
            mask = (K_mag >= (k_val - 0.5)) & (K_mag < (k_val + 0.5))
            E_k[idx] = torch.sum(E_modal[mask]).item()

    return k_bins, E_k


def compute_vorticity_and_q(u_frame_np, device):
    """
    Computes vorticity magnitude |omega| and Q-criterion on the full 3D grid
    using spectral velocity gradient tensor A_ij = du_i / dx_j.
    Input shape: (N, N, N, 3)
    """
    N = u_frame_np.shape[0]
    u = torch.tensor(u_frame_np, dtype=torch.float32, device=device).permute(3, 0, 1, 2)

    kx = torch.fft.fftfreq(N, d=1.0 / N, device=device)
    ky = kx
    kz = torch.fft.rfftfreq(N, d=1.0 / N, device=device)
    Kx, Ky, Kz = torch.meshgrid(kx, ky, kz, indexing="ij")
    K_vec = [Kx, Ky, Kz]

    with torch.no_grad():
        u_hat = torch.fft.rfftn(u, dim=(1, 2, 3))
        A = torch.zeros((3, 3, N, N, N), dtype=torch.float32, device=device)
        for i in range(3):
            for j in range(3):
                A[i, j] = torch.fft.irfftn(1j * K_vec[j] * u_hat[i], s=(N, N, N))

        curl_x = A[2, 1] - A[1, 2]
        curl_y = A[0, 2] - A[2, 0]
        curl_z = A[1, 0] - A[0, 1]

        omega_sq = curl_x**2 + curl_y**2 + curl_z**2
        omega_mag = torch.sqrt(omega_sq)

        S_sq = torch.zeros((N, N, N), dtype=torch.float32, device=device)
        for i in range(3):
            for j in range(3):
                S_ij = 0.5 * (A[i, j] + A[j, i])
                S_sq += S_ij**2

        Q_field = 0.25 * omega_sq - 0.5 * S_sq

    return omega_mag.cpu().numpy(), Q_field.cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_h5 = PROJECT_ROOT / "data" / "test_data_64.h5"
    weights_path = PROJECT_ROOT / "checkpoints" / "fno3d_weights_latest.pth"
    fig_dir = PROJECT_ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    MODES = 24
    WIDTH = 20

    with h5py.File(test_h5, "r") as f:
        fdns_traj = f["velocity"][0]  # (101, 64, 64, 64, 3)

    u0_64 = fdns_traj[0]
    num_steps = fdns_traj.shape[0] - 1

    print("Running 3D FNO Surrogate (64^3) rollout...")
    fno_traj = run_fno_rollout(weights_path, u0_64, num_steps=num_steps, device=device, modes=MODES, width=WIDTH)

    print("Running Native Pseudo-Spectral Solver (64^3)...")
    nat64_traj = run_native_64_solver(u0_64, T=10.0, dt=0.01, snapshot_dt=0.1, nu=0.000625)

    # =========================================================================
    # FIGURE 5: 3D Isotropic Energy Spectra E(k) Three-Way Comparison
    # =========================================================================
    spectrum_steps = {40: 4.0, 85: 8.5, 100: 10.0}
    fig5, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), sharey=True)

    for ax, (t_idx, t_val) in zip(axes, spectrum_steps.items()):
        k_bins, Ek_fdns = compute_isotropic_energy_spectrum(fdns_traj[t_idx], device)
        _, Ek_fno = compute_isotropic_energy_spectrum(fno_traj[t_idx], device)
        _, Ek_nat = compute_isotropic_energy_spectrum(nat64_traj[t_idx], device)

        valid_fdns = (Ek_fdns > 1e-7) & (k_bins >= 2)
        valid_fno = (Ek_fno > 1e-7) & (k_bins >= 2)
        valid_nat = (Ek_nat > 1e-7) & (k_bins >= 2)

        ax.loglog(
            k_bins[valid_fdns], Ek_fdns[valid_fdns],
            "k-", lw=2.0, label=r"Filtered DNS ($256^3 \rightarrow 64^3$)"
        )
        ax.loglog(
            k_bins[valid_fno], Ek_fno[valid_fno],
            "r--", lw=2.0, label=r"3D FNO Surrogate ($64^3$)"
        )
        ax.loglog(
            k_bins[valid_nat], Ek_nat[valid_nat],
            color="tab:blue", linestyle="-.", lw=1.7, label=r"Native Solver ($64^3$)"
        )

        # Reference Kolmogorov k^(-5/3) slope
        k_ref = np.array([3.0, 15.0])
        e_ref = Ek_fdns[2] * (k_ref / 3.0) ** (-5.0 / 3.0) * 1.8
        ax.loglog(k_ref, e_ref, "g:", lw=1.4, label=r"$k^{-5/3}$ Reference")

        ax.axvline(x=MODES, color="gray", linestyle=":", lw=1.4, label=rf"$k_{{\mathrm{{modes}}}} = {MODES}$")
        ax.set_title(f"Energy Spectrum at $t = {t_val:.1f}$ s")
        ax.set_xlabel(r"Wavenumber $k$")
        ax.set_xlim(1, 32)
        ax.set_ylim(1e-7, 2e-1)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    axes[0].set_ylabel(r"Energy Spectrum $E(k)$")
    axes[0].legend(loc="lower left")

    plt.tight_layout()
    fig5_path = fig_dir / "fig5_energy_spectrum_Ek_comparison.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close(fig5)
    print(f"Saved -> {fig5_path}")

    # =========================================================================
    # FIGURE 6: Spatial Vorticity |omega| and Q-Criterion at Peak Enstrophy (t = 8.5s)
    # =========================================================================
    contour_idx = 85  # t = 8.5 s
    mid_z = fdns_traj.shape[3] // 2

    w_fdns, q_fdns = compute_vorticity_and_q(fdns_traj[contour_idx], device)
    w_fno, q_fno = compute_vorticity_and_q(fno_traj[contour_idx], device)
    w_nat, q_nat = compute_vorticity_and_q(nat64_traj[contour_idx], device)

    w_fdns_z, w_fno_z, w_nat_z = w_fdns[:, :, mid_z], w_fno[:, :, mid_z], w_nat[:, :, mid_z]
    q_fdns_z, q_fno_z, q_nat_z = q_fdns[:, :, mid_z], q_fno[:, :, mid_z], q_nat[:, :, mid_z]

    fig6, axes = plt.subplots(2, 3, figsize=(15.5, 9.4))
    extent = [0, 2 * math.pi, 0, 2 * math.pi]

    # Shared color limits across the 3 cases for fair visual comparison
    w_vmax = max(w_fdns_z.max(), w_fno_z.max(), w_nat_z.max())
    q_lim = max(
        np.abs(q_fdns_z).max(),
        np.abs(q_fno_z).max(),
        np.abs(q_nat_z).max(),
    ) * 0.75

    # Row 1: Vorticity Magnitude |omega|
    w_panels = [
        (w_fdns_z, r"Filtered DNS ($256^3 \rightarrow 64^3$): $|\boldsymbol{\omega}|$"),
        (w_fno_z, r"3D FNO Surrogate ($64^3$): $|\boldsymbol{\omega}|$"),
        (w_nat_z, r"Native Solver ($64^3$): $|\boldsymbol{\omega}|$"),
    ]
    for col, (arr, title) in enumerate(w_panels):
        im = axes[0, col].imshow(arr.T, origin="lower", extent=extent, cmap="inferno", vmin=0, vmax=w_vmax)
        axes[0, col].set_title(f"{title} ($t=8.5$ s)")
        plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

    # Row 2: Q-Criterion
    q_panels = [
        (q_fdns_z, r"Filtered DNS ($256^3 \rightarrow 64^3$): $Q$-Criterion"),
        (q_fno_z, r"3D FNO Surrogate ($64^3$): $Q$-Criterion"),
        (q_nat_z, r"Native Solver ($64^3$): $Q$-Criterion"),
    ]
    for col, (arr, title) in enumerate(q_panels):
        im = axes[1, col].imshow(arr.T, origin="lower", extent=extent, cmap="RdBu_r", vmin=-q_lim, vmax=q_lim)
        axes[1, col].set_title(f"{title} ($t=8.5$ s)")
        plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")

    plt.tight_layout()
    fig6_path = fig_dir / "fig6_vorticity_Q_three_way.png"
    plt.savefig(fig6_path, dpi=300)
    plt.close(fig6)
    print(f"Saved -> {fig6_path}")


if __name__ == "__main__":
    main()
