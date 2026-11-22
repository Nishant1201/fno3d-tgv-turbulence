import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from fno_model import FNO3d

# Set publication-quality Matplotlib defaults
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 13
})

def compute_spectral_quantities(u_tensor):
    """
    Computes Kinetic Energy, Enstrophy, Vorticity Magnitude |omega|, 
    and Q-criterion in PyTorch using exact spectral derivatives.
    Input shape: (1, N, N, N, 3)
    """
    N = u_tensor.shape[1]
    u = u_tensor.permute(0, 4, 1, 2, 3).float()  # (1, 3, N, N, N)
    
    # 1. Volume-Averaged Kinetic Energy
    ke = 0.5 * torch.mean(torch.sum(u * u, dim=1)).item()
    
    # Wavenumber grid for spectral derivatives
    kx = torch.fft.fftfreq(N, d=1.0/N, device=u.device)
    ky = kx
    kz = torch.fft.rfftfreq(N, d=1.0/N, device=u.device)
    Kx, Ky, Kz = torch.meshgrid(kx, ky, kz, indexing='ij')
    K_vec = [Kx, Ky, Kz]
    
    u_hat = torch.fft.rfftn(u, dim=(2, 3, 4))
    
    # 2. Velocity Gradient Tensor A_ij = du_i / dx_j (shape: 3, 3, N, N, N)
    A = torch.zeros((3, 3, N, N, N), dtype=torch.float32, device=u.device)
    for i in range(3):
        for j in range(3):
            A[i, j] = torch.fft.irfftn(1j * K_vec[j] * u_hat[0, i], s=(N, N, N))
            
    # 3. Vorticity components and Enstrophy
    curl_x = A[2, 1] - A[1, 2]
    curl_y = A[0, 2] - A[2, 0]
    curl_z = A[1, 0] - A[0, 1]
    
    omega_sq = curl_x**2 + curl_y**2 + curl_z**2
    enstrophy = (0.5 * torch.mean(omega_sq)).item()
    omega_mag = torch.sqrt(omega_sq)
    
    # 4. Strain-rate tensor S_ij = 0.5 * (A_ij + A_ji) and Q-Criterion
    # Using general definition Q = 0.25 * |omega|^2 - 0.5 * S_ij * S_ij
    S_sq = torch.zeros((N, N, N), dtype=torch.float32, device=u.device)
    for i in range(3):
        for j in range(3):
            S_ij = 0.5 * (A[i, j] + A[j, i])
            S_sq += S_ij**2
            
    Q_field = 0.25 * omega_sq - 0.5 * S_sq
    
    return ke, enstrophy, omega_mag.cpu().numpy(), Q_field.cpu().numpy()

def compute_isotropic_energy_spectrum(u_tensor):
    """
    Computes the 3D shell-averaged isotropic kinetic energy spectrum E(k).
    Satisfies Parseval's theorem: sum(E(k)) == Total Kinetic Energy.
    Input shape: (1, N, N, N, 3)
    """
    N = u_tensor.shape[1]
    u = u_tensor.permute(0, 4, 1, 2, 3).float()  # (1, 3, N, N, N)
    
    # Full 3D FFT normalized by N^3
    u_hat = torch.fft.fftn(u, dim=(2, 3, 4)) / (N**3)
    
    # Pointwise modal kinetic energy density: 0.5 * |u_hat(k)|^2
    E_modal = 0.5 * torch.sum(torch.abs(u_hat[0])**2, dim=0)  # (N, N, N)
    
    kx = torch.fft.fftfreq(N, d=1.0/N, device=u.device)
    Kx, Ky, Kz = torch.meshgrid(kx, kx, kx, indexing='ij')
    K_mag = torch.sqrt(Kx**2 + Ky**2 + Kz**2)
    
    k_max = N // 2
    k_bins = np.arange(1, k_max + 1)
    E_k = np.zeros(k_max, dtype=np.float64)
    
    # Spherical shell integration: k - 0.5 <= |K| < k + 0.5
    for idx, k_val in enumerate(k_bins):
        mask = (K_mag >= (k_val - 0.5)) & (K_mag < (k_val + 0.5))
        E_k[idx] = torch.sum(E_modal[mask]).item()
        
    return k_bins, E_k

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    MODES = 24
    WIDTH = 20
    model = FNO3d(modes=MODES, width=WIDTH).to(device)
    model.load_state_dict(torch.load("fno3d_weights.pth", map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded FNO3d(modes={MODES}, width={WIDTH}) on {device}.")

    times = np.arange(0, 10.0 + 1e-5, 0.1)
    fno_ke, fno_enst = [], []
    true_ke, true_enst = [], []
    
    # Snapshot indices for spectral & spatial diagnostics: t = 4.0s (40), 8.5s (85), 10.0s (100)
    spectrum_steps = {40: 4.0, 85: 8.5, 100: 10.0}
    contour_step = 85  # t = 8.5s (Peak Enstrophy / Turbulent Dissipation)
    
    spectra_data = {}
    spatial_snapshots = {}
    
    CHUNK_SIZE = 10
    
    print("Executing Autoregressive Rollout vs. Filtered DNS (64^3)...")
    with h5py.File("test_data_64.h5", 'r') as f:
        test_data_ref = f['velocity']
        num_timesteps = test_data_ref.shape[1]
        
        with torch.no_grad():
            u_curr = torch.tensor(test_data_ref[0, 0]).unsqueeze(0).to(device)
            ke_0, enst_0, _, _ = compute_spectral_quantities(u_curr)
            fno_ke.append(ke_0); fno_enst.append(enst_0)
            true_ke.append(ke_0); true_enst.append(enst_0)
            
            for chunk_start in range(1, num_timesteps, CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, num_timesteps)
                true_chunk_cpu = torch.tensor(test_data_ref[0, chunk_start:chunk_end])
                
                for i in range(chunk_end - chunk_start):
                    t_idx = chunk_start + i
                    
                    # Full FP32 autoregressive rollout step
                    u_pred = u_curr + model(u_curr)
                    u_true = true_chunk_cpu[i].unsqueeze(0).to(device)
                    
                    ke_p, enst_p, omega_p, Q_p = compute_spectral_quantities(u_pred)
                    ke_t, enst_t, omega_t, Q_t = compute_spectral_quantities(u_true)
                    
                    fno_ke.append(ke_p); fno_enst.append(enst_p)
                    true_ke.append(ke_t); true_enst.append(enst_t)
                    
                    # Store isotropic energy spectra at target times
                    if t_idx in spectrum_steps:
                        k_bins, Ek_true = compute_isotropic_energy_spectrum(u_true)
                        _, Ek_pred = compute_isotropic_energy_spectrum(u_pred)
                        spectra_data[spectrum_steps[t_idx]] = (k_bins, Ek_true, Ek_pred)
                        
                    # Store z = pi mid-plane slices at peak dissipation (t = 8.5s)
                    if t_idx == contour_step:
                        mid_z = u_true.shape[3] // 2
                        spatial_snapshots['omega_true'] = omega_t[:, :, mid_z]
                        spatial_snapshots['omega_pred'] = omega_p[:, :, mid_z]
                        spatial_snapshots['Q_true'] = Q_t[:, :, mid_z]
                        spatial_snapshots['Q_pred'] = Q_p[:, :, mid_z]
                        
                    u_curr = u_pred

    # Compute summary relative L2 errors over the full trajectory
    true_ke_arr, fno_ke_arr = np.array(true_ke), np.array(fno_ke)
    true_enst_arr, fno_enst_arr = np.array(true_enst), np.array(fno_enst)
    rel_l2_ke = np.linalg.norm(true_ke_arr - fno_ke_arr) / np.linalg.norm(true_ke_arr) * 100
    rel_l2_enst = np.linalg.norm(true_enst_arr - fno_enst_arr) / np.linalg.norm(true_enst_arr) * 100
    print(f"Rollout Complete | Trajectory Relative L2 Error -> KE: {rel_l2_ke:.2f}% | Enstrophy: {rel_l2_enst:.2f}%")

    # =========================================================================
    # FIGURE 1: Temporal Evolution of Kinetic Energy & Enstrophy
    # =========================================================================
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(times, true_ke, 'k-', lw=2, label=r'Filtered DNS ($256^3 \rightarrow 64^3$)')
    ax1.plot(times, fno_ke, 'r--', lw=2, label=r'3D FNO Surrogate ($64^3$)')
    ax1.set_title('Kinetic Energy Decay vs. Time')
    ax1.set_xlabel('Time $t$ (s)')
    ax1.set_ylabel(r'Kinetic Energy $E_k(t)$')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(times, true_enst, 'k-', lw=2, label=r'Filtered DNS ($256^3 \rightarrow 64^3$)')
    ax2.plot(times, fno_enst, 'r--', lw=2, label=r'3D FNO Surrogate ($64^3$)')
    ax2.set_title('Enstrophy Evolution vs. Time')
    ax2.set_xlabel('Time $t$ (s)')
    ax2.set_ylabel(r'Enstrophy $\mathcal{E}(t)$')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("fig1_temporal_ke_enstrophy.png", dpi=300)
    plt.close(fig1)
    print("Saved -> fig1_temporal_ke_enstrophy.png")

    # =========================================================================
    # FIGURE 2: 3D Isotropic Energy Spectra E(k) across Flow Regimes
    # =========================================================================
    fig2, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    
    for ax, (t_val, (k_bins, Ek_true, Ek_pred)) in zip(axes, spectra_data.items()):
        # Filter out zero-energy dealiased shells for clean log-log plotting
        valid_true = (Ek_true > 1e-7) & (k_bins >= 2) #valid_true = Ek_true > 1e-12
        valid_pred = (Ek_pred > 1e-7) & (k_bins >= 2) #valid_pred = Ek_pred > 1e-12
        
        ax.loglog(k_bins[valid_true], Ek_true[valid_true], 'k-', lw=2, label=r'Filtered DNS ($256^3 \rightarrow 64^3$)')
        ax.loglog(k_bins[valid_pred], Ek_pred[valid_pred], 'r--', lw=2, label=r'3D FNO Surrogate ($64^3$)')
        
        # Reference Kolmogorov k^(-5/3) slope in the inertial range (k in [3, 15])
        k_ref = np.array([3.0, 15.0])
        e_ref = Ek_true[2] * (k_ref / 3.0)**(-5.0 / 3.0) * 1.8
        ax.loglog(k_ref, e_ref, 'b-.', lw=1.2, alpha=0.8, label=r'$k^{-5/3}$ Reference')
        
        # FNO Spectral Convolution Cutoff (k_modes = 24)
        ax.axvline(x=MODES, color='gray', linestyle=':', lw=1.5, label=rf'$k_{{\mathrm{{modes}}}} = {MODES}$')
        
        ax.set_title(f'Energy Spectrum at $t = {t_val:.1f}$ s')
        ax.set_xlabel(r'Wavenumber $k$')
        ax.set_xlim(1, 32)
        ax.set_ylim(1e-8, 2e-1)
        ax.grid(True, which="both", linestyle='--', alpha=0.5)
        
    axes[0].set_ylabel(r'Energy Spectrum $E(k)$')
    axes[0].legend(loc='lower left')
    
    plt.tight_layout()
    plt.savefig("fig2_energy_spectrum_Ek.png", dpi=300)
    plt.close(fig2)
    print("Saved -> fig2_energy_spectrum_Ek.png")

    # =========================================================================
    # FIGURE 3: Spatial Vorticity |omega| and Q-Criterion at Peak Dissipation
    # =========================================================================
    fig3, axes = plt.subplots(2, 3, figsize=(15, 9.2))
    extent = [0, 2 * np.pi, 0, 2 * np.pi]
    
    # Row 1: Vorticity Magnitude |omega|
    w_true = spatial_snapshots['omega_true']
    w_pred = spatial_snapshots['omega_pred']
    w_err = np.abs(w_true - w_pred)
    w_vmax = max(w_true.max(), w_pred.max())
    
    im00 = axes[0, 0].imshow(w_true.T, origin='lower', extent=extent, cmap='inferno', vmin=0, vmax=w_vmax)
    axes[0, 0].set_title(r'Filtered DNS: $|\boldsymbol{\omega}|$ ($t=8.5$ s)')
    plt.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    im01 = axes[0, 1].imshow(w_pred.T, origin='lower', extent=extent, cmap='inferno', vmin=0, vmax=w_vmax)
    axes[0, 1].set_title(r'3D FNO Surrogate: $|\boldsymbol{\omega}|$ ($t=8.5$ s)')
    plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    im02 = axes[0, 2].imshow(w_err.T, origin='lower', extent=extent, cmap='viridis', vmin=0)
    axes[0, 2].set_title(r'Absolute Error: $||\boldsymbol{\omega}_{\mathrm{fDNS}}| - |\boldsymbol{\omega}_{\mathrm{FNO}}||$')
    plt.colorbar(im02, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Row 2: Q-Criterion (Vortex Cores where Q > 0 vs Strain-Dominated where Q < 0)
    q_true = spatial_snapshots['Q_true']
    q_pred = spatial_snapshots['Q_pred']
    q_err = np.abs(q_true - q_pred)
    q_lim = max(np.abs(q_true).max(), np.abs(q_pred).max()) * 0.8
    
    im10 = axes[1, 0].imshow(q_true.T, origin='lower', extent=extent, cmap='RdBu_r', vmin=-q_lim, vmax=q_lim)
    axes[1, 0].set_title(r'Filtered DNS: $Q$-Criterion ($t=8.5$ s)')
    plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im11 = axes[1, 1].imshow(q_pred.T, origin='lower', extent=extent, cmap='RdBu_r', vmin=-q_lim, vmax=q_lim)
    axes[1, 1].set_title(r'3D FNO Surrogate: $Q$-Criterion ($t=8.5$ s)')
    plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    im12 = axes[1, 2].imshow(q_err.T, origin='lower', extent=extent, cmap='viridis', vmin=0)
    axes[1, 2].set_title(r'Absolute Error: $|Q_{\mathrm{fDNS}} - Q_{\mathrm{FNO}}|$')
    plt.colorbar(im12, ax=axes[1, 2], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')

    plt.tight_layout()
    plt.savefig("fig3_vorticity_and_Q_contours.png", dpi=300)
    plt.close(fig3)
    print("Saved -> fig3_vorticity_and_Q_contours.png")
