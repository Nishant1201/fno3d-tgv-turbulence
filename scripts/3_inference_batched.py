
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from fno_model import FNO3d

def compute_physics_metrics(u_tensor):
    N = u_tensor.shape[1]
    u = u_tensor.permute(0, 4, 1, 2, 3) 
    
    ke = 0.5 * torch.mean(torch.sum(u * u, dim=1))
    
    kx = torch.fft.fftfreq(N, 1.0/N, device=u.device)
    ky = kx
    kz = torch.fft.rfftfreq(N, 1.0/N, device=u.device)
    Kx, Ky, Kz = torch.meshgrid(kx, ky, kz, indexing='ij')
    
    u_hat = torch.fft.rfftn(u, dim=(2, 3, 4))
    
    curl_x = torch.fft.irfftn(1j * (Ky * u_hat[:, 2] - Kz * u_hat[:, 1]), s=(N, N, N))
    curl_y = torch.fft.irfftn(1j * (Kz * u_hat[:, 0] - Kx * u_hat[:, 2]), s=(N, N, N))
    curl_z = torch.fft.irfftn(1j * (Kx * u_hat[:, 1] - Ky * u_hat[:, 0]), s=(N, N, N))
    
    enstrophy = 0.5 * torch.mean(curl_x**2 + curl_y**2 + curl_z**2)
    return ke.item(), enstrophy.item()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FNO3d(modes=24, width=20).to(device)
    model.load_state_dict(torch.load("fno3d_weights.pth", weights_only=True))
    model.eval()
    print("Model loaded successfully.")

    times = np.arange(0, 10.0 + 1e-5, 0.1)
    fno_ke, fno_enst = [], []
    true_ke, true_enst = [], []
    
    print("Running Autoregressive Inference at 64^3 (Zero-Shot Super-Resolution)...")
    
    # 10 timesteps = ~2.01 GB of System RAM, perfectly safe for a 16 GB machine
    CHUNK_SIZE = 10 
    
    with h5py.File("test_data_64.h5", 'r') as f:
        test_data_ref = f['velocity']
        num_timesteps = test_data_ref.shape[1]
        
        with torch.no_grad():
            u_curr = torch.tensor(test_data_ref[0, 0]).unsqueeze(0).to(device) 
            
            ke_0, enst_0 = compute_physics_metrics(u_curr)
            fno_ke.append(ke_0); fno_enst.append(enst_0)
            true_ke.append(ke_0); true_enst.append(enst_0)
            
            # Iterate over the disk data in chunks to optimize I/O
            for chunk_start in range(1, num_timesteps, CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, num_timesteps)
                
                print(f"Loading chunk {chunk_start} to {chunk_end-1} from disk into System RAM...")
                
                # 1. HDF5 to System RAM (Batched, fast I/O)
                true_chunk_cpu = torch.tensor(test_data_ref[0, chunk_start:chunk_end])
                
                # 2. System RAM to VRAM (Sequential, stays under 8GB limit)
                for i in range(chunk_end - chunk_start):
                    #with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        #u_pred = u_curr + model(u_curr)
                    # replace the autocast block with direct float32 evaluation:
                    u_pred = u_curr + model(u_curr)
                    ke_pred, enst_pred = compute_physics_metrics(u_pred.float())
                    fno_ke.append(ke_pred)
                    fno_enst.append(enst_pred)
                    
                    u_true = true_chunk_cpu[i].unsqueeze(0).to(device)
                    ke_t, enst_t = compute_physics_metrics(u_true)
                    true_ke.append(ke_t)
                    true_enst.append(enst_t)
                    
                    u_curr = u_pred 
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Update legends in LaTeX format to reflect 256^3 DNS truncated to 64^3
    ax1.plot(times, true_ke, 'k-', label=r'CuPy Ground Truth ($256^3 \rightarrow 64^3$)')
    ax1.plot(times, fno_ke, 'r--', label=r'FNO Prediction ($64^3$)')
    ax1.set_title('Kinetic Energy Decay vs Time')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Kinetic Energy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(times, true_enst, 'k-', label=r'CuPy Ground Truth ($256^3 \rightarrow 64^3$)')
    ax2.plot(times, fno_enst, 'r--', label=r'FNO Prediction ($64^3$)')
    ax2.set_title('Enstrophy Evolution vs Time')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Enstrophy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("fno_inference_64_batched.png", dpi=300)
    print("Done! Evaluation saved to fno_inference_64_batched.png")
