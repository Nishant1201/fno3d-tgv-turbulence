
import os
import math
import torch
import numpy as np
import cupy as cp
import h5py

def spectral_truncate(u_high, N_low, N_sim):
    if u_high.shape[1] == N_low:
        return u_high.astype(cp.float32)
    
    # Cast to complex64 for FFT inside truncation
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
    return (u_low * (N_low / N_sim)**3).astype(cp.float32)

def generate_tgv_to_hdf5(filepath, num_samples, N_sim=256, N_save=64, T=10.0, dt=0.01, snapshot_dt=0.1, nu=0.000625):
    print(f"Generating data into {filepath} | Sim Res: {N_sim}^3 (float64) | Save Res: {N_save}^3 (float32)")
    
    timesteps = int(round(T / snapshot_dt)) + 1
    
    with h5py.File(filepath, 'w') as f:
        dataset = f.create_dataset('velocity', 
                                   shape=(num_samples, timesteps, N_save, N_save, N_save, 3), 
                                   dtype='float32', 
                                   chunks=(1, 1, N_save, N_save, N_save, 3),
                                   compression='lzf')
                                   
        x = cp.linspace(0, 2 * math.pi, N_sim, endpoint=False, dtype=cp.float64)
        X = cp.meshgrid(x, x, x, indexing='ij')
        
        kx = cp.fft.fftfreq(N_sim, 1.0 / N_sim).astype(cp.float64)
        ky = kx
        kz = cp.fft.rfftfreq(N_sim, 1.0 / N_sim).astype(cp.float64)
        K = cp.array(cp.meshgrid(kx, ky, kz, indexing='ij'))
        
        K2 = cp.sum(K * K, axis=0)
        K2_safe = cp.where(K2 == 0, 1.0, K2)
        K_over_K2 = K / K2_safe
        
        kmax_dealias = 2.0 / 3.0 * (N_sim / 2 + 1)
        dealias = (cp.abs(K[0]) < kmax_dealias) & (cp.abs(K[1]) < kmax_dealias) & (cp.abs(K[2]) < kmax_dealias)
        
        # High-precision arrays for physics solver
        U = cp.empty((3, N_sim, N_sim, N_sim), dtype=cp.float64)
        U_hat = cp.empty((3, N_sim, N_sim, N_sim//2 + 1), dtype=cp.complex128)
        U_hat0 = U_hat.copy()
        U_hat1 = U_hat.copy()
        dU = cp.empty_like(U_hat)
        curl = cp.empty_like(U)
        P_hat = cp.empty((N_sim, N_sim, N_sim//2 + 1), dtype=cp.complex128)
        
        for s in range(num_samples):
            print(f"  -> Simulating trajectory {s+1}/{num_samples}...")
            px, py, pz = cp.random.rand(3, dtype=cp.float64) * 2 * math.pi
            U[0] = cp.sin(X[0] + px) * cp.cos(X[1] + py) * cp.cos(X[2] + pz)
            U[1] = -cp.cos(X[0] + px) * cp.sin(X[1] + py) * cp.cos(X[2] + pz)
            U[2] = 0.0
            
            for i in range(3):
                U_hat[i] = cp.fft.rfftn(U[i], axes=(0, 1, 2))
                
            t = 0.0; tstep = 0; snap_idx = 0
            
            # Save t=0
            u_save = cp.asnumpy(spectral_truncate(U, N_save, N_sim)).transpose(1, 2, 3, 0)
            dataset[s, snap_idx] = u_save
            snap_idx += 1
            
            while t < T - 1e-5:
                t += dt; tstep += 1
                U_hat0[:] = U_hat #U_hat0 = U_hat.copy()
                U_hat1[:] = U_hat #U_hat1 = U_hat.copy()
                
                for rk in range(4):
                    if rk > 0:
                        for i in range(3): U[i] = cp.fft.irfftn(U_hat[i], axes=(0, 1, 2))
                    
                    curl[0] = cp.fft.irfftn(1j * (K[1]*U_hat[2] - K[2]*U_hat[1]), axes=(0,1,2))
                    curl[1] = cp.fft.irfftn(1j * (K[2]*U_hat[0] - K[0]*U_hat[2]), axes=(0,1,2))
                    curl[2] = cp.fft.irfftn(1j * (K[0]*U_hat[1] - K[1]*U_hat[0]), axes=(0,1,2))
                    
                    dU[0] = cp.fft.rfftn(U[1]*curl[2] - U[2]*curl[1], axes=(0,1,2))
                    dU[1] = cp.fft.rfftn(U[2]*curl[0] - U[0]*curl[2], axes=(0,1,2))
                    dU[2] = cp.fft.rfftn(U[0]*curl[1] - U[1]*curl[0], axes=(0,1,2))
                    
                    dU *= dealias
                    P_hat[:] = cp.sum(dU * K_over_K2, axis=0)
                    dU -= P_hat * K
                    dU -= nu * K2 * U_hat
                    
                    b_rk = [0.5, 0.5, 1.0, 0.0]
                    a_rk = [1./6., 1./3., 1./3., 1./6.]
                    
                    if rk < 3: U_hat[:] = U_hat0 + b_rk[rk] * dt * dU
                    U_hat1 += a_rk[rk] * dt * dU
                    
                U_hat[:] = U_hat1
                for i in range(3): U[i] = cp.fft.irfftn(U_hat[i], axes=(0, 1, 2))
                
                if tstep % int(round(snapshot_dt / dt)) == 0:
                    u_save = cp.asnumpy(spectral_truncate(U, N_save, N_sim)).transpose(1, 2, 3, 0)
                    dataset[s, snap_idx] = u_save
                    snap_idx += 1
            
            cp.get_default_memory_pool().free_all_blocks()

if __name__ == "__main__":
    generate_tgv_to_hdf5("train_data.h5", num_samples=10, N_sim=256, N_save=64, T=10.0)
    #generate_tgv_to_hdf5("test_data_256.h5", num_samples=1, N_sim=256, N_save=256, T=10.0)
    generate_tgv_to_hdf5("test_data_64.h5", num_samples=1, N_sim=256, N_save=64, T=10.0)
