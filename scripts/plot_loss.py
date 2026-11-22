import os
import re
import numpy as np
import matplotlib.pyplot as plt

def parse_training_logs(log_filepath):
    epochs = []
    losses = []
    
    # Matches lines formatted as: "Epoch 1/200 | Loss: 0.002127"
    pattern = re.compile(r"Epoch\s+(\d+)/\d+\s*\|\s*Loss:\s*([0-9.eE+-]+)")
    
    if not os.path.exists(log_filepath):
        raise FileNotFoundError(f"Log file not found at: {log_filepath}")
        
    with open(log_filepath, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                
    if not epochs:
        raise ValueError(f"No valid 'Epoch X/Y | Loss: Z' entries found in {log_filepath}.")
        
    return np.array(epochs), np.array(losses)

if __name__ == "__main__":
    log_path = os.path.join("logs", "training_logs.txt")
    output_plot = os.path.join("logs", "training_loss_curve.png")
    
    epochs, losses = parse_training_logs(log_path)
    print(f"Parsed {len(epochs)} epochs from {log_path} (Min Loss: {losses.min():.6f} at Epoch {epochs[losses.argmin()]})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Linear Scale Plot
    ax1.plot(epochs, losses, color='tab:blue', lw=2, label='Unrolled MSE Loss')
    ax1.set_title('Training Loss vs. Epochs (Linear Scale)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Unrolled Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # 2. Logarithmic Scale Plot (Highlights late-epoch convergence & StepLR drops)
    ax2.semilogy(epochs, losses, color='tab:red', lw=2, label='Unrolled MSE Loss (Log)')
    
    # Mark StepLR decay intervals (every 50 epochs)
    max_ep = epochs.max()
    for step_ep in range(50, max_ep + 1, 50):
        if step_ep < max_ep:
            ax2.axvline(
                x=step_ep, color='k', linestyle=':', alpha=0.7,
                label='LR Halved (StepLR)' if step_ep == 50 else None
            )

    ax2.set_title('Training Loss vs. Epochs (Log Scale)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(r'Total Unrolled Loss ($\log_{10}$)')
    ax2.grid(True, which="both", linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"Saved loss plot to {output_plot}")
