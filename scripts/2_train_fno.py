import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py

# Ensure project root is in sys.path regardless of working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fno_model import FNO3d

class HDF5TGVDataset(Dataset):
    def __init__(self, h5_filepath):
        self.filepath = str(h5_filepath)
        with h5py.File(self.filepath, 'r') as f:
            self.samples = f['velocity'].shape[0]
            self.timesteps = f['velocity'].shape[1]
            self.sequences_per_sample = self.timesteps - 2
            
    def __len__(self):
        return self.samples * self.sequences_per_sample

    def __getitem__(self, idx):
        s = idx // self.sequences_per_sample
        t = idx % self.sequences_per_sample
        
        with h5py.File(self.filepath, 'r') as f:
            x = f['velocity'][s, t]
            y1 = f['velocity'][s, t+1]
            y2 = f['velocity'][s, t+2]
            
        return torch.tensor(x), torch.tensor(y1), torch.tensor(y2)

def log_message(msg, log_filepath):
    print(msg, flush=True)
    with open(log_filepath, "a") as lf:
        lf.write(msg + "\n")

if __name__ == "__main__":
    data_dir = PROJECT_ROOT / "data"
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    log_dir = PROJECT_ROOT / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "training_logs.txt"
    latest_ckpt = ckpt_dir / "fno3d_checkpoint_latest.pth"
    latest_weights = ckpt_dir / "fno3d_weights_latest.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_message(f"Training on: {device}", log_file)
    
    dataset = HDF5TGVDataset(data_dir / "train_data.h5")
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)
    
    model = FNO3d(modes=24, width=20).to(device)
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    target_epochs = 200
    start_epoch = 0

    if latest_ckpt.exists():
        log_message(f"Found checkpoint '{latest_ckpt.name}'. Resuming training...", log_file)
        checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch']
        log_message(f"Successfully resumed from epoch {start_epoch}. Target is {target_epochs} epochs.", log_file)
    else:
        log_message("No checkpoint found. Starting training from scratch...", log_file)

    model.train()
    
    for ep in range(start_epoch, target_epochs):
        train_loss = 0.0
        for x, y1, y2 in train_loader:
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            x.requires_grad_(True)
            optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                pred1 = x + model(x)
                loss1 = F.mse_loss(pred1, y1)
                
                pred2 = pred1 + model(pred1)
                loss2 = F.mse_loss(pred2, y2)
                
                loss = loss1 + loss2
                
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        avg_loss = train_loss / len(train_loader)
        log_message(f"Epoch {ep+1}/{target_epochs} | Loss: {avg_loss:.6f}", log_file)

        # Save both latest and epoch-numbered checkpoints every 50 epochs
        if (ep + 1) % 50 == 0 or (ep + 1) == target_epochs:
            state_dict = {
                'epoch': ep + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict()
            }
            # 1. Save latest pointers for seamless resumption & inference
            torch.save(state_dict, latest_ckpt)
            torch.save(model.state_dict(), latest_weights)
            
            # 2. Save historical epoch-stamped snapshots automatically
            epoch_ckpt = ckpt_dir / f"fno3d_checkpoint_epoch{ep+1}.pth"
            epoch_weights = ckpt_dir / f"fno3d_weights_epoch{ep+1}.pth"
            torch.save(state_dict, epoch_ckpt)
            torch.save(model.state_dict(), epoch_weights)
            
            log_message(f"--> Saved latest & epoch-{ep+1} checkpoints to {ckpt_dir}", log_file)

# Legacy version of the training script for reference and backward compatibility
'''
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
from fno_model import FNO3d

class HDF5TGVDataset(Dataset):
    def __init__(self, h5_filepath):
        self.filepath = h5_filepath
        with h5py.File(self.filepath, 'r') as f:
            self.samples = f['velocity'].shape[0]
            self.timesteps = f['velocity'].shape[1]
            self.sequences_per_sample = self.timesteps - 2
            
    def __len__(self):
        return self.samples * self.sequences_per_sample

    def __getitem__(self, idx):
        s = idx // self.sequences_per_sample
        t = idx % self.sequences_per_sample
        
        with h5py.File(self.filepath, 'r') as f:
            x = f['velocity'][s, t]
            y1 = f['velocity'][s, t+1]
            y2 = f['velocity'][s, t+2]
            
        return torch.tensor(x), torch.tensor(y1), torch.tensor(y2)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    dataset = HDF5TGVDataset("train_data.h5")
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)
    
    model = FNO3d(modes=24, width=20).to(device)
    
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # ---------------------------------------------------------
    # RESUME LOGIC IMPLEMENTATION
    # ---------------------------------------------------------
    target_epochs = 200  # Change this to 100, 150, etc., to train in chunks
    start_epoch = 0
    checkpoint_file = "fno3d_checkpoint.pth"

    if os.path.exists(checkpoint_file):
        print(f"Found checkpoint '{checkpoint_file}'. Resuming training...")
        # weights_only=False is required to load optimizer/scheduler states safely
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch']
        
        print(f"Successfully resumed from epoch {start_epoch}. Target is {target_epochs} epochs.")
    else:
        print("No checkpoint found. Starting training from scratch...")

    model.train()
    
    for ep in range(start_epoch, target_epochs):
        train_loss = 0.0
        for x, y1, y2 in train_loader:
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            
            x.requires_grad_(True)
            optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                pred1 = x + model(x)
                loss1 = F.mse_loss(pred1, y1)
                
                pred2 = pred1 + model(pred1)
                loss2 = F.mse_loss(pred2, y2)
                
                loss = loss1 + loss2
                
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        #print(f"Epoch {ep+1}/{target_epochs} | Loss: {train_loss/len(train_loader):.6f}")
        print(f"Epoch {ep+1}/{target_epochs} | Loss: {train_loss/len(train_loader):.4e}")

        # Save checkpoint every 50 epochs AND at the final target epoch
        if (ep + 1) % 50 == 0 or (ep + 1) == target_epochs:
            torch.save({
                'epoch': ep + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict()
            }, checkpoint_file)
            
            # Save the clean weights file so 3_inference_batched.py works without modifications
            torch.save(model.state_dict(), "fno3d_weights.pth")
            print(f"--> Saved checkpoint and weights at epoch {ep+1}")

'''