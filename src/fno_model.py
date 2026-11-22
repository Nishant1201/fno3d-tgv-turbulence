
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = (1 / (in_channels * out_channels))
        
        self.w1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, modes, modes, dtype=torch.cfloat))
        self.w2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, modes, modes, dtype=torch.cfloat))
        self.w3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, modes, modes, dtype=torch.cfloat))
        self.w4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, modes, modes, dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        x = x.float() 
        batchsize = x.shape[0]
        
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

        m = self.modes
        out_ft[:, :, :m, :m, :m] = self.compl_mul3d(x_ft[:, :, :m, :m, :m], self.w1)
        out_ft[:, :, -m:, :m, :m] = self.compl_mul3d(x_ft[:, :, -m:, :m, :m], self.w2)
        out_ft[:, :, :m, -m:, :m] = self.compl_mul3d(x_ft[:, :, :m, -m:, :m], self.w3)
        out_ft[:, :, -m:, -m:, :m] = self.compl_mul3d(x_ft[:, :, -m:, -m:, :m], self.w4)

        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class FNO3d(nn.Module):
    def __init__(self, modes, width):
        super(FNO3d, self).__init__()
        self.width = width
        self.p = nn.Linear(6, self.width)
        self.conv0 = SpectralConv3d(self.width, self.width, modes)
        self.conv1 = SpectralConv3d(self.width, self.width, modes)
        self.conv2 = SpectralConv3d(self.width, self.width, modes)
        self.conv3 = SpectralConv3d(self.width, self.width, modes)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.q1 = nn.Linear(self.width, 64)
        self.q2 = nn.Linear(64, 3) 

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x).permute(0, 4, 1, 2, 3) 
        
        x = checkpoint(lambda v: F.gelu(self.conv0(v) + self.w0(v)), x, use_reentrant=False)
        x = checkpoint(lambda v: F.gelu(self.conv1(v) + self.w1(v)), x, use_reentrant=False)
        x = checkpoint(lambda v: F.gelu(self.conv2(v) + self.w2(v)), x, use_reentrant=False)
        x = self.conv3(x) + self.w3(x)
        
        x = x.permute(0, 2, 3, 4, 1) 
        x = F.gelu(self.q1(x))
        x = self.q2(x)
        return x

    def get_grid(self, shape, device):
        b, x, y, z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, x), dtype=torch.float32).reshape(1, x, 1, 1, 1).repeat([b, 1, y, z, 1])
        gridy = torch.tensor(np.linspace(0, 1, y), dtype=torch.float32).reshape(1, 1, y, 1, 1).repeat([b, x, 1, z, 1])
        gridz = torch.tensor(np.linspace(0, 1, z), dtype=torch.float32).reshape(1, 1, 1, z, 1).repeat([b, x, y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
