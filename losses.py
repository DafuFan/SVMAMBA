import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim


class SobelMag(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)[None,None]
        ky = torch.tensor([[ 1,2,1],[ 0,0,0],[-1,-2,-1]], dtype=torch.float32)[None,None]
        self.register_buffer('kx', kx)  # (1,1,3,3)
        self.register_buffer('ky', ky)
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,1,H,W]
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.abs(gx) + torch.abs(gy)

class Fusionloss(nn.Module):

    def __init__(self, data_range: float = 1.0, beta: float = 8.0, gamma: float = 1.0):
        super().__init__()
        self.grad = SobelMag()
        self.data_range = data_range
        self.beta = beta      
        self.gamma = gamma    

    def forward(self, fused: torch.Tensor, vis: torch.Tensor, ir: torch.Tensor):
        F = fused[:, :1, :, :]
        V = vis[:,   :1, :, :]
        I = ir[:,    :1, :, :]

        T = torch.max(V, I)
        L_pix = (F - T).abs().mean()

        Gf = self.grad(F)
        Gv = self.grad(V)
        Gi = self.grad(I)
        Tgrad = torch.max(Gv, Gi)
        L_grad = (Gf - Tgrad).abs().mean()


        S_v = 1.0 - ms_ssim(F, V, data_range=self.data_range, size_average=True)
        S_i = 1.0 - ms_ssim(F, I, data_range=self.data_range, size_average=True)
        S_struct = 0.5 * (S_v + S_i)

        total = L_pix + self.beta * L_grad + self.gamma * S_struct

        logs = {
            "total": total.detach(),
            "L_pix": L_pix.detach(),
            "L_grad": L_grad.detach(),
            "S_struct": S_struct.detach(),
            "S_v": S_v.detach(),
            "S_i": S_i.detach(),
        }
        return total, logs