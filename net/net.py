import time
import math
from functools import partial
from typing import Optional, Callable
import torch
import torch.nn as nn
import numpy as np
import numbers
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import torch.utils.checkpoint as checkpoint
from math import log
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from net.u2net import U2NETP
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class PatchEmbed2D(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=32, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape
        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x


class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)   # 1×1 卷积
        self.bn   = nn.BatchNorm2d(planes)          # BN
        self.relu = nn.ReLU(inplace=True)           # ReLU

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=True)  # 有偏置

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.ln(to_3d(x)), h, w)


class MaskGuidedFusion(nn.Module):
    def __init__(self):
        super(MaskGuidedFusion, self).__init__()

    def forward(self, ir, vi, mask):
        mask = mask.float()
        fused = mask * ir + (1.0 - mask) * vi

        return fused

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)


        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y



    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class VSSBlockWithMask(nn.Module):
    def __init__(self, dim, mask_in_channels=1, d_state=16, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, use_ss2d=True, use_checkpoint=False):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln1     = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.dwconv  = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.linear_r = nn.Linear(dim, dim)
        self.ss2d = SS2D(d_model=dim) if use_ss2d else nn.Identity()
        self.ln2  = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.ln3  = nn.LayerNorm(dim)
        self.ffn1 = nn.Linear(dim, 4 * dim)
        self.act  = nn.GELU()
        self.ffn2 = nn.Linear(4 * dim, dim)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.mask_branch = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        res1 = x
        gate = self.mask_branch(mask) if mask is not None else 1.0

        y = x.permute(0, 2, 3, 1).contiguous()         
        y_ln1 = self.ln1(y)                            
        y_r = F.silu(self.linear_r(y_ln1))             

        y = self.linear1(y_ln1)
        y = y.permute(0, 3, 1, 2).contiguous()          
        y = self.dwconv(y)
        if isinstance(gate, torch.Tensor):
            y = y * gate                               
        y = F.silu(y)

        y = y.permute(0, 2, 3, 1).contiguous()          
        y = self.ss2d(y)
        y = y.permute(0, 3, 1, 2).contiguous()         
        if isinstance(gate, torch.Tensor):
            y = y * gate                                
        y = y.permute(0, 2, 3, 1).contiguous()        
        y = self.ln2(y)                               
        y = y * y_r
        y = self.linear2(y)
        y = y + res1.permute(0, 2, 3, 1)               
        y_ffn_in = self.ln3(y)
        y_ffn = self.ffn2(self.drop(self.act(self.ffn1(y_ffn_in))))
        y = y + self.drop(y_ffn)

        return y.permute(0, 3, 1, 2).contiguous()       

class VSSLayer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlockWithMask(
                dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True:  
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, mask=None, fused=None):
        stage_identity = x  
        for blk in self.blocks:
            if self.use_checkpoint:
                if mask is not None:
                    def fn(_x, _m):
                        out = blk(_x.permute(0, 3, 1, 2).contiguous(),
                                  _m.permute(0, 3, 1, 2).contiguous())
                        return out.permute(0, 2, 3, 1).contiguous()
                    x = checkpoint.checkpoint(fn, x, mask)
                else:
                    def fn(_x):
                        out = blk(_x.permute(0, 3, 1, 2).contiguous(), None)
                        return out.permute(0, 2, 3, 1).contiguous()
                    x = checkpoint.checkpoint(fn, x)
            else:
                if mask is not None:
                    out = blk(x.permute(0, 3, 1, 2).contiguous(),
                              mask.permute(0, 3, 1, 2).contiguous())
                else:
                    out = blk(x.permute(0, 3, 1, 2).contiguous(), None)
                x = out.permute(0, 2, 3, 1).contiguous()

        x = x + stage_identity

        if fused is not None:
            assert fused.shape == x.shape, f"fused {fused.shape} != x {x.shape}"
            x = x + fused

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class VSSLayer_up(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            upsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlockWithMask(
                dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True:  
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() 
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x, mask=None):
        if self.upsample is not None:
            x = self.upsample(x) 

        for blk in self.blocks:
            if self.use_checkpoint:
                if mask is not None:
                    def fn(_x, _m):
                        out = blk(
                            _x.permute(0, 3, 1, 2).contiguous(),  
                            _m.permute(0, 3, 1, 2).contiguous() 
                        )
                        return out.permute(0, 2, 3, 1).contiguous() 

                    x = checkpoint.checkpoint(fn, x, mask)
                else:
                    def fn(_x):
                        out = blk(_x.permute(0, 3, 1, 2).contiguous(), None)
                        return out.permute(0, 2, 3, 1).contiguous()

                    x = checkpoint.checkpoint(fn, x)
            else:
                if mask is not None:
                    out = blk(
                        x.permute(0, 3, 1, 2).contiguous(),
                        mask.permute(0, 3, 1, 2).contiguous()
                    )
                else:
                    out = blk(x.permute(0, 3, 1, 2).contiguous(), None)

                x = out.permute(0, 2, 3, 1).contiguous()  

        return x


class FusionModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        r = max(1, channels // reduction)
        C = channels

        self.ir_mlp = nn.Sequential(
            nn.Conv2d(2 * C, r, kernel_size=1, bias=False), 
            nn.ReLU(inplace=True),
            nn.Conv2d(r, C, kernel_size=1, bias=False)      
        )
        self.ir_sigmoid = nn.Sigmoid()

        self.vis_spatial = nn.Sequential(
            nn.Conv2d(2, C, kernel_size=7, padding=3, bias=False), 
            nn.ReLU(inplace=True),
            nn.Conv2d(C, 1, kernel_size=1, bias=False)            
        )
        self.vis_sigmoid = nn.Sigmoid()

        self.fuse_1x1   = nn.Conv2d(2 * C, C, kernel_size=1, bias=False)
        self.fuse_gate  = nn.Sigmoid()
        self.id_proj    = nn.Conv2d(2 * C, C, kernel_size=1, bias=False)
        
    def forward(self, x_ir: torch.Tensor, x_vis: torch.Tensor) -> torch.Tensor:
        ir_avg = F.adaptive_avg_pool2d(x_ir, 1)           
        ir_max = F.adaptive_max_pool2d(x_ir, 1)             
        ir_cat = torch.cat([ir_avg, ir_max], dim=1)          
        ir_w   = self.ir_sigmoid(self.ir_mlp(ir_cat))         
        f_ir   = x_ir * ir_w                                 
        vis_avg = torch.mean(x_vis, dim=1, keepdim=True)      
        vis_max, _ = torch.max(x_vis, dim=1, keepdim=True)    
        vis_cat = torch.cat([vis_avg, vis_max], dim=1)       
        vis_w   = self.vis_sigmoid(self.vis_spatial(vis_cat)) 
        f_vis   = x_vis * vis_w                          
        add = f_ir + f_vis                                   
        sub = torch.abs(f_ir - f_vis)                      
        z_in = torch.cat([add, sub], dim=1)               
        z   = self.fuse_1x1(z_in)                        
        g   = self.fuse_gate(z)                              
        idn = self.id_proj(z_in)                         
        out = z * g + idn                          
        return out

class MultiScaleFusion(nn.Module):
    def __init__(self, dims, reduction=4):
        super().__init__()
        self.num_layers = len(dims)
        self.fusion_blocks = nn.ModuleList([
            FusionModule(channels=dim, reduction=reduction)
            for dim in dims
        ])

    def forward(self, x_ir_list, x_vis_list):
        fused_list = []
        for fusion_block, ir_feat, vis_feat in zip(self.fusion_blocks, x_ir_list, x_vis_list):
            ir_feat = ir_feat.permute(0, 3, 1, 2).contiguous()
            vis_feat = vis_feat.permute(0, 3, 1, 2).contiguous()

            C_expected = fusion_block.ir_mlp[0].in_channels // 2  
            assert ir_feat.shape[1] == C_expected, f"IR channels {ir_feat.shape[1]} != {C_expected}"
            assert vis_feat.shape[1] == C_expected, f"VIS channels {vis_feat.shape[1]} != {C_expected}"
            fused = fusion_block(ir_feat, vis_feat)  
            fused = fused.permute(0, 2, 3, 1).contiguous()  
            fused_list.append(fused)
        return fused_list

    def fuse_stage(self, i, ir_bhwc, vis_bhwc):
        ir = ir_bhwc.permute(0, 3, 1, 2).contiguous()
        vi = vis_bhwc.permute(0, 3, 1, 2).contiguous()
        fused = self.fusion_blocks[i](ir, vi).permute(0, 2, 3, 1).contiguous()
        return fused


class VSSM(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[1, 1, 1, 1], depths_decoder=[1, 1, 1, 1],
                 dims=[32, 64, 128, 256], dims_decoder=[256, 128, 64, 32], d_state=16, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 u2net_weight_path="", freeze_u2net=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed1 = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                                         norm_layer=norm_layer if patch_norm else None)
        self.patch_embed2 = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                                         norm_layer=norm_layer if patch_norm else None)

        self.ape = False
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed1 = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            self.absolute_pos_embed2 = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed1, std=.02)
            trunc_normal_(self.absolute_pos_embed2, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers1 = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers1.append(layer)

        self.layers2 = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers2.append(layer)

        self.u2net = U2NETP(3, 1)  
        state_dict = torch.load(u2net_weight_path, map_location="cuda")
        self.u2net.load_state_dict(state_dict)
        if freeze_u2net:
            for param in self.u2net.parameters():
                param.requires_grad = False
            self.u2net.eval()

        self.mask_fusion = MaskGuidedFusion()
        self.mask_embed = PatchEmbed2D(patch_size=4, in_chans=1, embed_dim=32)
        self.mask_down1 = PatchMerging2D(dim=32)
        self.mask_down2 = PatchMerging2D(dim=64)
        self.mask_down3 = PatchMerging2D(dim=128)
        self.multi_fusion = MultiScaleFusion(dims=dims, reduction=4)

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_up.append(layer)

        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1] // 4, 1, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    def forward_features1(self, x, mask_list=None):
        skip_list = []
        x = self.patch_embed1(x)

        if self.ape:
            x = x + self.absolute_pos_embed1
        x = self.pos_drop(x)
        for i, layer in enumerate(self.layers1):
            skip_list.append(x)
            if mask_list is not None:
                x = layer(x, mask_list[i])
            else:
                x = layer(x)

        return x, skip_list

    def forward_features2(self, x, mask_list=None):
        skip_list = []
        x = self.patch_embed2(x)

        if self.ape:
            x = x + self.absolute_pos_embed2
        x = self.pos_drop(x)
        for i, layer in enumerate(self.layers2):
            skip_list.append(x)
            if mask_list is not None:
                x = layer(x, mask_list[i])
            else:
                x = layer(x)

        return x, skip_list

    def forward_features_both_online(self, image_vis, image_ir, mask_list=None):
        x1 = self.patch_embed1(image_vis) 
        x2 = self.patch_embed2(image_ir)  
        if self.ape:
            x1 = x1 + self.absolute_pos_embed1
            x2 = x2 + self.absolute_pos_embed2
        x1 = self.pos_drop(x1);
        x2 = self.pos_drop(x2)
        fused_skips = []  

        for i, (layer1, layer2) in enumerate(zip(self.layers1, self.layers2)):
            fused_i = self.multi_fusion.fuse_stage(i, x2, x1)
            m = mask_list[i] if mask_list is not None else None
            if i < self.num_layers - 1:
                x1 = layer1(x1, m, fused=fused_i)
                x2 = layer2(x2, m, fused=fused_i)
            else:
                x1 = layer1(x1, m)
                x2 = layer2(x2, m)
            fused_skips.append(fused_i)

        return x1, x2, fused_skips

    def forward_features_up(self, x, skip_list, mask_list=None):
        for inx, layer_up in enumerate(self.layers_up):
            if mask_list is not None:
                x = layer_up(x, mask_list[-(inx + 1)])
            else:
                x = layer_up(x)
            if inx > 0:
                skip = skip_list[-(inx + 1)] 
                assert skip.shape[-2:] == x.shape[-2:] and skip.shape[-1] == x.shape[-1], \
                    f"Skip {skip.shape} != X {x.shape} at up-stage {inx}"
                x = x + skip
        return x

    def forward_final(self, x):
        x = self.final_up(x)
        x = x.permute(0, 3, 1, 2)
        x = self.final_conv(x)
        return x

    def forward(self, image_vis, image_ir, r=0.2,):
        x_vis_origin = image_vis[:, :1]  
        x_ir_origin = image_ir[:, :1]
        mask = self.u2net(image_ir)[0].float()

        fused_gray = self.mask_fusion(x_ir_origin, x_vis_origin, mask) 
        m1 = self.mask_embed(fused_gray)
        m2 = self.mask_down1(m1)
        m3 = self.mask_down2(m2)
        m4 = self.mask_down3(m3)
        mask_list = [m1, m2, m3, m4]

        x_1 = image_vis[:, :1]
        x_2 = image_ir[:, :1]

        x1, x2, fused_skips = self.forward_features_both_online(image_vis, image_ir, mask_list)
        x = fused_skips[-1]
        x = self.forward_features_up(x, fused_skips, mask_list)
        x = self.forward_final(x) + x_1 + x_2
        x = torch.sigmoid(x)

        return x





