
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple 20-joint Kinect-v1 graph for UTD-MHAD original.
BONES_20 = [
    (0,1),(1,2),(2,3),
    (2,4),(4,5),(5,6),(6,7),
    (2,8),(8,9),(9,10),(10,11),
    (0,12),(12,13),(13,14),(14,15),
    (0,16),(16,17),(17,18),(18,19),
]

def adjacency20():
    A = torch.eye(20)
    for i, j in BONES_20:
        A[i,j] = 1
        A[j,i] = 1
    d = A.sum(1).clamp_min(1)
    D = torch.diag(d.pow(-0.5))
    return D @ A @ D

class STGCNBlock(nn.Module):
    def __init__(self, c1, c2, drop=0.1):
        super().__init__()
        self.register_buffer("A", adjacency20())
        self.gcn = nn.Conv2d(c1, c2, 1, bias=False)
        self.tcn = nn.Conv2d(c2, c2, (5,1), padding=(2,0), bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.drop = nn.Dropout(drop)
        self.act = nn.ReLU(inplace=True)
        self.res = nn.Identity() if c1 == c2 else nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )

    def forward(self, x):
        r = self.res(x)
        x = torch.einsum("vw,bctw->bctv", self.A, x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = self.drop(self.bn(x))
        return self.act(x + r)

class SkeletonEncoder(nn.Module):
    def __init__(self, dim=256, tokens=8):
        super().__init__()
        self.net = nn.Sequential(
            STGCNBlock(3,64),
            STGCNBlock(64,128),
            STGCNBlock(128,dim),
        )
        self.tokens = tokens

    def forward(self, x):
        # [B,T,20,3] -> [B,3,T,20]
        x = x.permute(0,3,1,2)
        x = self.net(x).mean(-1)  # [B,D,T]
        x = F.adaptive_avg_pool1d(x, self.tokens)
        return x.transpose(1,2)    # [B,8,D]

class IMUEncoder(nn.Module):
    def __init__(self, dim=256, tokens=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(6,64,7,padding=3),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64,128,5,padding=2),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128,dim,3,padding=1),
            nn.BatchNorm1d(dim), nn.GELU(),
        )
        self.tokens = tokens

    def forward(self, x):
        x = self.net(x.transpose(1,2))
        x = F.adaptive_avg_pool1d(x, self.tokens)
        return x.transpose(1,2)

class SkeletonOnly(nn.Module):
    def __init__(self, nc=27, dim=256):
        super().__init__()
        self.enc = SkeletonEncoder(dim)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim,nc))
    def forward(self, sk, imu):
        z = self.enc(sk)
        return {"logits": self.head(z.mean(1))}

class IMUOnly(nn.Module):
    def __init__(self, nc=27, dim=256):
        super().__init__()
        self.enc = IMUEncoder(dim)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim,nc))
    def forward(self, sk, imu):
        z = self.enc(imu)
        return {"logits": self.head(z.mean(1))}

class LateFusionSI(nn.Module):
    def __init__(self, nc=27, dim=256):
        super().__init__()
        self.sk = SkeletonEncoder(dim)
        self.im = IMUEncoder(dim)
        self.head = nn.Sequential(
            nn.LayerNorm(dim*2),
            nn.Linear(dim*2, dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim, nc),
        )
    def forward(self, sk, imu):
        s = self.sk(sk).mean(1)
        i = self.im(imu).mean(1)
        return {"logits": self.head(torch.cat([s,i], -1))}

class TemporalAlignSI(nn.Module):
    def __init__(self, dim=256, heads=4):
        super().__init__()
        self.ns = nn.LayerNorm(dim)
        self.ni = nn.LayerNorm(dim)
        self.s2i = nn.MultiheadAttention(dim,heads,batch_first=True,dropout=0.1)
        self.i2s = nn.MultiheadAttention(dim,heads,batch_first=True,dropout=0.1)

    def forward(self, s, i):
        ds, A_si = self.s2i(self.ns(s), self.ni(i), i, need_weights=True, average_attn_weights=True)
        di, A_is = self.i2s(self.ni(i), self.ns(s), s, need_weights=True, average_attn_weights=True)
        return s + ds, i + di, A_si, A_is

class ReliabilitySI(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        h = max(64, dim//2)
        self.s = nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,h),nn.GELU(),nn.Linear(h,1))
        self.i = nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,h),nn.GELU(),nn.Linear(h,1))
    def forward(self, s, i):
        return self.s(s), self.i(i)

class RATAMAR_SI(nn.Module):
    def __init__(self, nc=27, dim=256, heads=4, layers=2, temp=0.7):
        super().__init__()
        self.sk = SkeletonEncoder(dim)
        self.im = IMUEncoder(dim)
        self.align = TemporalAlignSI(dim, heads)
        self.rel = ReliabilitySI(dim)

        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim*4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim,nc))
        self.temp = temp

    def forward(self, sk, imu):
        s = self.sk(sk)
        i = self.im(imu)

        s, i, A_si, A_is = self.align(s, i)

        rs, ri = self.rel(s, i)
        rel_logits = torch.cat([rs,ri], dim=-1)
        alpha = torch.softmax(rel_logits / self.temp, dim=-1)

        fused = alpha[...,0:1] * s + alpha[...,1:2] * i
        fused = self.temporal(fused)

        return {
            "logits": self.head(fused.mean(1)),
            "r_skel": torch.sigmoid(rs),
            "r_imu": torch.sigmoid(ri),
            "alpha": alpha,
            "align_skel_to_imu": A_si,
            "align_imu_to_skel": A_is,
        }
