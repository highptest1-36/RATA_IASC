
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        A[i, j] = 1.0
        A[j, i] = 1.0
    d = A.sum(1).clamp_min(1.0)
    D = torch.diag(d.pow(-0.5))
    return D @ A @ D


class STGCNBlock(nn.Module):
    def __init__(self, c1, c2, dropout=0.1):
        super().__init__()
        self.register_buffer("A", adjacency20())
        self.gcn = nn.Conv2d(c1, c2, 1, bias=False)
        self.tcn = nn.Conv2d(c2, c2, (5, 1), padding=(2, 0), bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)
        self.res = nn.Identity() if c1 == c2 else nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )

    def forward(self, x):
        residual = self.res(x)
        x = torch.einsum("vw,bctw->bctv", self.A, x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = self.drop(self.bn(x))
        return self.act(x + residual)


class SkeletonEncoder(nn.Module):
    def __init__(self, dim=256, tokens=8):
        super().__init__()
        self.net = nn.Sequential(
            STGCNBlock(3, 64),
            STGCNBlock(64, 128),
            STGCNBlock(128, dim),
        )
        self.tokens = int(tokens)

    def forward(self, x):
        # x: [B,T,20,3] -> [B,3,T,20]
        x = x.permute(0, 3, 1, 2)
        x = self.net(x).mean(-1)                 # [B,D,T]
        x = F.adaptive_avg_pool1d(x, self.tokens)
        return x.transpose(1, 2)                 # [B,tokens,D]


class IMUEncoder(nn.Module):
    def __init__(self, dim=256, tokens=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(6, 64, 7, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 5, padding=2, bias=False),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, dim, 3, padding=1, bias=False),
            nn.BatchNorm1d(dim), nn.GELU(),
        )
        self.tokens = int(tokens)

    def forward(self, x):
        x = self.net(x.transpose(1, 2))           # [B,D,T]
        x = F.adaptive_avg_pool1d(x, self.tokens)
        return x.transpose(1, 2)                  # [B,tokens,D]


def _pool_scalar_sequence(x, tokens):
    # x: [B,T]
    return F.adaptive_avg_pool1d(x.unsqueeze(1), tokens).transpose(1, 2)  # [B,tokens,1]


def skeleton_quality_descriptors(sk, tokens=8):
    # Energy and temporal-difference energy. Missing skeleton => both near zero.
    energy = sk.abs().mean(dim=(2, 3))  # [B,T]
    diff = torch.zeros_like(energy)
    diff[:, 1:] = (sk[:, 1:] - sk[:, :-1]).abs().mean(dim=(2, 3))
    return torch.cat([
        _pool_scalar_sequence(energy, tokens),
        _pool_scalar_sequence(diff, tokens),
    ], dim=-1)  # [B,tokens,2]


def imu_quality_descriptors(imu, tokens=8):
    energy = imu.abs().mean(dim=2)
    diff = torch.zeros_like(energy)
    diff[:, 1:] = (imu[:, 1:] - imu[:, :-1]).abs().mean(dim=2)
    return torch.cat([
        _pool_scalar_sequence(energy, tokens),
        _pool_scalar_sequence(diff, tokens),
    ], dim=-1)


class SinusoidalTemporalEncoding(nn.Module):
    def __init__(self, dim=256, max_tokens=64):
        super().__init__()
        pe = torch.zeros(max_tokens, dim)
        pos = torch.arange(max_tokens, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]].to(dtype=x.dtype, device=x.device)


class IntrinsicReliability(nn.Module):
    """
    Predicts modality quality BEFORE cross-modal alignment.
    Receives only the modality's own feature tokens + simple raw-input quality descriptors.
    """
    def __init__(self, dim=256, desc_dim=2):
        super().__init__()
        hidden = max(64, dim // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(dim + desc_dim),
            nn.Linear(dim + desc_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, tokens, descriptors):
        return self.net(torch.cat([tokens, descriptors], dim=-1))


class ReliabilityGatedTemporalAlign(nn.Module):
    """
    Bidirectional cross-modal temporal alignment.
    Positional encoding is applied before attention.
    Cross-modal residual is gated by the OTHER modality's pre-alignment reliability.
    """
    def __init__(self, dim=256, heads=4, dropout=0.1):
        super().__init__()
        self.pos = SinusoidalTemporalEncoding(dim)
        self.ns = nn.LayerNorm(dim)
        self.ni = nn.LayerNorm(dim)
        self.s_queries_i = nn.MultiheadAttention(
            dim, heads, batch_first=True, dropout=dropout
        )
        self.i_queries_s = nn.MultiheadAttention(
            dim, heads, batch_first=True, dropout=dropout
        )
        self.out_s = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.out_i = nn.Sequential(nn.Linear(dim, dim), nn.GELU())

    def forward(self, s, i, r_s, r_i):
        sp = self.pos(self.ns(s))
        ip = self.pos(self.ni(i))

        ds, A_si = self.s_queries_i(
            query=sp, key=ip, value=i,
            need_weights=True, average_attn_weights=True
        )
        di, A_is = self.i_queries_s(
            query=ip, key=sp, value=s,
            need_weights=True, average_attn_weights=True
        )

        # Avoid a bad/missing modality contaminating the good branch.
        s_out = s + r_i * self.out_s(ds)
        i_out = i + r_s * self.out_i(di)

        return s_out, i_out, A_si, A_is


class LateFusionSI(nn.Module):
    def __init__(self, nc=27, dim=256):
        super().__init__()
        self.sk = SkeletonEncoder(dim)
        self.im = IMUEncoder(dim)
        self.head = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim, nc),
        )

    def forward(self, sk, imu):
        s = self.sk(sk).mean(1)
        i = self.im(imu).mean(1)
        return {"logits": self.head(torch.cat([s, i], dim=-1))}


class SkeletonOnly(nn.Module):
    def __init__(self, nc=27, dim=256):
        super().__init__()
        self.enc = SkeletonEncoder(dim)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, nc))

    def forward(self, sk, imu):
        z = self.enc(sk)
        return {"logits": self.head(z.mean(1))}


class IMUOnly(nn.Module):
    def __init__(self, nc=27, dim=256):
        super().__init__()
        self.enc = IMUEncoder(dim)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, nc))

    def forward(self, sk, imu):
        z = self.enc(imu)
        return {"logits": self.head(z.mean(1))}


class RATAMAR_SI_v2(nn.Module):
    """
    RATA-MAR-SI v2.1 (architecture unchanged)

    Key changes vs v1:
    1) reliability estimated BEFORE cross-modal alignment,
    2) temporal positional encoding,
    3) reliability-gated cross-modal residuals,
    4) auxiliary unimodal classification heads,
    5) normalized positive reliability fusion.
    """
    def __init__(
        self,
        nc=27,
        dim=256,
        tokens=8,
        heads=4,
        transformer_layers=2,
        fusion_temperature=0.7,
        use_alignment=True,
        use_reliability=True,
    ):
        super().__init__()
        self.tokens = tokens
        self.use_alignment = bool(use_alignment)
        self.use_reliability = bool(use_reliability)
        self.temp = float(fusion_temperature)

        self.sk = SkeletonEncoder(dim, tokens=tokens)
        self.im = IMUEncoder(dim, tokens=tokens)

        self.rel_s = IntrinsicReliability(dim)
        self.rel_i = IntrinsicReliability(dim)
        self.align = ReliabilityGatedTemporalAlign(dim, heads)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=transformer_layers)

        self.aux_s = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, nc))
        self.aux_i = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, nc))
        self.main_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, nc))

    def forward(self, sk, imu):
        s = self.sk(sk)
        i = self.im(imu)

        d_s = skeleton_quality_descriptors(sk, self.tokens)
        d_i = imu_quality_descriptors(imu, self.tokens)

        if self.use_reliability:
            logit_s = self.rel_s(s, d_s)
            logit_i = self.rel_i(i, d_i)
            r_s = torch.sigmoid(logit_s)
            r_i = torch.sigmoid(logit_i)
        else:
            logit_s = torch.zeros(
                s.shape[0], s.shape[1], 1, device=s.device, dtype=s.dtype
            )
            logit_i = torch.zeros_like(logit_s)
            r_s = torch.ones_like(logit_s)
            r_i = torch.ones_like(logit_i)

        # IMPORTANT: reliability is measured on uncontaminated modality features.
        aux_s_logits = self.aux_s(s.mean(1))
        aux_i_logits = self.aux_i(i.mean(1))

        if self.use_alignment:
            s_aligned, i_aligned, A_si, A_is = self.align(s, i, r_s, r_i)
        else:
            s_aligned, i_aligned = s, i
            B, T = s.shape[:2]
            eye = torch.eye(T, device=s.device, dtype=s.dtype).unsqueeze(0).expand(B, -1, -1)
            A_si = eye
            A_is = eye

        # Positive quality weights. Softmax(log(r)) is equivalent to normalized
        # positive reliability with temperature control.
        eps = 1e-4
        rel_log = torch.cat([
            torch.log(r_s.clamp_min(eps)),
            torch.log(r_i.clamp_min(eps)),
        ], dim=-1)
        alpha = torch.softmax(rel_log / self.temp, dim=-1)

        fused = (
            alpha[..., 0:1] * s_aligned
            + alpha[..., 1:2] * i_aligned
        )

        fused = self.temporal(fused)
        logits = self.main_head(fused.mean(1))

        return {
            "logits": logits,
            "aux_skel_logits": aux_s_logits,
            "aux_imu_logits": aux_i_logits,
            "r_skel": r_s,
            "r_imu": r_i,
            "rel_logit_skel": logit_s,
            "rel_logit_imu": logit_i,
            "alpha": alpha,
            "align_skel_to_imu": A_si,
            "align_imu_to_skel": A_is,
        }
