"""Shared helpers for the Revision-1 analyses of RATA-MAR (IASC-94359).

Only re-uses the frozen v2.1 code in src_rata_mar_si_v21 and the saved final checkpoints.
Nothing in src_rata_mar_si_v21 or runs/RATA_MAR_SI_v21 is modified.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

warnings.filterwarnings('ignore', message='.*enable_nested_tensor.*')
warnings.filterwarnings('ignore', message='.*pin_memory.*')

PROJECT = Path(os.environ.get('RATA_PROJECT', Path(__file__).resolve().parents[2]))  # repository root
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src_rata_mar_si_v21.models_v21 import (  # noqa: E402
    RATAMAR_SI_v2, LateFusionSI, SkeletonOnly, IMUOnly,
)
from src_rata_mar_si_v21.train_v21 import (  # noqa: E402
    official_train_test, apply_scenario, temporal_delay_zero_pad, seed_everything,
)

PROCESSED = PROJECT / 'processed/UTD_MHAD_original_SI'
MANIFEST = PROCESSED / 'manifest_27class.csv'
SK_NPY = PROCESSED / 'utd_mhad_skeleton_T64_f16.npy'
IMU_NPY = PROCESSED / 'utd_mhad_inertial_T64_f16.npy'
FINAL_MODELS = PROJECT / 'runs/RATA_MAR_SI_v21/final_models'
R1 = PROJECT / 'REVISION_R1'
OUT = R1 / 'experiments/outputs'

SEEDS = [0, 1, 2]
CORRUPTION_SEEDS = [1103, 2207, 3301, 4409, 5519]
BATCH = 64

# Display name -> (constructor, checkpoint pattern). "RATA-MAR" is the v2.1 model of the paper.
MODEL_SPECS = {
    'LateFusion-Clean': (LateFusionSI, 'LateFusion-Clean_seed{s}.pt'),
    'LateFusion-Robust': (LateFusionSI, 'LateFusion-Robust_seed{s}.pt'),
    'RATA-MAR': (RATAMAR_SI_v2, 'RATA-MAR-SI-v2.1_seed{s}.pt'),
    'NoAlignment': (lambda: RATAMAR_SI_v2(use_alignment=False), 'Ablation_NoAlignment_seed{s}.pt'),
    'NoReliability': (lambda: RATAMAR_SI_v2(use_reliability=False), 'Ablation_NoReliability_seed{s}.pt'),
    'SkeletonOnly': (SkeletonOnly, 'SkeletonOnly_seed{s}.pt'),
    'IMUOnly': (IMUOnly, 'IMUOnly_seed{s}.pt'),
}


def device_name():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(name, seed, device):
    ctor, pattern = MODEL_SPECS[name]
    seed_everything(seed)
    model = ctor()
    ckpt = torch.load(FINAL_MODELS / pattern.format(s=seed), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    return model.to(device).eval()


def load_test_tensors(device):
    """Official cross-subject test set, in the same order as the paper's test loader."""
    _, test = official_train_test(MANIFEST)
    test = test.reset_index(drop=True)
    sk = np.load(SK_NPY, mmap_mode='r')
    im = np.load(IMU_NPY, mmap_mode='r')
    idx = test.row_index.astype(int).to_numpy()
    sk_t = torch.from_numpy(np.asarray(sk[idx], dtype=np.float32)).to(device)
    im_t = torch.from_numpy(np.asarray(im[idx], dtype=np.float32)).to(device)
    y = torch.from_numpy(test.label.astype(int).to_numpy()).to(device)
    return sk_t, im_t, y, test


@torch.no_grad()
def forward_all(model, sk, imu, keep_attention=False):
    """Batched forward over the whole test set (batch 64, same order as the paper)."""
    keys = ['logits', 'aux_skel_logits', 'aux_imu_logits']
    out = {k: [] for k in keys}
    extra = {'r_skel': [], 'r_imu': [], 'alpha': [], 'r_skel_tok': [], 'r_imu_tok': []}
    att = []
    for b in range(0, sk.shape[0], BATCH):
        o = model(sk[b:b + BATCH], imu[b:b + BATCH])
        for k in keys:
            if k in o:
                out[k].append(o[k].float().cpu())
        if 'r_skel' in o:
            extra['r_skel'].append(o['r_skel'].mean((1, 2)).float().cpu())
            extra['r_imu'].append(o['r_imu'].mean((1, 2)).float().cpu())
            extra['alpha'].append(o['alpha'].mean(1).float().cpu())
            extra['r_skel_tok'].append(o['r_skel'][..., 0].float().cpu())
            extra['r_imu_tok'].append(o['r_imu'][..., 0].float().cpu())
            if keep_attention:
                att.append(o['align_skel_to_imu'].float().cpu())
    res = {k: torch.cat(v) for k, v in out.items() if v}
    res.update({k: torch.cat(v) for k, v in extra.items() if v})
    if att:
        res['att_s2i'] = torch.cat(att)
    return res


@torch.no_grad()
def predict_original_scenario(model, sk, imu, scenario, scenario_seed=20260928):
    """Exact replica of train_v21.evaluate(): same batches, same per-batch generator seeds."""
    preds, rs, ri, al = [], [], [], []
    for bi, b in enumerate(range(0, sk.shape[0], BATCH)):
        s, i = apply_scenario(sk[b:b + BATCH], imu[b:b + BATCH], scenario,
                              seed=int(scenario_seed) + bi * 1000003)
        o = model(s, i)
        preds.append(o['logits'].argmax(-1).cpu())
        if 'r_skel' in o:
            rs.append(o['r_skel'].mean((1, 2)).cpu()); ri.append(o['r_imu'].mean((1, 2)).cpu())
            al.append(o['alpha'].mean(1).cpu())
    res = {'pred': torch.cat(preds).numpy()}
    if rs:
        res.update(r_skel=torch.cat(rs).numpy(), r_imu=torch.cat(ri).numpy(), alpha=torch.cat(al).numpy())
    return res


# ----------------------------------------------------------------------------------------
# Extended (unseen) degradation scenarios. All randomness is drawn from a CPU generator so
# that the corrupted inputs are identical on CPU and GPU.
# ----------------------------------------------------------------------------------------
RIGHT_ARM = [8, 9, 10, 11]   # right shoulder, elbow, wrist, hand (0-based, UTD-MHAD order)


def _u(g, shape, lo, hi):
    return lo + (hi - lo) * torch.rand(shape, generator=g)


def _ri(g, lo, hi, n):
    return torch.randint(lo, hi + 1, (n,), generator=g)


def _segments_mask(g, B, T, n_seg, lo, hi):
    """Boolean mask [B,T] with n_seg random segments of random length in [lo,hi]."""
    m = torch.zeros(B, T, dtype=torch.bool)
    for b in range(B):
        for _ in range(n_seg):
            L = int(_ri(g, lo, hi, 1))
            st = int(_ri(g, 0, T - L, 1))
            m[b, st:st + L] = True
    return m


def _freeze(x, g):
    """Sensor stuck at a random time t0 in [16, 48]: hold x[t0] until the end."""
    x = x.clone()
    t0 = _ri(g, 16, 48, x.shape[0])
    for b in range(x.shape[0]):
        x[b, t0[b]:] = x[b, t0[b]]
    return x


def _time_index_warp(x, src_index):
    """Linear interpolation of x[B,T,...] at fractional source indices [B,T]; outside -> 0."""
    B, T = x.shape[:2]
    flat = x.reshape(B, T, -1)
    lo = src_index.floor().long()
    w = (src_index - lo.float()).unsqueeze(-1)
    valid = ((src_index >= 0) & (src_index <= T - 1)).unsqueeze(-1).float()
    lo_c = lo.clamp(0, T - 1)
    hi_c = (lo + 1).clamp(0, T - 1)
    a = torch.gather(flat, 1, lo_c.unsqueeze(-1).expand(-1, -1, flat.shape[-1]))
    b = torch.gather(flat, 1, hi_c.unsqueeze(-1).expand(-1, -1, flat.shape[-1]))
    return ((1 - w) * a + w * b) * valid


def _shift_rows(x, shifts):
    return temporal_delay_zero_pad(x, torch.as_tensor(shifts, dtype=torch.long))


def _derangement_other_class(g, y):
    """For each sample pick a donor sample from a different class."""
    y = y.cpu()
    n = y.shape[0]
    donor = torch.empty(n, dtype=torch.long)
    for k in range(n):
        while True:
            j = int(torch.randint(0, n, (1,), generator=g))
            if y[j] != y[k]:
                donor[k] = j
                break
    return donor


# name -> (affected stream, stochastic?, description used in the paper table)
EXT_SCENARIOS = {
    # inertial-side, not seen during training
    'imu_bias':          ('IMU', True,  'Constant per-channel offset, uniform in [-1, 1] channel SD'),
    'imu_gain':          ('IMU', True,  'Per-channel calibration gain, uniform in [0.6, 1.4]'),
    'imu_drift':         ('IMU', True,  'Bias accumulating linearly to a final offset in [-2, 2] channel SD'),
    'imu_frozen':        ('IMU', True,  'Sensor output frozen from a random time step in [16, 48]'),
    'imu_saturation':    ('IMU', False, 'Signal clipped at +/-1 channel SD'),
    'imu_intermittent':  ('IMU', True,  'Four zero-filled gaps of 2-8 time steps at random positions'),
    'imu_rate_mismatch': ('IMU', True,  'Sampling-rate mismatch: time axis rescaled by a factor in [0.8, 1.2]'),
    'imu_clock_drift':   ('IMU', True,  'Delay growing linearly from 0 to 16 time steps (random sign)'),
    'imu_random_delay':  ('IMU', True,  'Integer delay drawn uniformly from [-16, 16] per sequence'),
    # skeleton-side
    'skel_tracking_loss': ('Skeleton', True,  'Three zero-filled tracking gaps of 4-12 frames'),
    'skel_frozen':        ('Skeleton', True,  'Skeleton frozen from a random frame in [16, 48]'),
    'skel_jitter_bursts': ('Skeleton', True,  'Two 8-frame bursts of Gaussian jitter (SD 0.20)'),
    'skel_arm_occlusion': ('Skeleton', False, 'Right-arm joints (shoulder to hand) missing for the whole sequence'),
    'skel_offset':        ('Skeleton', True,  'Global translation offset, uniform in [-0.2, 0.2] per axis'),
    # simultaneous degradation of both streams
    'both_noise':          ('Both', True, 'Skeleton noise (SD 0.10) and IMU noise (SD 0.25)'),
    'both_noise_strong':   ('Both', True, 'Skeleton noise (SD 0.20) and IMU noise (SD 0.50)'),
    'jointdrop_imudelay':  ('Both', True, 'Joint dropout (30%) and IMU delay of 8 steps (random sign)'),
    'both_intermittent':   ('Both', True, 'Skeleton tracking gaps and IMU gaps, drawn independently'),
    'both_severe':         ('Both', True, 'Joint dropout (50%) and IMU noise (SD 0.50)'),
    'skelnoise_imubias':   ('Both', True, 'Skeleton noise (SD 0.10) and IMU offset in [-1, 1] channel SD'),
    # plausible-looking but semantically wrong input
    'imu_conflict':  ('IMU', True,      'IMU stream taken from a test sequence of another class'),
    'skel_conflict': ('Skeleton', True, 'Skeleton stream taken from a test sequence of another class'),
}


def apply_extended(sk, imu, y, scenario, seed):
    """sk [N,T,20,3], imu [N,T,6] on any device. Returns corrupted copies on the same device."""
    dev = sk.device
    g = torch.Generator().manual_seed(int(seed))
    s = sk.detach().cpu().clone()
    i = imu.detach().cpu().clone()
    N, T = s.shape[:2]

    def imu_delay_random_sign(amount):
        sign = torch.where(torch.rand(N, generator=g) < 0.5, -1, 1)
        return _shift_rows(i, sign * amount)

    if scenario == 'imu_bias':
        i = i + _u(g, (N, 1, 6), -1.0, 1.0)
    elif scenario == 'imu_gain':
        i = i * _u(g, (N, 1, 6), 0.6, 1.4)
    elif scenario == 'imu_drift':
        end = _u(g, (N, 1, 6), -2.0, 2.0)
        ramp = torch.linspace(0, 1, T).view(1, T, 1)
        i = i + ramp * end
    elif scenario == 'imu_frozen':
        i = _freeze(i, g)
    elif scenario == 'imu_saturation':
        i = i.clamp(-1.0, 1.0)
    elif scenario == 'imu_intermittent':
        m = _segments_mask(g, N, T, 4, 2, 8)
        i[m] = 0
    elif scenario == 'imu_rate_mismatch':
        f = _u(g, (N, 1), 0.8, 1.2)
        src = torch.arange(T).float().view(1, T) * f
        i = _time_index_warp(i, src)
    elif scenario == 'imu_clock_drift':
        sign = torch.where(torch.rand(N, 1, generator=g) < 0.5, -1.0, 1.0)
        delay = sign * torch.linspace(0, 16, T).view(1, T)
        src = torch.arange(T).float().view(1, T) - delay
        i = _time_index_warp(i, src)
    elif scenario == 'imu_random_delay':
        i = _shift_rows(i, _ri(g, -16, 16, N))
    elif scenario == 'skel_tracking_loss':
        m = _segments_mask(g, N, T, 3, 4, 12)
        s[m] = 0
    elif scenario == 'skel_frozen':
        s = _freeze(s, g)
    elif scenario == 'skel_jitter_bursts':
        m = _segments_mask(g, N, T, 2, 8, 8).float().view(N, T, 1, 1)
        s = s + m * 0.20 * torch.randn(s.shape, generator=g)
    elif scenario == 'skel_arm_occlusion':
        s[:, :, RIGHT_ARM, :] = 0
    elif scenario == 'skel_offset':
        s = s + _u(g, (N, 1, 1, 3), -0.2, 0.2)
    elif scenario == 'both_noise':
        s = s + 0.10 * torch.randn(s.shape, generator=g)
        i = i + 0.25 * torch.randn(i.shape, generator=g)
    elif scenario == 'both_noise_strong':
        s = s + 0.20 * torch.randn(s.shape, generator=g)
        i = i + 0.50 * torch.randn(i.shape, generator=g)
    elif scenario == 'jointdrop_imudelay':
        mask = torch.rand((N, 20), generator=g) < 0.30
        s = s.masked_fill(mask[:, None, :, None], 0)
        i = imu_delay_random_sign(8)
    elif scenario == 'both_intermittent':
        s[_segments_mask(g, N, T, 3, 4, 12)] = 0
        i[_segments_mask(g, N, T, 4, 2, 8)] = 0
    elif scenario == 'both_severe':
        mask = torch.rand((N, 20), generator=g) < 0.50
        s = s.masked_fill(mask[:, None, :, None], 0)
        i = i + 0.50 * torch.randn(i.shape, generator=g)
    elif scenario == 'skelnoise_imubias':
        s = s + 0.10 * torch.randn(s.shape, generator=g)
        i = i + _u(g, (N, 1, 6), -1.0, 1.0)
    elif scenario == 'imu_conflict':
        i = i[_derangement_other_class(g, y)]
    elif scenario == 'skel_conflict':
        s = s[_derangement_other_class(g, y)]
    else:
        raise ValueError(scenario)
    return s.to(dev), i.to(dev)


def shift_stream(x, shift):
    """Zero-padded shift of a whole batch (positive = delayed)."""
    return temporal_delay_zero_pad(x, int(shift))


def accuracy(pred, y):
    return float((np.asarray(pred) == np.asarray(y)).mean())


def macro_f1(pred, y):
    from sklearn.metrics import f1_score
    return float(f1_score(y, pred, average='macro', zero_division=0))
