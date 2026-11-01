"""Revision-1 training of ADDITIONAL models (A100). The published RATA-MAR checkpoints are not touched.

Groups (each run = 100 epochs, official cross-subject protocol, last-epoch weights, same recipe as the paper):
  R0     RATA-MAR K=8 default, seed 0 only        -> checks that the training pipeline reproduces the paper
  K      token count K in {4, 16, 32}, seeds 0-2  -> Comment 7
  T      reliability targets (removal, dropout, noise), seeds 0-2 -> Comment 4
           T_low (0, 0.40, 0.45) | T_high (0, 0.80, 0.85) | T_flat (0, 0.50, 0.50)
  B      re-implemented robust-fusion baselines on the SAME encoders and SAME corrupted-view training,
           seeds 0-2 -> Comment 1
           TMC-style  : per-modality Dirichlet evidence + Dempster-Shafer combination (Han et al., TPAMI 2023)
           QMF-style  : energy-based per-sample modality confidence for decision fusion + correctness-ranking
                        regulariser (Zhang et al., ICML 2023)

Usage:  python -I r1_train_variants_gpu.py [--groups R0,K,T,B] [--seeds 0,1,2]
Results: experiments/outputs/gpu_train/variants_per_seed.csv (+ extended-suite CSV); resumable.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import r1_common as C
import src_rata_mar_si_v21.train_v21 as T
from src_rata_mar_si_v21.models_v21 import RATAMAR_SI_v2, SkeletonEncoder, IMUEncoder

OUT = C.OUT / 'gpu_train'
CKPT = OUT / 'final_models'
EPOCHS, LR, BATCH = 100, 3e-4, 64
_ORIG_CORRUPT = T.corrupt_dual_view


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ---------------------------------------------------------------- reliability-target variants
def use_targets(targets):
    """Remap the reliability targets produced by the ORIGINAL corruption routine.
    The random-number sequence is unchanged, so (0, 0.60, 0.65) reproduces the paper exactly."""
    rem, drop, noise = targets
    if targets == (0.0, 0.60, 0.65):
        T.corrupt_dual_view = _ORIG_CORRUPT
        return

    def corrupt(sk, imu):
        csk, cimu, q, sh = _ORIG_CORRUPT(sk, imu)
        q2 = q.clone()
        q2[torch.isclose(q, torch.tensor(0.60, device=q.device))] = drop
        q2[torch.isclose(q, torch.tensor(0.65, device=q.device))] = noise
        q2[q == 0.0] = rem
        return csk, cimu, q2, sh
    T.corrupt_dual_view = corrupt


# ---------------------------------------------------------------- re-implemented baselines
def _mlp_head(dim, nc):
    return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(0.2), nn.Linear(dim, nc))


class TMCStyle(nn.Module):
    """Trusted multi-view classification on the shared encoders (two views: skeleton, IMU)."""
    def __init__(self, nc=27, dim=256):
        super().__init__()
        self.nc = nc
        self.sk, self.im = SkeletonEncoder(dim), IMUEncoder(dim)
        self.h_s, self.h_i = _mlp_head(dim, nc), _mlp_head(dim, nc)

    def ds_combine(self, a1, a2):
        K = self.nc
        S1, S2 = a1.sum(1, keepdim=True), a2.sum(1, keepdim=True)
        b1, b2 = (a1 - 1) / S1, (a2 - 1) / S2
        u1, u2 = K / S1, K / S2
        bb = torch.bmm(b1.unsqueeze(2), b2.unsqueeze(1))
        conflict = bb.sum((1, 2)) - torch.diagonal(bb, dim1=1, dim2=2).sum(1)
        scale = (1 - conflict).clamp_min(1e-6).unsqueeze(1)
        b = (b1 * b2 + b1 * u2 + b2 * u1) / scale
        u = (u1 * u2) / scale
        S = K / u
        return b * S + 1

    def forward(self, sk, imu):
        a_s = F.softplus(self.h_s(self.sk(sk).mean(1))) + 1
        a_i = F.softplus(self.h_i(self.im(imu).mean(1))) + 1
        a = self.ds_combine(a_s, a_i)
        return {'logits': torch.log(a), 'alphas': (a_s, a_i, a),
                'uncertainty': self.nc / a.sum(1)}


def _edl_loss(alpha, y, nc, lam):
    S = alpha.sum(1, keepdim=True)
    yo = F.one_hot(y, nc).float()
    ace = (yo * (torch.digamma(S) - torch.digamma(alpha))).sum(1)
    at = yo + (1 - yo) * alpha
    St = at.sum(1, keepdim=True)
    kl = (torch.lgamma(St).squeeze(1) - torch.lgamma(torch.tensor(float(nc), device=alpha.device))
          - torch.lgamma(at).sum(1)
          + ((at - 1) * (torch.digamma(at) - torch.digamma(St))).sum(1))
    return (ace + lam * kl).mean()


def tmc_loss(out, y, lam):
    return sum(_edl_loss(a, y, 27, lam) for a in out['alphas'])


class QMFStyle(nn.Module):
    """Quality-aware decision fusion: energy-based confidence (detached) weights unimodal logits."""
    def __init__(self, nc=27, dim=256):
        super().__init__()
        self.sk, self.im = SkeletonEncoder(dim), IMUEncoder(dim)
        self.h_s, self.h_i = _mlp_head(dim, nc), _mlp_head(dim, nc)

    def forward(self, sk, imu):
        ls, li = self.h_s(self.sk(sk).mean(1)), self.h_i(self.im(imu).mean(1))
        cs = (torch.logsumexp(ls, 1, keepdim=True) / 10)
        ci = (torch.logsumexp(li, 1, keepdim=True) / 10)
        return {'logits': ls * cs.detach() + li * ci.detach(), 'uni': (ls, li), 'conf': (cs, ci)}


class History:
    """Running per-sample correctness of each unimodal head (clean view), as in correctness ranking."""
    def __init__(self, n):
        self.correct = torch.zeros(n, 2)
        self.count = 0

    def update(self, idx, ok_s, ok_i):
        self.correct[idx, 0] += ok_s.float().cpu()
        self.correct[idx, 1] += ok_i.float().cpu()

    def rank_loss(self, conf, idx, m, epoch):
        if epoch < 2:
            return conf.sum() * 0
        h = (self.correct[idx, m] / max(epoch - 1, 1)).to(conf.device)
        c = conf.squeeze(1)
        perm = torch.roll(torch.arange(len(c), device=c.device), 1)
        target = torch.sign(h - h[perm])
        margin = (h - h[perm]).abs()
        return (F.relu(-target * (c - c[perm]) + margin)).mean()


# ---------------------------------------------------------------- training
def dual_view(model_kind, model, sk, imu, y, idx, epoch, hist):
    clean = model(sk, imu)
    csk, cimu, q, _ = T.corrupt_dual_view(sk, imu)
    corr = model(csk, cimu)
    if model_kind == 'TMC':
        lam = min(1.0, epoch / 50.0)
        return tmc_loss(clean, y, lam) + 0.65 * tmc_loss(corr, y, lam)
    if model_kind == 'QMF':
        def ce_all(o):
            ls, li = o['uni']
            return F.cross_entropy(o['logits'], y) + F.cross_entropy(ls, y) + F.cross_entropy(li, y)
        ls, li = clean['uni']
        hist.update(idx.cpu(), ls.argmax(1) == y, li.argmax(1) == y)
        reg = hist.rank_loss(clean['conf'][0], idx, 0, epoch) + hist.rank_loss(clean['conf'][1], idx, 1, epoch)
        return ce_all(clean) + 0.65 * ce_all(corr) + 0.1 * reg
    raise ValueError(model_kind)


def train(kind, model, ftrain, seed, dev):
    loader = T.make_loader(ftrain, C.SK_NPY, C.IMU_NPY, batch=BATCH, shuffle=True, seed=seed)
    if kind == 'RATA':
        return T.train_fixed(model, loader, epochs=EPOCHS, lr=LR, mode='rata_v2', seed=seed, device=dev)
    T.seed_everything(seed)
    model = model.to(dev)
    id2i = {sid: k for k, sid in enumerate(ftrain.sample_id.tolist())}
    hist = History(len(ftrain))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda', enabled=dev.startswith('cuda'))
    for ep in range(1, EPOCHS + 1):
        model.train()
        for b in loader:
            sk, imu, y = b['skeleton'].to(dev), b['imu'].to(dev), b['label'].to(dev)
            idx = torch.tensor([id2i[s] for s in b['sample_id']])
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=dev.startswith('cuda') and kind != 'TMC'):
                loss = dual_view(kind, model, sk, imu, y, idx, ep, hist)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
        sched.step()
        if ep == 1 or ep % 20 == 0:
            log(f'  {kind} seed {seed} epoch {ep}/{EPOCHS} loss {float(loss):.4f}')
    return model


# ---------------------------------------------------------------- evaluation (paper protocol)
def evaluate_row(model, test_loader, dev):
    row = {}
    r = T.evaluate(model, test_loader, 'clean', device=dev)
    row['clean_top1'], row['clean_macro_f1'] = r['top1'], r['macro_f1']
    for sc in ['missing_skeleton', 'missing_imu']:
        row[sc] = T.evaluate(model, test_loader, sc, device=dev)['top1']
    for sc in ['skeleton_noise', 'imu_noise', 'joint_drop_30']:
        row[sc] = T.evaluate_repeated_corruption(model, test_loader, sc, C.CORRUPTION_SEEDS, device=dev)[0]['top1']
    for a in [4, 8, 16]:
        row[f'delay_pm{a}'] = T.evaluate_shift_average(model, test_loader, a, device=dev)['top1']
    return row


def evaluate_extended(model, dev, sk, im, y):
    yn = y.cpu().numpy()
    rows = []
    for sc, (stream, stoch, _) in C.EXT_SCENARIOS.items():
        for cs in (C.CORRUPTION_SEEDS if stoch else [C.CORRUPTION_SEEDS[0]]):
            s2, i2 = C.apply_extended(sk, im, y, sc, cs)
            p = C.forward_all(model, s2, i2)['logits'].argmax(-1).numpy()
            rows.append({'scenario': sc, 'stream': stream, 'corruption_seed': cs, 'top1': C.accuracy(p, yn)})
    return rows


def run_specs(groups, seeds):
    specs = []
    if 'R0' in groups:
        specs.append(('R0_RATA_K8_default', 'RATA', dict(tokens=8), (0.0, 0.60, 0.65), [0]))
    if 'K' in groups:
        for k in (4, 16, 32):
            specs.append((f'K{k}', 'RATA', dict(tokens=k), (0.0, 0.60, 0.65), seeds))
    if 'T' in groups:
        for name, t in (('T_low', (0.0, 0.40, 0.45)), ('T_high', (0.0, 0.80, 0.85)), ('T_flat', (0.0, 0.50, 0.50))):
            specs.append((name, 'RATA', dict(tokens=8), t, seeds))
    if 'B' in groups:
        specs.append(('TMC-style', 'TMC', {}, (0.0, 0.60, 0.65), seeds))
        specs.append(('QMF-style', 'QMF', {}, (0.0, 0.60, 0.65), seeds))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', default='R0,K,T,B')
    ap.add_argument('--seeds', default='0,1,2')
    args = ap.parse_args()
    dev = C.device_name()
    assert dev == 'cuda', 'Training must run on the A100 runtime.'
    groups = args.groups.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]
    OUT.mkdir(parents=True, exist_ok=True); CKPT.mkdir(parents=True, exist_ok=True)
    res_path, ext_path = OUT / 'variants_per_seed.csv', OUT / 'variants_extended_per_run.csv'
    done = pd.read_csv(res_path) if res_path.exists() else pd.DataFrame(columns=['variant', 'seed'])
    ftrain, ftest = T.official_train_test(C.MANIFEST)
    test_loader = T.make_loader(ftest, C.SK_NPY, C.IMU_NPY, batch=BATCH, shuffle=False, seed=0)
    sk, im, y, _ = C.load_test_tensors(dev)
    log('GPU:', torch.cuda.get_device_name(0))
    for name, kind, kw, targets, sds in run_specs(groups, seeds):
        for seed in sds:
            if ((done.variant == name) & (done.seed == seed)).any():
                log('skip', name, seed); continue
            log('=== train', name, 'seed', seed)
            t0 = time.time()
            use_targets(targets)
            T.seed_everything(seed)                       # BEFORE construction, as in the paper
            model = {'RATA': lambda: RATAMAR_SI_v2(**kw), 'TMC': TMCStyle, 'QMF': QMFStyle}[kind]()
            ck = CKPT / f'{name}_seed{seed}.pt'
            if ck.exists():
                model.load_state_dict(torch.load(ck, map_location=dev)['model']); model = model.to(dev)
            else:
                model = train(kind, model, ftrain, seed, dev)
                torch.save({'model': model.state_dict(), 'metadata': {'variant': name, 'seed': seed, 'targets': targets,
                                                                      **kw, 'epochs': EPOCHS}}, ck)
            use_targets((0.0, 0.60, 0.65))
            model.eval()
            row = {'variant': name, 'kind': kind, 'seed': seed, 'targets': str(targets), 'tokens': kw.get('tokens', 8),
                   **evaluate_row(model, test_loader, dev), 'train_minutes': (time.time() - t0) / 60}
            ext = evaluate_extended(model, dev, sk, im, y)
            for e in ext:
                e.update(variant=name, seed=seed)
            row['extended_mean_top1'] = float(pd.DataFrame(ext).groupby('scenario').top1.mean().mean())
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
            done.to_csv(res_path, index=False)
            new_ext = pd.DataFrame(ext)
            if ext_path.exists():
                new_ext = pd.concat([pd.read_csv(ext_path), new_ext], ignore_index=True)
            new_ext.to_csv(ext_path, index=False)
            log(f'    clean {row["clean_top1"]*100:.2f} | miss-skel {row["missing_skeleton"]*100:.2f} | '
                f'miss-imu {row["missing_imu"]*100:.2f} | ext-mean {row["extended_mean_top1"]*100:.2f} | '
                f'{row["train_minutes"]:.1f} min')
            del model; torch.cuda.empty_cache()
    log('ALL TRAINING DONE ->', res_path)


if __name__ == '__main__':
    main()
