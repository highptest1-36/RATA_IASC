from __future__ import annotations

import io
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader


def seed_everything(seed: int = 0):
    """Seed Python/NumPy/PyTorch. Call this BEFORE model construction."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Prefer repeatability over benchmark auto-tuning.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return seed


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class UTDataset(Dataset):
    def __init__(self, df, sk_npy, imu_npy):
        self.df = df.reset_index(drop=True).copy()
        self.sk = np.load(sk_npy, mmap_mode='r')
        self.im = np.load(imu_npy, mmap_mode='r')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ridx = int(row.row_index)
        return {
            'skeleton': torch.from_numpy(np.asarray(self.sk[ridx], dtype=np.float32)),
            'imu': torch.from_numpy(np.asarray(self.im[ridx], dtype=np.float32)),
            'label': torch.tensor(int(row.label), dtype=torch.long),
            'sample_id': str(row.sample_id),
            'subject': torch.tensor(int(row.subject), dtype=torch.long),
        }


def make_loader(df, sk_npy, imu_npy, batch=64, shuffle=False, workers=2, seed=0):
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(
        UTDataset(df, sk_npy, imu_npy),
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=g,
        worker_init_fn=_seed_worker,
    )


def subject_disjoint_dev_split(manifest_csv, val_subject=7):
    """Development only: train odd subjects except val; validate on one unseen odd subject."""
    df = pd.read_csv(manifest_csv)
    odd = [1, 3, 5, 7]
    train_subjects = [s for s in odd if s != int(val_subject)]
    train = df[df.subject.isin(train_subjects)].copy()
    val = df[df.subject == int(val_subject)].copy()
    official_test = df[df.subject.isin([2, 4, 6, 8])].copy()
    print('Publication-safe development split')
    print(' train subjects:', train_subjects, 'N=', len(train))
    print(' val subject   :', val_subject, 'N=', len(val))
    print(' locked test   :', [2,4,6,8], 'N=', len(official_test), '(DO NOT USE during development)')
    return train, val, official_test


def official_train_test(manifest_csv):
    df = pd.read_csv(manifest_csv)
    train = df[df.subject.isin([1,3,5,7])].copy()
    test = df[df.subject.isin([2,4,6,8])].copy()
    return train, test


def temporal_delay_zero_pad(x, shift):
    """Non-circular temporal delay. Positive = delayed, negative = advanced."""
    if isinstance(shift, int):
        shifts = torch.full((x.shape[0],), shift, device=x.device, dtype=torch.long)
    else:
        shifts = shift.to(device=x.device, dtype=torch.long)
    out = torch.zeros_like(x)
    T = x.shape[1]
    for b, s in enumerate(shifts.tolist()):
        if s == 0:
            out[b] = x[b]
        elif s > 0:
            if s < T:
                out[b, s:] = x[b, :T-s]
        else:
            k = -s
            if k < T:
                out[b, :T-k] = x[b, k:]
    return out


def corrupt_dual_view(sk, imu):
    """Random corruption for TRAINING augmentation only."""
    B = sk.shape[0]
    device = sk.device
    im0 = imu
    sk = sk.clone()
    imu = imu.clone()
    q = torch.ones(B, 2, device=device)

    for b in range(B):
        u = random.random()
        if u < 0.10:
            sk[b].zero_(); q[b,0] = 0.0
        elif u < 0.25:
            mask = torch.rand(sk.shape[2], device=device) < 0.30
            sk[b, :, mask, :] = 0; q[b,0] = 0.60
        elif u < 0.40:
            sk[b] += 0.08 * torch.randn_like(sk[b]); q[b,0] = 0.65

        u = random.random()
        if u < 0.10:
            imu[b].zero_(); q[b,1] = 0.0
        elif u < 0.25:
            start = random.randint(0, max(0, imu.shape[1]-16))
            imu[b, start:start+16] = 0; q[b,1] = 0.60
        elif u < 0.40:
            imu[b] += 0.20 * torch.randn_like(imu[b]); q[b,1] = 0.65

        if q[b,0] == 0 and q[b,1] == 0:
            imu[b] = im0[b]
            q[b,1] = 1.0

    shifts = torch.zeros(B, device=device, dtype=torch.long)
    delay_mask = torch.rand(B, device=device) < 0.35
    choices = torch.tensor([-16,-8,-4,4,8,16], device=device)
    if delay_mask.any():
        pick = torch.randint(0, len(choices), (int(delay_mask.sum().item()),), device=device)
        shifts[delay_mask] = choices[pick]
        imu = temporal_delay_zero_pad(imu, shifts)
    return sk, imu, q, shifts


def weighted_ce(logits, y, quality):
    per = F.cross_entropy(logits, y, reduction='none')
    denom = quality.sum().clamp_min(1.0)
    return (per * quality).sum() / denom


def reliability_loss(out, q):
    pred = torch.cat([out['r_skel'].mean(1), out['r_imu'].mean(1)], dim=1)
    return F.mse_loss(pred, q)


def dual_view_rata_loss(model, sk, imu, y, weights=None):
    if weights is None:
        weights = {
            'clean_cls': 1.00,
            'corrupt_cls': 0.65,
            'aux_clean': 0.18,
            'aux_corrupt': 0.12,
            'reliability': 0.15,
        }
    clean = model(sk, imu)
    csk, cimu, q, shifts = corrupt_dual_view(sk, imu)
    corrupt = model(csk, cimu)

    l_clean = F.cross_entropy(clean['logits'], y)
    l_corrupt = F.cross_entropy(corrupt['logits'], y)
    l_aux_clean = 0.5 * (
        F.cross_entropy(clean['aux_skel_logits'], y) +
        F.cross_entropy(clean['aux_imu_logits'], y)
    )
    l_aux_corrupt = 0.5 * (
        weighted_ce(corrupt['aux_skel_logits'], y, q[:,0]) +
        weighted_ce(corrupt['aux_imu_logits'], y, q[:,1])
    )

    if getattr(model, 'use_reliability', True):
        q_clean = torch.ones_like(q)
        l_rel = 0.5 * (reliability_loss(clean, q_clean) + reliability_loss(corrupt, q))
    else:
        l_rel = torch.zeros((), device=sk.device)

    total = (
        weights['clean_cls'] * l_clean +
        weights['corrupt_cls'] * l_corrupt +
        weights['aux_clean'] * l_aux_clean +
        weights['aux_corrupt'] * l_aux_corrupt +
        weights['reliability'] * l_rel
    )
    return total, {
        'clean_cls': l_clean,
        'corrupt_cls': l_corrupt,
        'aux_clean': l_aux_clean,
        'aux_corrupt': l_aux_corrupt,
        'rel': l_rel,
        'mean_abs_shift': shifts.abs().float().mean(),
    }


def dual_view_late_loss(model, sk, imu, y, corrupt_weight=0.65):
    clean = model(sk, imu)
    csk, cimu, q, shifts = corrupt_dual_view(sk, imu)
    corrupt = model(csk, cimu)
    l_clean = F.cross_entropy(clean['logits'], y)
    l_corrupt = F.cross_entropy(corrupt['logits'], y)
    return l_clean + corrupt_weight*l_corrupt, {
        'clean_cls': l_clean,
        'corrupt_cls': l_corrupt,
        'mean_abs_shift': shifts.abs().float().mean(),
    }


STOCHASTIC_SCENARIOS = {
    'skeleton_noise', 'skeleton_noise_strong',
    'imu_noise', 'imu_noise_strong',
    'joint_drop_30', 'joint_drop_50'
}


def _generator(device, seed):
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def apply_scenario(sk, imu, scenario, seed=0):
    """Deterministic evaluation corruption when seed is fixed."""
    sk = sk.clone()
    imu = imu.clone()
    g = _generator(sk.device, seed)

    if scenario == 'clean':
        pass
    elif scenario == 'missing_skeleton':
        sk.zero_()
    elif scenario == 'missing_imu':
        imu.zero_()
    elif scenario == 'skeleton_noise':
        sk += 0.10 * torch.randn(sk.shape, generator=g, device=sk.device, dtype=sk.dtype)
    elif scenario == 'skeleton_noise_strong':
        sk += 0.20 * torch.randn(sk.shape, generator=g, device=sk.device, dtype=sk.dtype)
    elif scenario == 'imu_noise':
        imu += 0.25 * torch.randn(imu.shape, generator=g, device=imu.device, dtype=imu.dtype)
    elif scenario == 'imu_noise_strong':
        imu += 0.50 * torch.randn(imu.shape, generator=g, device=imu.device, dtype=imu.dtype)
    elif scenario == 'joint_drop_30':
        mask = torch.rand((sk.shape[0], sk.shape[2]), generator=g, device=sk.device) < 0.30
        sk = sk.masked_fill(mask[:,None,:,None], 0)
    elif scenario == 'joint_drop_50':
        mask = torch.rand((sk.shape[0], sk.shape[2]), generator=g, device=sk.device) < 0.50
        sk = sk.masked_fill(mask[:,None,:,None], 0)
    elif scenario.startswith('imu_delay_'):
        code = scenario.replace('imu_delay_', '')
        sign = 1 if code.startswith('p') else -1
        amount = int(code[1:])
        imu = temporal_delay_zero_pad(imu, sign * amount)
    else:
        raise ValueError(scenario)
    return sk, imu


@torch.no_grad()
def evaluate(model, loader, scenario='clean', device='cuda', scenario_seed=20260928, return_predictions=False):
    model.eval()
    ys, ps, ids = [], [], []
    rs, ri, alphas = [], [], []
    aux_s_ok, aux_i_ok = [], []

    for bi, batch in enumerate(loader):
        sk = batch['skeleton'].to(device, non_blocking=True)
        imu = batch['imu'].to(device, non_blocking=True)
        y = batch['label'].to(device, non_blocking=True)
        batch_seed = int(scenario_seed) + bi * 1000003
        sk, imu = apply_scenario(sk, imu, scenario, seed=batch_seed)
        out = model(sk, imu)
        pred = out['logits'].argmax(-1)
        ys.extend(y.cpu().tolist())
        ps.extend(pred.cpu().tolist())
        ids.extend(list(batch['sample_id']))

        if 'r_skel' in out:
            rs.extend(out['r_skel'].mean((1,2)).cpu().tolist())
            ri.extend(out['r_imu'].mean((1,2)).cpu().tolist())
            alphas.extend(out['alpha'].mean(1).cpu().tolist())
            aux_s = out['aux_skel_logits'].argmax(-1)
            aux_i = out['aux_imu_logits'].argmax(-1)
            aux_s_ok.extend((aux_s == y).float().cpu().tolist())
            aux_i_ok.extend((aux_i == y).float().cpu().tolist())

    result = {
        'scenario': scenario,
        'top1': accuracy_score(ys, ps),
        'macro_f1': f1_score(ys, ps, average='macro', zero_division=0),
    }
    if rs:
        a = np.asarray(alphas)
        result.update({
            'mean_r_skel': float(np.mean(rs)),
            'mean_r_imu': float(np.mean(ri)),
            'mean_alpha_skel': float(a[:,0].mean()),
            'mean_alpha_imu': float(a[:,1].mean()),
            'aux_skel_acc': float(np.mean(aux_s_ok)),
            'aux_imu_acc': float(np.mean(aux_i_ok)),
        })
    if return_predictions:
        result['sample_id'] = ids
        result['y_true'] = ys
        result['y_pred'] = ps
    return result


def evaluate_shift_average(model, loader, amount, device='cuda'):
    plus = evaluate(model, loader, f'imu_delay_p{amount}', device=device)
    minus = evaluate(model, loader, f'imu_delay_m{amount}', device=device)
    out = {
        'scenario': f'imu_delay_pm{amount}',
        'top1': 0.5*(plus['top1']+minus['top1']),
        'macro_f1': 0.5*(plus['macro_f1']+minus['macro_f1']),
    }
    for key in ['mean_r_skel','mean_r_imu','mean_alpha_skel','mean_alpha_imu','aux_skel_acc','aux_imu_acc']:
        if key in plus and key in minus:
            out[key] = 0.5*(plus[key]+minus[key])
    return out


def evaluate_repeated_corruption(model, loader, scenario, seeds, device='cuda'):
    rows = []
    for s in seeds:
        r = evaluate(model, loader, scenario, device=device, scenario_seed=int(s))
        r['corruption_seed'] = int(s)
        rows.append(r)
    df = pd.DataFrame(rows)
    summary = {
        'scenario': scenario,
        'top1': float(df.top1.mean()),
        'top1_corruption_std': float(df.top1.std(ddof=1)) if len(df)>1 else 0.0,
        'macro_f1': float(df.macro_f1.mean()),
        'macro_f1_corruption_std': float(df.macro_f1.std(ddof=1)) if len(df)>1 else 0.0,
    }
    # Aggregate reliability diagnostics if present.
    for c in ['mean_r_skel','mean_r_imu','mean_alpha_skel','mean_alpha_imu','aux_skel_acc','aux_imu_acc']:
        if c in df.columns:
            summary[c] = float(df[c].mean())
    return summary, df


def train_model(model, train_loader, val_loader, out_dir, epochs=40, lr=3e-4,
                mode='clean', device='cuda', seed=42, force=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir/'best.pt'
    hist_path = out_dir/'history.csv'
    if ckpt_path.exists() and hist_path.exists() and not force:
        print('Reuse existing development checkpoint:', ckpt_path)
        return pd.read_csv(hist_path)

    seed_everything(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs,1))
    scaler = torch.amp.GradScaler('cuda', enabled=device.startswith('cuda'))
    hist, best = [], -1.0

    for ep in range(1, epochs+1):
        model.train(); sums = {}; n = 0
        for batch in train_loader:
            sk = batch['skeleton'].to(device, non_blocking=True)
            imu = batch['imu'].to(device, non_blocking=True)
            y = batch['label'].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=device.startswith('cuda')):
                if mode == 'clean':
                    out = model(sk, imu); loss = F.cross_entropy(out['logits'], y); parts={'clean_cls':loss}
                elif mode == 'late_robust':
                    loss, parts = dual_view_late_loss(model, sk, imu, y)
                elif mode == 'rata_v2':
                    loss, parts = dual_view_rata_loss(model, sk, imu, y)
                else:
                    raise ValueError(mode)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            bs = y.shape[0]; n += bs
            sums['loss'] = sums.get('loss',0.0) + float(loss.detach())*bs
            for k,v in parts.items(): sums[k] = sums.get(k,0.0) + float(v.detach())*bs
        sched.step()
        val = evaluate(model, val_loader, 'clean', device=device)
        row = {'epoch':ep, 'train_loss':sums['loss']/n, 'val_top1':val['top1'],
               'val_macro_f1':val['macro_f1'], 'lr':opt.param_groups[0]['lr']}
        for k,v in sums.items():
            if k!='loss': row[k]=v/n
        hist.append(row)
        print(f"Epoch {ep:03d}/{epochs} | loss {row['train_loss']:.4f} | val top1 {100*row['val_top1']:.2f}% | F1 {100*row['val_macro_f1']:.2f}%")
        pd.DataFrame(hist).to_csv(hist_path, index=False)
        if row['val_top1'] > best:
            best = row['val_top1']
            torch.save({'model':model.state_dict(),'epoch':ep,'val_top1':best,'mode':mode,'seed':seed}, ckpt_path)
    return pd.DataFrame(hist)


def train_fixed(model, train_loader, epochs=100, lr=3e-4, mode='clean', seed=0, device='cuda'):
    # IMPORTANT: seed_everything(seed) must also have been called BEFORE model construction.
    seed_everything(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs,1))
    scaler = torch.amp.GradScaler('cuda', enabled=device.startswith('cuda'))
    for ep in range(1,epochs+1):
        model.train()
        for batch in train_loader:
            sk=batch['skeleton'].to(device); imu=batch['imu'].to(device); y=batch['label'].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=device.startswith('cuda')):
                if mode=='clean':
                    out=model(sk,imu); loss=F.cross_entropy(out['logits'],y)
                elif mode=='late_robust':
                    loss,_=dual_view_late_loss(model,sk,imu,y)
                elif mode=='rata_v2':
                    loss,_=dual_view_rata_loss(model,sk,imu,y)
                else: raise ValueError(mode)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            scaler.step(opt); scaler.update()
        sched.step()
        if ep==1 or ep%20==0 or ep==epochs:
            print(f'seed {seed} | epoch {ep}/{epochs}')
    return model


def load_best(model, path, device='cuda'):
    ckpt=torch.load(path,map_location=device)
    model.load_state_dict(ckpt['model'])
    return model.to(device)


def save_model(model, path, metadata=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.state_dict(),'metadata':metadata or {}},path)


def load_saved(model, path, device='cuda'):
    ckpt=torch.load(path,map_location=device)
    model.load_state_dict(ckpt['model'])
    return model.to(device)
