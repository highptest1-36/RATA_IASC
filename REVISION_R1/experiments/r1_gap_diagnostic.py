"""Diagnostic for zero-filled temporal gaps (skeleton tracking loss, intermittent loss of both streams).
Compares the models on the gapped input and on the same input after forward-filling each gap with the last valid
frame/sample (a standard preprocessing step). Inference only. Output: outputs/<dev>/G_gap_diagnostic.csv"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import torch
import r1_common as C


def forward_fill(x):
    """x [N,T,...]: frames whose values are all zero are replaced by the last non-zero frame (vectorized)."""
    flat = x.reshape(x.shape[0], x.shape[1], -1)
    zero = (flat.abs().sum(-1) == 0)
    t = torch.arange(x.shape[1], device=x.device).expand_as(zero)
    last = torch.where(zero, torch.full_like(t, -1), t).cummax(dim=1).values
    filled = flat.gather(1, last.clamp(min=0).unsqueeze(-1).expand_as(flat))
    out = torch.where((last >= 0).unsqueeze(-1), filled, flat)
    return out.reshape(x.shape)


dev = C.device_name()
sk, im, y, _ = C.load_test_tensors(dev)
yn = y.cpu().numpy()
rows = []
for name in ['LateFusion-Robust', 'RATA-MAR', 'NoAlignment', 'NoReliability']:
    for s in C.SEEDS:
        m = C.load_model(name, s, dev)
        for sc in ['skel_tracking_loss', 'both_intermittent', 'imu_intermittent']:
            for cs in C.CORRUPTION_SEEDS:
                s2, i2 = C.apply_extended(sk, im, y, sc, cs)
                for fill in (False, True):
                    a, b = (forward_fill(s2), forward_fill(i2)) if fill else (s2, i2)
                    o = C.forward_all(m, a, b)
                    r = {'model': name, 'seed': s, 'scenario': sc, 'corruption_seed': cs, 'forward_fill': fill,
                         'top1': C.accuracy(o['logits'].argmax(-1).numpy(), yn)}
                    if 'r_skel' in o:
                        r.update(r_skel=float(o['r_skel'].mean()), alpha_skel=float(o['alpha'][:, 0].mean()))
                    rows.append(r)
        print(name, s, 'done', flush=True)
df = pd.DataFrame(rows)
df.to_csv(C.OUT / dev / 'G_gap_diagnostic.csv', index=False)
print((df.groupby(['scenario', 'forward_fill', 'model']).top1.mean().unstack() * 100).round(1))
