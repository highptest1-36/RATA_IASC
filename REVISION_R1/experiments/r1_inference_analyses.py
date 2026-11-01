"""Revision-1 inference-only analyses (no training; frozen v2.1 checkpoints).

Stages (run all by default; each stage resumes from its saved outputs):
  A  reproduce   per-sample predictions for every original test scenario + check against the paper CSVs
  B  stats       paired bootstrap 95% CI and sign-flip permutation tests (Comment 10, 12)
  C  extended    unseen / realistic / simultaneous / conflicting degradations (Comments 3, 5, 6, 8, 13, 14)
  D  delay       fine delay sweep on both streams + cross-attention offset analysis (Comment 2)
  E  detect      reliability vs. auxiliary-head disagreement as degradation detectors; selective prediction (5, 6)

Usage:  python -I r1_inference_analyses.py [--stages ABCDE] [--force]
Outputs: REVISION_R1/experiments/outputs/<device>/...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import r1_common as C

FUSION = ['LateFusion-Clean', 'LateFusion-Robust', 'RATA-MAR', 'NoAlignment', 'NoReliability']
RELIABILITY_MODELS = ['RATA-MAR', 'NoAlignment']
ORIG_DET = ['clean', 'missing_skeleton', 'missing_imu']
ORIG_STOCH = ['skeleton_noise', 'imu_noise', 'joint_drop_30']
DELAYS = [4, 8, 16]
SWEEP = [0, 2, 4, 6, 8, 12, 16, 20, 24]
SKEL_SWEEP = [8, 16]


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ------------------------------------------------------------------------------------- A
def stage_reproduce(out, dev, sk, im, y):
    path = out / 'A_correct_per_sample.npz'
    if path.exists():
        log('A: reuse', path)
        return dict(np.load(path))
    yn = y.cpu().numpy()
    store, rows = {}, []
    for name in FUSION + ['SkeletonOnly', 'IMUOnly']:
        for s in C.SEEDS:
            m = C.load_model(name, s, dev)
            scen = ['clean'] if name in ('SkeletonOnly', 'IMUOnly') else ORIG_DET + ORIG_STOCH + [
                f'imu_delay_{d}{a}' for a in DELAYS for d in ('p', 'm')]
            for sc in scen:
                seeds = C.CORRUPTION_SEEDS if sc in ORIG_STOCH else [20260928]
                corr = []
                for cs in seeds:
                    p = C.predict_original_scenario(m, sk, im, sc, cs)['pred']
                    corr.append(p == yn)
                    rows.append({'model': name, 'seed': s, 'scenario': sc, 'corruption_seed': cs,
                                 'top1': C.accuracy(p, yn), 'macro_f1': C.macro_f1(p, yn)})
                store[f'{name}|{s}|{sc}'] = np.mean(corr, 0)
            log('A:', name, 'seed', s, 'done')
    np.savez_compressed(path, **store)
    pd.DataFrame(rows).to_csv(out / 'A_original_scenarios_per_run.csv', index=False)
    return store


def paper_comparison(out):
    """Compare re-evaluated accuracies with the per-seed CSVs used for the submitted paper."""
    runs = pd.read_csv(out / 'A_original_scenarios_per_run.csv')
    g = runs.groupby(['model', 'seed', 'scenario']).top1.mean().reset_index()
    piv = {}
    for _, r in g.iterrows():
        key = r.scenario
        if key.startswith('imu_delay_'):
            key = 'delay_pm' + key[11:]
        piv.setdefault((r.model, r.seed, key), []).append(r.top1)
    mine = {k: float(np.mean(v)) for k, v in piv.items()}
    paper = pd.read_csv(C.PROJECT / 'runs/RATA_MAR_SI_v21/tables/FINAL_XSUB_per_seed.csv')
    abl = pd.read_csv(C.PROJECT / 'runs/RATA_MAR_SI_v21/tables/FINAL_Ablation_per_seed.csv')
    name_map = {'RATA-MAR-SI-v2.1': 'RATA-MAR'}
    rows = []
    for _, r in paper.iterrows():
        mdl = name_map.get(r.model, r.model)
        for col in ['clean_top1', 'missing_skeleton', 'missing_imu', 'skeleton_noise', 'imu_noise',
                    'joint_drop_30', 'delay_pm4', 'delay_pm8', 'delay_pm16']:
            if pd.isna(r.get(col)):
                continue
            key = 'clean' if col == 'clean_top1' else col
            if (mdl, r.training_seed, key) in mine:
                rows.append({'model': mdl, 'seed': r.training_seed, 'metric': key, 'paper': r[col],
                             'rerun': mine[(mdl, r.training_seed, key)]})
    for _, r in abl[abl.variant != 'Full'].iterrows():
        for col in ['clean_top1', 'missing_skeleton', 'missing_imu', 'joint_drop_30', 'imu_noise',
                    'delay_pm4', 'delay_pm8', 'delay_pm16']:
            key = 'clean' if col == 'clean_top1' else col
            if (r.variant, r.training_seed, key) in mine:
                rows.append({'model': r.variant, 'seed': r.training_seed, 'metric': key, 'paper': r[col],
                             'rerun': mine[(r.variant, r.training_seed, key)]})
    df = pd.DataFrame(rows)
    df['abs_diff_pp'] = (df.paper - df.rerun).abs() * 100
    df.to_csv(out / 'A_reproduction_check.csv', index=False)
    det = df[~df.metric.isin(ORIG_STOCH)]
    log(f'A: reproduction | deterministic max |diff| = {det.abs_diff_pp.max():.3f} pp '
        f'(n={len(det)}, exact={int((det.abs_diff_pp < 1e-6).sum())}) | stochastic max |diff| = '
        f'{df[df.metric.isin(ORIG_STOCH)].abs_diff_pp.max():.3f} pp')
    return df


# ------------------------------------------------------------------------------------- B
def _scenario_vector(store, model, scen):
    """Per-sample correctness averaged over training seeds (and corruption seeds / both delay
    directions), plus the per-training-seed accuracies."""
    per_seed = []
    for s in C.SEEDS:
        if scen.startswith('delay_pm'):
            a = scen[8:]
            per_seed.append(np.mean([store[f'{model}|{s}|imu_delay_{d}{a}'] for d in ('p', 'm')], 0))
        else:
            per_seed.append(store[f'{model}|{s}|{scen}'])
    return np.mean(per_seed, 0), [float(v.mean()) for v in per_seed]


def paired_test(a, b, n_boot=10000, n_perm=10000, rng=None):
    rng = rng or np.random.default_rng(20260415)
    d = a - b
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm = np.abs((signs * d).mean(1))
    p = (1 + np.sum(perm >= abs(d.mean()) - 1e-12)) / (n_perm + 1)
    return d.mean(), lo, hi, p


def stage_stats(out, store):
    comps = [('RATA-MAR', 'LateFusion-Robust'), ('RATA-MAR', 'LateFusion-Clean'),
             ('RATA-MAR', 'NoAlignment'), ('RATA-MAR', 'NoReliability')]
    scen = ['clean', 'missing_skeleton', 'missing_imu', 'skeleton_noise', 'imu_noise', 'joint_drop_30',
            'delay_pm4', 'delay_pm8', 'delay_pm16']
    rows = []
    for a, b in comps:
        for sc in scen:
            va, sa = _scenario_vector(store, a, sc)
            vb, sb = _scenario_vector(store, b, sc)
            dm, lo, hi, p = paired_test(va, vb)
            wins = int(sum(x > z for x, z in zip(sa, sb)))
            rows.append({'model_a': a, 'model_b': b, 'scenario': sc, 'acc_a': va.mean() * 100,
                         'acc_b': vb.mean() * 100, 'delta_pp': dm * 100, 'ci_lo_pp': lo * 100,
                         'ci_hi_pp': hi * 100, 'p_perm': p, 'seed_wins_a': wins,
                         'per_seed_a': ';'.join(f'{x*100:.2f}' for x in sa),
                         'per_seed_b': ';'.join(f'{x*100:.2f}' for x in sb)})
    df = pd.DataFrame(rows)
    # Holm-Bonferroni within each comparison family
    df['p_holm'] = np.nan
    for (a, b), g in df.groupby(['model_a', 'model_b']):
        order = g.p_perm.sort_values().index
        m = len(order)
        running = 0.0
        for k, ix in enumerate(order):
            running = max(running, min(1.0, (m - k) * df.loc[ix, 'p_perm']))
            df.loc[ix, 'p_holm'] = running
    df.to_csv(out / 'B_paired_tests.csv', index=False)
    log('B: paired tests written')
    return df


# ------------------------------------------------------------------------------------- C
def stage_extended(out, dev, sk, im, y):
    path = out / 'C_extended_per_run.csv'
    if path.exists():
        log('C: reuse', path)
        return pd.read_csv(path)
    yn = y.cpu().numpy()
    rows, per_sample = [], {}
    for name in FUSION:
        for s in C.SEEDS:
            m = C.load_model(name, s, dev)
            for sc, (stream, stoch, _) in C.EXT_SCENARIOS.items():
                for cs in (C.CORRUPTION_SEEDS if stoch else [C.CORRUPTION_SEEDS[0]]):
                    s2, i2 = C.apply_extended(sk, im, y, sc, cs)
                    o = C.forward_all(m, s2, i2)
                    p = o['logits'].argmax(-1).numpy()
                    r = {'model': name, 'seed': s, 'scenario': sc, 'stream': stream, 'corruption_seed': cs,
                         'top1': C.accuracy(p, yn), 'macro_f1': C.macro_f1(p, yn)}
                    if 'r_skel' in o:
                        r.update(r_skel=float(o['r_skel'].mean()), r_imu=float(o['r_imu'].mean()),
                                 alpha_skel=float(o['alpha'][:, 0].mean()))
                    rows.append(r)
            log('C:', name, 'seed', s, 'done')
        pd.DataFrame(rows).to_csv(path, index=False)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------- D
def stage_delay(out, dev, sk, im, y):
    path = out / 'D_delay_sweep.csv'
    if path.exists():
        log('D: reuse', path)
        return
    yn = y.cpu().numpy()
    rows, att_rows, att_mats = [], [], {}
    for name in FUSION:
        for s in C.SEEDS:
            m = C.load_model(name, s, dev)
            for stream, values in (('IMU', SWEEP), ('Skeleton', [0] + SKEL_SWEEP)):
                for a in values:
                    for sign in ((1,) if a == 0 else (1, -1)):
                        d = sign * a
                        s2 = C.shift_stream(sk, d) if stream == 'Skeleton' else sk
                        i2 = C.shift_stream(im, d) if stream == 'IMU' else im
                        o = C.forward_all(m, s2, i2, keep_attention=(name == 'RATA-MAR' and stream == 'IMU'))
                        p = o['logits'].argmax(-1).numpy()
                        rows.append({'model': name, 'seed': s, 'stream': stream, 'shift': d,
                                     'abs_shift': a, 'top1': C.accuracy(p, yn)})
                        if 'att_s2i' in o:
                            A = o['att_s2i']                      # [N, Kq, Kk]
                            K = A.shape[-1]
                            off = (torch.arange(K).view(1, 1, K) - torch.arange(K).view(1, K, 1)).float()
                            exp_off = (A * off).sum(-1)           # [N, Kq]
                            interior = exp_off[:, 2:K - 2].mean().item()
                            att_rows.append({'seed': s, 'shift': d, 'expected_offset_tokens': exp_off.mean().item(),
                                             'expected_offset_interior': interior,
                                             'argmax_offset': (A.argmax(-1) - torch.arange(K)).float().mean().item(),
                                             'r_imu': float(o['r_imu'].mean()), 'alpha_skel': float(o['alpha'][:, 0].mean())})
                            if s == 0 and d in (0, 8, 16, -16, -8):
                                att_mats[f'shift_{d}'] = A.mean(0).numpy()
            log('D:', name, 'seed', s, 'done')
    pd.DataFrame(rows).to_csv(path, index=False)
    pd.DataFrame(att_rows).to_csv(out / 'D_attention_offset.csv', index=False)
    np.savez_compressed(out / 'D_attention_maps_seed0.npz', **att_mats)


# ------------------------------------------------------------------------------------- E
def _js(p, q, eps=1e-8):
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (torch.log(a + eps) - torch.log(b + eps))).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _scores(o):
    p_main = F.softmax(o['logits'], -1)
    ps, pi = F.softmax(o['aux_skel_logits'], -1), F.softmax(o['aux_imu_logits'], -1)
    rmin = torch.minimum(o['r_skel'], o['r_imu'])
    return {
        'low_reliability': (1 - rmin).numpy(),                  # 1 - min modality reliability
        'aux_disagreement': _js(ps, pi).numpy(),               # cross-modal (in)consistency
        'low_confidence': (1 - p_main.max(-1).values).numpy(),  # 1 - max softmax probability
    }


def stage_detect(out, dev, sk, im, y):
    yn = y.cpu().numpy()
    scen = ['missing_skeleton', 'missing_imu', 'imu_frozen', 'imu_bias', 'imu_drift', 'imu_gain',
            'imu_intermittent', 'skel_tracking_loss', 'skel_frozen', 'skel_arm_occlusion',
            'both_noise_strong', 'both_severe', 'imu_conflict', 'skel_conflict']
    sel_scen = ['clean', 'both_noise_strong', 'both_severe', 'both_intermittent', 'imu_conflict',
                'skel_conflict', 'missing_imu', 'imu_frozen']
    det_rows, sel_rows = [], []
    for s in C.SEEDS:
        m = C.load_model('RATA-MAR', s, dev)
        clean_o = C.forward_all(m, sk, im)
        clean_sc = _scores(clean_o)
        cache = {'clean': (clean_o, clean_sc)}
        for sc in sorted(set(scen + sel_scen) - {'clean'}):
            if sc in ('missing_skeleton', 'missing_imu'):
                s2, i2 = C.apply_scenario(sk, im, sc, seed=0)
            else:
                s2, i2 = C.apply_extended(sk, im, y, sc, C.CORRUPTION_SEEDS[0])
            o = C.forward_all(m, s2, i2)
            cache[sc] = (o, _scores(o))
        for sc in scen:
            _, cs = cache[sc]
            for k in clean_sc:
                lab = np.r_[np.zeros(len(yn)), np.ones(len(yn))]
                det_rows.append({'seed': s, 'scenario': sc, 'score': k,
                                 'auroc': roc_auc_score(lab, np.r_[clean_sc[k], cs[k]])})
        for sc in sel_scen:
            o, cs = cache[sc]
            correct = (o['logits'].argmax(-1).numpy() == yn)
            for k in ('low_confidence', 'low_reliability', 'aux_disagreement'):
                order = np.argsort(cs[k])            # most trusted first
                for cov in (1.0, 0.9, 0.8, 0.6):
                    n = int(round(cov * len(yn)))
                    sel_rows.append({'seed': s, 'scenario': sc, 'score': k, 'coverage': cov,
                                     'selective_acc': correct[order[:n]].mean()})
        log('E: seed', s, 'done')
    pd.DataFrame(det_rows).to_csv(out / 'E_detection_auroc.csv', index=False)
    pd.DataFrame(sel_rows).to_csv(out / 'E_selective_prediction.csv', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stages', default='ABCDE')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    dev = C.device_name()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    out = C.OUT / dev
    out.mkdir(parents=True, exist_ok=True)
    if args.force:
        for f in out.glob('*'):
            f.unlink()
    meta = {'device': dev, 'torch': torch.__version__,
            'gpu': torch.cuda.get_device_name(0) if dev == 'cuda' else None,
            'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    (out / 'run_info.json').write_text(json.dumps(meta, indent=2))
    log('device', meta)
    sk, im, y, _ = C.load_test_tensors(dev)
    store = None
    if 'A' in args.stages or 'B' in args.stages:
        store = stage_reproduce(out, dev, sk, im, y)
        paper_comparison(out)
    if 'B' in args.stages:
        stage_stats(out, store)
    if 'C' in args.stages:
        stage_extended(out, dev, sk, im, y)
    if 'D' in args.stages:
        stage_delay(out, dev, sk, im, y)
    if 'E' in args.stages:
        stage_detect(out, dev, sk, im, y)
    log('ALL DONE')


if __name__ == '__main__':
    main()
