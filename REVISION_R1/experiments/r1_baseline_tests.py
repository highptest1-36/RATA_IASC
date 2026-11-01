"""Paired tests of RATA-MAR against the re-implemented TMC-style and QMF-style baselines (Comment 10).
Inference only on the checkpoints in outputs/gpu_train/final_models; same scenarios, corruption seeds, bootstrap and
permutation test, and Holm correction as Table stats (stage B of r1_inference_analyses.py).
Output: outputs/<dev>/B_paired_tests_baselines.csv"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import torch
import r1_common as C
import r1_inference_analyses as A
import r1_train_variants_gpu as V

dev = C.device_name()
out = C.OUT / dev
sk, im, y, _ = C.load_test_tensors(dev)
yn = y.cpu().numpy()
store = dict(np.load(out / 'A_correct_per_sample.npz'))
scen = A.ORIG_DET + A.ORIG_STOCH + [f'imu_delay_{d}{a}' for a in A.DELAYS for d in ('p', 'm')]
ref = pd.read_csv(V.OUT / 'variants_per_seed.csv')
for name, cls in (('TMC-style', V.TMCStyle), ('QMF-style', V.QMFStyle)):
    for s in C.SEEDS:
        m = cls()
        m.load_state_dict(torch.load(V.CKPT / f'{name}_seed{s}.pt', map_location=dev)['model'])
        m = m.to(dev).eval()
        for sc in scen:
            seeds = C.CORRUPTION_SEEDS if sc in A.ORIG_STOCH else [20260928]
            store[f'{name}|{s}|{sc}'] = np.mean(
                [C.predict_original_scenario(m, sk, im, sc, cs)['pred'] == yn for cs in seeds], 0)
        # sanity check against the accuracies recorded during training
        r = ref[(ref.variant == name) & (ref.seed == s)].iloc[0]
        for sc, col in (('clean', 'clean_top1'), ('missing_imu', 'missing_imu'), ('skeleton_noise', 'skeleton_noise')):
            assert abs(store[f'{name}|{s}|{sc}'].mean() - r[col]) < 1e-9, (name, s, sc)
        A.log(name, 'seed', s, 'done')

scenarios = ['clean', 'missing_skeleton', 'missing_imu', 'skeleton_noise', 'imu_noise', 'joint_drop_30',
             'delay_pm4', 'delay_pm8', 'delay_pm16']
rows = []
for b in ('TMC-style', 'QMF-style'):
    for sc in scenarios:
        va, sa = A._scenario_vector(store, 'RATA-MAR', sc)
        vb, sb = A._scenario_vector(store, b, sc)
        dm, lo, hi, p = A.paired_test(va, vb)
        rows.append({'model_a': 'RATA-MAR', 'model_b': b, 'scenario': sc, 'acc_a': va.mean() * 100,
                     'acc_b': vb.mean() * 100, 'delta_pp': dm * 100, 'ci_lo_pp': lo * 100, 'ci_hi_pp': hi * 100,
                     'p_perm': p, 'seed_wins_a': int(sum(x > z for x, z in zip(sa, sb))),
                     'seed_wins_b': int(sum(z > x for x, z in zip(sa, sb)))})
df = pd.DataFrame(rows)
df['p_holm'] = np.nan
for _, g in df.groupby('model_b'):
    running = 0.0
    for k, ix in enumerate(g.p_perm.sort_values().index):
        running = max(running, min(1.0, (len(g) - k) * df.loc[ix, 'p_perm']))
        df.loc[ix, 'p_holm'] = running
df.to_csv(out / 'B_paired_tests_baselines.csv', index=False)
print(df.round(3).to_string())
