"""Run stage C (extended stress tests) for a subset of models into a separate CSV (used to parallelize on CPU).
Usage: python r1_stageC_subset.py NoAlignment NoReliability"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import r1_common as C
import r1_inference_analyses as A

dev = C.device_name()
out = C.OUT / dev
sk, im, y, _ = C.load_test_tensors(dev)
yn = y.cpu().numpy()
for name in sys.argv[1:]:
    rows = []
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
        A.log('C-subset:', name, 'seed', s, 'done')
        pd.DataFrame(rows).to_csv(out / f'C_extended_part_{name}.csv', index=False)
