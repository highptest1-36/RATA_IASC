"""Comment 12 diagnostic: is the missing-IMU accuracy of the full model limited by the residual reliability
(~0.28) that the estimator assigns to a zero-filled IMU stream? At inference only, the IMU reliability logits are
overridden by a constant (oracle gate) and the missing-IMU accuracy is re-measured. Output: outputs/<dev>/F_oracle_gate.csv"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import torch
import r1_common as C


class ConstLogit(torch.nn.Module):
    def __init__(self, value):
        super().__init__(); self.value = value

    def forward(self, tokens, descriptors):
        return torch.full(tokens.shape[:2] + (1,), self.value, device=tokens.device, dtype=tokens.dtype)


dev = C.device_name()
sk, im, y, _ = C.load_test_tensors(dev)
yn = y.cpu().numpy()
rows = []
for s in C.SEEDS:
    for setting, logit in (('learned', None), ('r_imu=0.10', -2.197), ('r_imu=0.01', -4.595), ('r_imu=1e-4 (clamp)', -9.21)):
        m = C.load_model('RATA-MAR', s, dev)
        if logit is not None:
            m.rel_i = ConstLogit(logit)
        o = C.forward_all(m, sk, torch.zeros_like(im))
        rows.append({'seed': s, 'setting': setting, 'missing_imu_top1': C.accuracy(o['logits'].argmax(-1).numpy(), yn),
                     'alpha_imu': float(o['alpha'][:, 1].mean()), 'r_imu': float(o['r_imu'].mean())})
        print(rows[-1], flush=True)
out = C.OUT / dev
out.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(out / 'F_oracle_gate.csv', index=False)
print(df.groupby('setting')[['missing_imu_top1', 'alpha_imu']].agg(['mean', 'std']))
