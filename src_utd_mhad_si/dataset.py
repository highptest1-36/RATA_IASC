
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm.auto import tqdm

ACTIONS_27 = [
    "right_arm_swipe_left",
    "right_arm_swipe_right",
    "right_hand_wave",
    "two_hand_front_clap",
    "right_arm_throw",
    "cross_arms_chest",
    "basketball_shoot",
    "right_hand_draw_x",
    "right_hand_draw_circle_clockwise",
    "right_hand_draw_circle_counterclockwise",
    "draw_triangle",
    "bowling_right_hand",
    "front_boxing",
    "baseball_swing_right",
    "tennis_right_forehand",
    "arm_curl_two_arms",
    "tennis_serve",
    "two_hand_push",
    "right_hand_knock_door",
    "right_hand_catch_object",
    "right_hand_pickup_throw",
    "jogging_in_place",
    "walking_in_place",
    "sit_to_stand",
    "stand_to_sit",
    "forward_lunge_left",
    "squat_arms_stretched",
]

SAMPLE_RE = re.compile(r"a(\d+)_s(\d+)_t(\d+)", re.I)

# UTD-MHAD original skeleton uses Kinect v1 20-joint body model.
# 0-based graph used only for visualization/model adjacency.
BONES_20 = [
    (0,1), (1,2), (2,3),         # head/shoulder/spine chain (generic topology)
    (2,4), (4,5), (5,6), (6,7), # left arm
    (2,8), (8,9), (9,10), (10,11), # right arm
    (0,12), (12,13), (13,14), (14,15), # left leg
    (0,16), (16,17), (17,18), (18,19), # right leg
]

def parse_sample_id(name: str):
    m = SAMPLE_RE.search(Path(name).stem)
    if not m:
        return None
    a, s, t = map(int, m.groups())
    return f"a{a}_s{s}_t{t}", a, s, t

def numeric_arrays(mat_path):
    d = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    out = []
    for k, v in d.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number) and v.size > 0:
            out.append((k, np.asarray(v)))
    return out

def inspect_mat(mat_path):
    return [(k, tuple(v.shape), str(v.dtype), int(v.size)) for k, v in numeric_arrays(mat_path)]

def _resample_time(x: np.ndarray, target_t: int):
    x = np.asarray(x, dtype=np.float32)
    if x.shape[0] == target_t:
        return x
    if x.shape[0] <= 1:
        return np.repeat(x[:1], target_t, axis=0)
    old = np.linspace(0.0, 1.0, x.shape[0], dtype=np.float32)
    new = np.linspace(0.0, 1.0, target_t, dtype=np.float32)
    flat = x.reshape(x.shape[0], -1)
    out = np.empty((target_t, flat.shape[1]), dtype=np.float32)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(new, old, flat[:, j])
    return out.reshape((target_t,) + x.shape[1:])

def canonical_skeleton(mat_path, target_t=64):
    arrs = numeric_arrays(mat_path)
    if not arrs:
        raise ValueError(f"No numeric arrays in {mat_path}")

    # Prefer the largest array containing a 20-joint and 3-coordinate axis.
    candidates = []
    for k, a in arrs:
        x = np.asarray(a).squeeze()
        score = 0
        if x.ndim == 3 and 20 in x.shape and 3 in x.shape:
            score += 1000
        if x.ndim == 2 and 60 in x.shape:
            score += 800
        score += min(x.size, 100000) / 100000
        candidates.append((score, k, x))
    candidates.sort(reverse=True, key=lambda z: z[0])

    for _, key, x in candidates:
        if x.ndim == 3 and 20 in x.shape and 3 in x.shape:
            shape = list(x.shape)
            ja = shape.index(20)
            ca = shape.index(3)
            ta = [i for i in range(3) if i not in (ja, ca)][0]
            x = np.transpose(x, (ta, ja, ca))
            break
        if x.ndim == 2 and 60 in x.shape:
            if x.shape[0] == 60:
                x = x.T.reshape(-1, 20, 3)
            else:
                x = x.reshape(-1, 20, 3)
            break
    else:
        raise ValueError(f"Could not infer original UTD-MHAD 20x3 skeleton in {mat_path}: {inspect_mat(mat_path)}")

    x = _resample_time(x.astype(np.float32), target_t)

    # Translation normalization: use joint 0 as sequence root.
    x = x - x[:, 0:1, :]

    # Robust scale normalization.
    radius = np.linalg.norm(x, axis=-1)
    valid = radius[radius > 1e-8]
    scale = np.percentile(valid, 90) if valid.size else 1.0
    x /= max(float(scale), 1e-4)

    return x.astype(np.float32)

def canonical_inertial(mat_path, target_t=64):
    arrs = numeric_arrays(mat_path)
    if not arrs:
        raise ValueError(f"No numeric arrays in {mat_path}")

    candidates = []
    for k, a in arrs:
        x = np.asarray(a).squeeze()
        if x.ndim != 2:
            continue
        score = 0
        if 6 in x.shape:
            score += 1000
        if "inert" in k.lower():
            score += 300
        score += min(x.size, 100000) / 100000
        candidates.append((score, k, x))
    if not candidates:
        raise ValueError(f"No usable inertial array in {mat_path}: {inspect_mat(mat_path)}")

    candidates.sort(reverse=True, key=lambda z: z[0])
    _, key, x = candidates[0]

    if x.shape[1] == 6:
        pass
    elif x.shape[0] == 6:
        x = x.T
    elif x.shape[1] > 6:
        x = x[:, :6]
    elif x.shape[0] > 6:
        x = x.T[:, :6]
    else:
        raise ValueError(f"Cannot infer 6 inertial channels from {x.shape} in {mat_path}")

    x = _resample_time(x.astype(np.float32), target_t)

    # Per-sequence z-score stabilization.
    mu = x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True)
    x = (x - mu) / np.maximum(sd, 1e-5)

    return x.astype(np.float32)

def build_manifest(extracted_root, out_csv):
    root = Path(extracted_root)

    skel, imu = {}, {}

    for p in root.rglob("*.mat"):
        parsed = parse_sample_id(p.name)
        if not parsed:
            continue
        sid, a, s, t = parsed
        low = (p.name + " " + str(p.parent)).lower()

        if "skeleton" in low or "skel" in low:
            skel.setdefault(sid, str(p))
        elif "inertial" in low:
            imu.setdefault(sid, str(p))

    common = sorted(
        set(skel) & set(imu),
        key=lambda sid: tuple(map(int, SAMPLE_RE.search(sid).groups()))
    )

    rows = []
    for ridx, sid in enumerate(common):
        _, a, s, t = parse_sample_id(sid)
        rows.append({
            "row_index": ridx,
            "sample_id": sid,
            "action": a,
            "label": a - 1,
            "action_name": ACTIONS_27[a - 1],
            "subject": s,
            "trial": t,
            "skeleton_path": skel[sid],
            "inertial_path": imu[sid],
            "official_xsub": "train" if s in {1,3,5,7} else "test",
        })

    df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("Skeleton files indexed:", len(skel))
    print("Inertial files indexed:", len(imu))
    print("Paired samples:", len(df))
    print("Actions:", sorted(df.action.unique().tolist()) if len(df) else [])
    print("Subjects:", sorted(df.subject.unique().tolist()) if len(df) else [])
    print("Trials:", sorted(df.trial.unique().tolist()) if len(df) else [])
    print("Official cross-subject:", df.official_xsub.value_counts().to_dict() if len(df) else {})
    return out_csv

def preprocess(manifest_csv, out_dir, target_t=64, overwrite=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(manifest_csv)

    sk_path = out_dir / f"utd_mhad_skeleton_T{target_t}_f16.npy"
    im_path = out_dir / f"utd_mhad_inertial_T{target_t}_f16.npy"
    meta_path = out_dir / f"utd_mhad_preprocess_T{target_t}.json"

    if sk_path.exists() and im_path.exists() and meta_path.exists() and not overwrite:
        print("Processed arrays already exist.")
        return sk_path, im_path

    sk = np.lib.format.open_memmap(
        sk_path, mode="w+", dtype=np.float16,
        shape=(len(df), target_t, 20, 3)
    )
    im = np.lib.format.open_memmap(
        im_path, mode="w+", dtype=np.float16,
        shape=(len(df), target_t, 6)
    )

    errors = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Preprocess Skeleton + IMU"):
        i = int(row.row_index)
        try:
            sk[i] = canonical_skeleton(row.skeleton_path, target_t).astype(np.float16)
            im[i] = canonical_inertial(row.inertial_path, target_t).astype(np.float16)
        except Exception as e:
            errors.append({"row_index": i, "sample_id": row.sample_id, "error": repr(e)})

    sk.flush()
    im.flush()

    meta_path.write_text(json.dumps({
        "rows": len(df),
        "error_count": len(errors),
        "errors": errors
    }, indent=2), encoding="utf-8")

    print("Preprocess errors:", len(errors))
    if errors:
        print("First errors:", errors[:5])

    return sk_path, im_path
