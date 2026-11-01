
from __future__ import annotations
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class UTDataset(Dataset):
    def __init__(self, df, sk_npy, imu_npy):
        self.df = df.reset_index(drop=True).copy()
        self.sk = np.load(sk_npy, mmap_mode="r")
        self.im = np.load(imu_npy, mmap_mode="r")
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        r = int(row.row_index)
        return {
            "skeleton": torch.from_numpy(np.asarray(self.sk[r], dtype=np.float32)),
            "imu": torch.from_numpy(np.asarray(self.im[r], dtype=np.float32)),
            "label": torch.tensor(int(row.label), dtype=torch.long),
            "sample_id": row.sample_id,
        }

def official_split(manifest_csv, val_fraction=0.15, seed=42):
    df = pd.read_csv(manifest_csv)
    train_pool = df[df.subject.isin([1,3,5,7])].copy()
    test = df[df.subject.isin([2,4,6,8])].copy()

    tr_idx, va_idx = train_test_split(
        np.arange(len(train_pool)),
        test_size=val_fraction,
        random_state=seed,
        stratify=train_pool.label.values,
    )
    train = train_pool.iloc[tr_idx].copy()
    val = train_pool.iloc[va_idx].copy()

    print("Development official-XSub split:",
          {"train":len(train),"val":len(val),"test":len(test)})
    print("Train/val subjects:", sorted(train_pool.subject.unique()))
    print("Test subjects:", sorted(test.subject.unique()))
    return train, val, test

def official_train_test(manifest_csv):
    df = pd.read_csv(manifest_csv)
    train = df[df.subject.isin([1,3,5,7])].copy()
    test = df[df.subject.isin([2,4,6,8])].copy()
    return train, test

def make_loader(df, sk_npy, imu_npy, batch=64, shuffle=False, workers=2):
    ds = UTDataset(df, sk_npy, imu_npy)
    return DataLoader(
        ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
        pin_memory=True, persistent_workers=workers>0
    )

def corrupt_train(sk, imu):
    B = sk.shape[0]
    device = sk.device
    sk = sk.clone()
    imu = imu.clone()
    q = torch.ones(B,2,device=device)

    for b in range(B):
        u = random.random()
        if u < 0.12:
            sk[b].zero_(); q[b,0] = 0.0
        elif u < 0.26:
            m = torch.rand(sk.shape[2],device=device) < 0.30
            sk[b,:,m,:] = 0; q[b,0] = 0.55
        elif u < 0.40:
            sk[b] += 0.08 * torch.randn_like(sk[b]); q[b,0] = 0.65

        u = random.random()
        if u < 0.12:
            imu[b].zero_(); q[b,1] = 0.0
        elif u < 0.26:
            start = random.randint(0, max(0, imu.shape[1]-16))
            imu[b,start:start+16] = 0; q[b,1] = 0.60
        elif u < 0.40:
            imu[b] += 0.20 * torch.randn_like(imu[b]); q[b,1] = 0.65

        if (q[b] == 0).all():
            q[b,1] = 1.0

        if random.random() < 0.30:
            imu[b] = torch.roll(imu[b], random.randint(-8,8), dims=0)

    return sk, imu, q

def apply_scenario(sk, imu, scenario):
    sk = sk.clone()
    imu = imu.clone()

    if scenario == "clean":
        pass
    elif scenario == "missing_skeleton":
        sk.zero_()
    elif scenario == "missing_imu":
        imu.zero_()
    elif scenario == "skeleton_noise":
        sk += 0.10 * torch.randn_like(sk)
    elif scenario == "imu_noise":
        imu += 0.25 * torch.randn_like(imu)
    elif scenario == "joint_drop_30":
        for b in range(sk.shape[0]):
            m = torch.rand(sk.shape[2],device=sk.device) < 0.30
            sk[b,:,m,:] = 0
    elif scenario.startswith("imu_shift"):
        shift = int(scenario.replace("imu_shift",""))
        imu = torch.roll(imu, shift, dims=1)
    else:
        raise ValueError(scenario)

    return sk, imu

@torch.no_grad()
def evaluate(model, loader, scenario="clean", device="cuda"):
    model.eval()
    ys, ps = [], []
    rs, ri = [], []

    for batch in loader:
        sk = batch["skeleton"].to(device, non_blocking=True)
        imu = batch["imu"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        sk, imu = apply_scenario(sk, imu, scenario)
        out = model(sk, imu)
        pred = out["logits"].argmax(-1)

        ys += y.cpu().tolist()
        ps += pred.cpu().tolist()

        if "r_skel" in out:
            rs += out["r_skel"].mean((1,2)).cpu().tolist()
            ri += out["r_imu"].mean((1,2)).cpu().tolist()

    result = {
        "scenario": scenario,
        "top1": accuracy_score(ys, ps),
        "macro_f1": f1_score(ys, ps, average="macro", zero_division=0),
    }
    if rs:
        result["mean_r_skel"] = float(np.mean(rs))
        result["mean_r_imu"] = float(np.mean(ri))
    return result

def train_model(model, train_loader, val_loader, out_dir,
                epochs=60, lr=3e-4, use_corruption=False,
                rel_weight=0.2, device="cuda"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs,1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))

    hist = []
    best = -1.0

    for ep in range(1, epochs+1):
        model.train()
        sums = {"loss":0.0,"cls":0.0,"rel":0.0,"n":0}

        for batch in train_loader:
            sk = batch["skeleton"].to(device, non_blocking=True)
            imu = batch["imu"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            if use_corruption:
                sk, imu, q = corrupt_train(sk, imu)
            else:
                q = torch.ones(sk.shape[0],2,device=device)

            opt.zero_grad(set_to_none=True)

            with torch.autocast("cuda",dtype=torch.float16,enabled=device.startswith("cuda")):
                out = model(sk, imu)
                lcls = F.cross_entropy(out["logits"], y)

                lrel = torch.tensor(0.0, device=device)
                if "r_skel" in out:
                    pq = torch.cat([
                        out["r_skel"].mean(1),
                        out["r_imu"].mean(1)
                    ], dim=1)
                    lrel = F.mse_loss(pq, q)

                loss = lcls + rel_weight * lrel

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            scaler.step(opt)
            scaler.update()

            bs = y.shape[0]
            sums["loss"] += float(loss.detach()) * bs
            sums["cls"] += float(lcls.detach()) * bs
            sums["rel"] += float(lrel.detach()) * bs
            sums["n"] += bs

        sched.step()
        val = evaluate(model, val_loader, "clean", device)

        row = {
            "epoch": ep,
            "train_loss": sums["loss"]/sums["n"],
            "cls_loss": sums["cls"]/sums["n"],
            "rel_loss": sums["rel"]/sums["n"],
            "val_top1": val["top1"],
            "val_macro_f1": val["macro_f1"],
            "lr": opt.param_groups[0]["lr"],
        }
        hist.append(row)

        print(
            f"Epoch {ep:03d}/{epochs} | "
            f"loss {row['train_loss']:.4f} | cls {row['cls_loss']:.4f} | "
            f"rel {row['rel_loss']:.4f} | "
            f"val top1 {100*row['val_top1']:.2f}% | "
            f"F1 {100*row['val_macro_f1']:.2f}%"
        )

        pd.DataFrame(hist).to_csv(out_dir/"history.csv",index=False)

        if row["val_top1"] > best:
            best = row["val_top1"]
            torch.save(
                {"model":model.state_dict(),"epoch":ep,"val_top1":best},
                out_dir/"best.pt"
            )

    return pd.DataFrame(hist)

def train_fixed(model, train_loader, epochs=100, lr=3e-4,
                use_corruption=False, rel_weight=0.2, device="cuda"):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs,1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))

    for ep in range(1,epochs+1):
        model.train()
        for batch in train_loader:
            sk = batch["skeleton"].to(device)
            imu = batch["imu"].to(device)
            y = batch["label"].to(device)

            if use_corruption:
                sk, imu, q = corrupt_train(sk, imu)
            else:
                q = torch.ones(sk.shape[0],2,device=device)

            opt.zero_grad(set_to_none=True)

            with torch.autocast("cuda",dtype=torch.float16,enabled=device.startswith("cuda")):
                out = model(sk,imu)
                lcls = F.cross_entropy(out["logits"],y)
                lrel = torch.tensor(0.0,device=device)
                if "r_skel" in out:
                    pq = torch.cat([
                        out["r_skel"].mean(1),
                        out["r_imu"].mean(1)
                    ],1)
                    lrel = F.mse_loss(pq,q)
                loss = lcls + rel_weight*lrel

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            scaler.step(opt)
            scaler.update()

        sched.step()
        if ep == 1 or ep % 20 == 0 or ep == epochs:
            print(f"fixed train epoch {ep}/{epochs}")

    return model

def load_best(model, path, device="cuda"):
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model"])
    return model.to(device)
