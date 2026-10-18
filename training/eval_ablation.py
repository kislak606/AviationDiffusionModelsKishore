"""
eval_ablation.py - evaluate Sinusoidal / RoPE-OG / RoPE-TS the same way.

Put it next to your train scripts and run from the folder you trained in
(so checkpoints_cfm, checkpoints_cfm_rope_og, checkpoints_cfm_rope_ts resolve):

    python eval_ablation.py --nc_path path\\to\\data.nc --subset 100000

Use the SAME --subset you trained with so val/test splits match.
Outputs: ablation_results/summary.json, per-sample .npz files, markdown table on stdout.
"""
import argparse, importlib, json, sys
from pathlib import Path

import numpy as np
import torch
import netCDF4 as nc

sys.path.insert(0, str(Path(__file__).parent.parent))
from DataLoaders.ADSBdataset import get_dataloaders
from models.cfm import euler_sample

RUNS = {
    "Sinusoidal": ("models.dit", {
        "best": "checkpoints_cfm/best.pt",
        "last": "checkpoints_cfm/last.pt"}),
    "RoPE-OG": ("models.dit_RoPE_original", {
        "best": "checkpoints_cfm_rope_og/best_rope_og.pt",
        "last": "checkpoints_cfm_rope_og/last_rope_og.pt"}),
    "RoPE-TS": ("models.dit_RoPE_timestamps", {
        "best": "checkpoints_cfm_rope_ts/best_rope_ts.pt",
        "last": "checkpoints_cfm_rope_ts/last_rope_ts.pt"}),
}


def load_model(module_name, ckpt_path, device):
    mod = importlib.import_module(module_name)
    model = mod.TrajectoryDiT(d_model=256, n_heads=8, n_layers=6).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["ema_state"])        # EMA weights, same as validate()
    model.eval()
    meta = {
        "epoch": int(ckpt["epoch"]) + 1,             # saved epoch is 0-indexed
        "logged_val_fde": ckpt.get("val_fde"),
        "params": sum(p.numel() for p in model.parameters()),
    }
    return model, meta


@torch.no_grad()
def evaluate(model, loader, std, device, K, steps, max_batches, seed):
    torch.manual_seed(seed)
    fde_min, fde_mean, ade_min, ade_mean = [], [], [], []
    hor_min, hor_mean = [], []
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        obs, fut, trel = b["obs"].to(device), b["fut"].to(device), b["t_rel"].to(device)
        preds = euler_sample(model, obs, trel, n_samples=K, n_steps=steps, device=str(device))
        err = torch.norm(preds[..., :2] * std - (fut[..., :2] * std)[None], dim=-1)  # (K,B,43)
        fde, ade = err[:, :, -1], err.mean(-1)                                       # (K,B)
        fde_min.append(fde.min(0).values.cpu());  fde_mean.append(fde.mean(0).cpu())
        ade_min.append(ade.min(0).values.cpu());  ade_mean.append(ade.mean(0).cpu())
        hor_min.append(err.min(0).values.cpu());  hor_mean.append(err.mean(0).cpu())
    cat = lambda xs: torch.cat(xs).numpy()
    return dict(minFDE=cat(fde_min), meanFDE=cat(fde_mean), minADE=cat(ade_min),
                meanADE=cat(ade_mean), hor_min=cat(hor_min), hor_mean=cat(hor_mean))


@torch.no_grad()
def baselines(loader, std, device, max_batches, k=5):
    """Hold-last-position and constant-velocity (from last k obs points) FDE/ADE."""
    out = {"hold": [], "cv": []}
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        obs, fut, trel = b["obs"].to(device), b["fut"].to(device), b["t_rel"].to(device)
        p = obs[..., :2] * std
        g = fut[..., :2] * std
        t_obs, t_fut = trel[:, :43], trel[:, 43:]
        last = p[:, -1]
        hold = last[:, None].expand_as(g)
        dt = (t_obs[:, -1] - t_obs[:, -k]).clamp(min=1e-6)
        v = (p[:, -1] - p[:, -k]) / dt[:, None]
        cv = last[:, None] + v[:, None] * (t_fut - t_obs[:, -1:])[..., None]
        for name, pr in (("hold", hold), ("cv", cv)):
            e = torch.norm(pr - g, dim=-1)
            out[name].append(torch.stack([e[:, -1], e.mean(-1)], 1).cpu())
    return {n: torch.cat(v).numpy() for n, v in out.items()}   # (N,2): FDE, ADE


def paired_boot(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc_path", required=True)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--max_batches", type=int, default=0, help="0 = full split")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--ckpts", default="best,last")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ablation_results")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.mkdir(exist_ok=True)

    ds = nc.Dataset(a.nc_path, "r")
    feat_std = np.array(ds.feature_std)
    ds.close()
    std = torch.tensor(feat_std[:2], device=device, dtype=torch.float32)
    print("feature_std:", feat_std.tolist())

    tr, va, te = get_dataloaders(a.nc_path, batch_size=a.batch_size, subset=a.subset)
    loaders = {"val": va, "test": te}
    for n, l in (("train", tr), ("val", va), ("test", te)):
        try:
            print(f"{n}: {len(l.dataset)} samples")
        except Exception:
            print(f"{n}: {len(l)} batches")
    b = next(iter(va))
    print("t_rel stats (val batch): mean %.3f std %.3f min %.3f max %.3f" % (
        b["t_rel"].mean(), b["t_rel"].std(), b["t_rel"].min(), b["t_rel"].max()))

    summary, store = {}, {}
    for split in a.splits.split(","):
        summary[split] = {"baselines": {}}
        bl = baselines(loaders[split], std, device, a.max_batches)
        for n, arr in bl.items():
            summary[split]["baselines"][n] = {"FDE": float(arr[:, 0].mean()), "ADE": float(arr[:, 1].mean())}
        for name, (mod, files) in RUNS.items():
            for which in a.ckpts.split(","):
                path = Path(files[which])
                if not path.exists():
                    print("missing", path); continue
                model, meta = load_model(mod, path, device)
                r = evaluate(model, loaders[split], std, device, a.K, a.steps, a.max_batches, a.seed)
                key = f"{name}/{which}"
                store[(split, key)] = r
                f = r["minFDE"]
                summary[split][key] = {
                    **meta,
                    "minFDE": float(f.mean()), "minFDE_median": float(np.median(f)),
                    "minFDE_p90": float(np.percentile(f, 90)),
                    "minADE": float(r["minADE"].mean()),
                    "meanFDE": float(r["meanFDE"].mean()), "meanADE": float(r["meanADE"].mean()),
                    "n": int(len(f)),
                }
                np.savez(out / f"{split}_{name}_{which}.npz", **r)
                print(split, key, summary[split][key])

        # paired comparisons on per-trajectory minFDE (best checkpoints)
        names = [n for n in RUNS if (split, f"{n}/best") in store]
        summary[split]["paired_minFDE_diff"] = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                x = store[(split, f"{names[i]}/best")]["minFDE"]
                y = store[(split, f"{names[j]}/best")]["minFDE"]
                m, lo, hi = paired_boot(x, y)
                summary[split]["paired_minFDE_diff"][f"{names[i]} - {names[j]}"] = {"mean": m, "ci95": [lo, hi]}

    json.dump(summary, open(out / "summary.json", "w"), indent=2)

    for split, s in summary.items():
        print(f"\n### {split}\n| run | epoch | minFDE | median | p90 | minADE | meanFDE | meanADE |\n|---|---|---|---|---|---|---|---|")
        for k, v in s.items():
            if "/" in k:
                print(f"| {k} | {v['epoch']} | {v['minFDE']:.1f} | {v['minFDE_median']:.1f} | {v['minFDE_p90']:.1f} | "
                      f"{v['minADE']:.1f} | {v['meanFDE']:.1f} | {v['meanADE']:.1f} |")
        print("baselines:", s["baselines"])
        print("paired diffs (negative = first is better):", s.get("paired_minFDE_diff"))


if __name__ == "__main__":
    main()