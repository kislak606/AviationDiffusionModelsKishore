import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from copy import deepcopy

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from DataLoaders.ADSBdataset import get_dataloaders
from models.dit_RoPE_original import TrajectoryDiT
from models.cfm import sample_flow_time, forward_cfm, euler_sample


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay  = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()

    def update(self, model):
        with torch.no_grad():
            for s_param, param in zip(self.shadow.parameters(), model.parameters()):
                s_param.data = self.decay * s_param.data + (1 - self.decay) * param.data

    def __call__(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)


def validate(ema_model, val_loader, device, feat_std, max_batches=20):
    ema_model.shadow.eval()
    total_fde = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if n_batches >= max_batches:
                break

            obs   = batch["obs"].to(device)
            fut   = batch["fut"].to(device)
            t_rel = batch["t_rel"].to(device)

            preds = euler_sample(ema_model.shadow, obs, t_rel,
                                  n_samples=5, n_steps=20, device=str(device))

            feat_std_xy    = torch.tensor(feat_std[:2], device=device)
            preds_xy       = preds[..., :2] * feat_std_xy
            fut_xy         = fut[..., :2]   * feat_std_xy
            fde_per_sample = torch.norm(
                preds_xy[:, :, -1, :] - fut_xy[None, :, -1, :], dim=-1
            )
            min_fde    = fde_per_sample.min(dim=0).values.mean()
            total_fde += min_fde.item()
            n_batches += 1

    return total_fde / n_batches


def train(
    nc_path,
    output_dir   = "checkpoints_cfm_rope_og",
    epochs       = 100,
    batch_size   = 64,
    lr           = 1e-4,
    weight_decay = 0.01,
    grad_clip    = 1.0,
    warmup_steps = 1000,
    d_model      = 256,
    n_heads      = 8,
    n_layers     = 6,
    device       = "cuda",
    subset       = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} — CFM + RoPE OG")

    train_loader, val_loader, test_loader = get_dataloaders(
        nc_path, batch_size=batch_size, subset=subset,
    )

    import netCDF4 as nc
    ds       = nc.Dataset(nc_path, "r")
    feat_std = np.array(ds.feature_std)
    ds.close()

    model = TrajectoryDiT(d_model=d_model, n_heads=n_heads, n_layers=n_layers).to(device)
    ema   = EMA(model, decay=0.9999)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Resume from checkpoint ────────────────────────────────────────────
    start_epoch = 0
    best_fde    = float("inf")
    global_step = 0
    last_ckpt   = output_dir / "last.pt"
    best_ckpt   = output_dir / "best.pt"

    if last_ckpt.exists():
        print(f"Resuming from {last_ckpt}...")
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        ema.shadow.load_state_dict(ckpt["ema_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])   # restores the annealed LR
        start_epoch = ckpt["epoch"] + 1

        # Continue the cosine schedule instead of restarting it
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        else:
            # older checkpoints didn't save the scheduler
            scheduler.last_epoch = start_epoch

        # Skip the warmup, it already happened
        global_step = start_epoch * len(train_loader)

        # Don't let a worse checkpoint overwrite best.pt
        if best_ckpt.exists():
            best_val = torch.load(best_ckpt, map_location="cpu").get("val_fde")
            if best_val is not None:
                best_fde = best_val

        print(f"Resumed at