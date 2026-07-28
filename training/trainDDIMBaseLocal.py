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
from models.dit import TrajectoryDiT
from models.ddim import make_cosine_schedule, forward_diffusion


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


def validate(ema_model, val_loader, alphas_cumprod, device, feat_std, max_batches=20):
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

            x = torch.randn_like(fut)
            T = len(alphas_cumprod)
            step_size = T // 20
            timesteps = list(range(0, T, step_size))[::-1]

            for i, t_val in enumerate(timesteps):
                t_tensor   = torch.full((obs.shape[0],), t_val, device=device, dtype=torch.long)
                noise_pred = ema_model(obs, x, t_tensor.float() / T, t_rel)
                a_bar      = alphas_cumprod[t_val]
                a_bar_prev = alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0)
                x0_pred    = (x - torch.sqrt(1 - a_bar) * noise_pred) / torch.sqrt(a_bar)
                x          = torch.sqrt(a_bar_prev) * x0_pred + torch.sqrt(1 - a_bar_prev) * noise_pred

            feat_std_xy = torch.tensor(feat_std[:2], device=device)
            pred_xy     = x[:, -1, :2] * feat_std_xy
            true_xy     = fut[:, -1, :2] * feat_std_xy
            fde         = torch.norm(pred_xy - true_xy, dim=-1).mean()
            total_fde  += fde.item()
            n_batches  += 1

    return total_fde / n_batches


def train(
    nc_path,
    output_dir   = "checkpoints",
    T            = 1000,
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
    print(f"Training on {device}")

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

    betas, alphas, alphas_cumprod = make_cosine_schedule(T=T)
    alphas_cumprod = alphas_cumprod.to(device)

    # FIX: optimizer created BEFORE resume block so it exists when we load state
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Resume from checkpoint ────────────────────────────────────────────
    start_epoch = 0
    last_ckpt   = output_dir / "last.pt"
    if last_ckpt.exists():
        print(f"Resuming from {last_ckpt}...")
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        ema.shadow.load_state_dict(ckpt["ema_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed at epoch {start_epoch}")
    # ─────────────────────────────────────────────────────────────────────

    best_fde    = float("inf")
    global_step = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            obs   = batch["obs"].to(device)
            fut   = batch["fut"].to(device)
            t_rel = batch["t_rel"].to(device)

            t          = torch.randint(0, T, (obs.shape[0],), device=device)
            x_t, noise = forward_diffusion(fut, t, alphas_cumprod)
            noise_pred = model(obs, x_t, t.float() / T, t_rel)
            loss       = nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if global_step < warmup_steps:
                lr_scale = (global_step + 1) / warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = lr * lr_scale

            ema.update(model)
            total_loss  += loss.item()
            n_batches   += 1
            global_step += 1

        avg_loss = total_loss / n_batches
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            val_fde = validate(ema, val_loader, alphas_cumprod, device, feat_std, max_batches=20)
            print(f"Epoch {epoch+1:03d} | loss {avg_loss:.4f} | val FDE {val_fde:.1f}m")
            checkpoint = {
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "ema_state":       ema.shadow.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_fde":         val_fde,
            }
            torch.save(checkpoint, output_dir / "last.pt")
            if val_fde < best_fde:
                best_fde = val_fde
                torch.save(checkpoint, output_dir / "best.pt")
                print(f"  ✓ best model saved (FDE {best_fde:.1f}m)")
        else:
            print(f"Epoch {epoch+1:03d} | loss {avg_loss:.4f}")
            checkpoint = {
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "ema_state":       ema.shadow.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_fde":         None,
            }
            torch.save(checkpoint, output_dir / "last.pt")


if __name__ == "__main__":
    train(
        nc_path    = r"D:\trajectories_adsblol_seq86_stage2.nc",
        output_dir = "checkpoints",
        epochs     = 100,
        batch_size = 64,
        d_model    = 256,
        n_layers   = 6,
        subset     = 100_000,
    )