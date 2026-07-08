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


# ── EMA (Exponential Moving Average) ─────────────────────────────────────────
# Keeps a shadow copy of model weights that updates slowly
# Used for validation/inference — more stable than the live model
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay      = decay
        self.shadow     = deepcopy(model)   # copy of model weights
        self.shadow.eval()

    def update(self, model):
        # For each parameter: shadow = decay * shadow + (1 - decay) * live
        with torch.no_grad():
            for s_param, param in zip(self.shadow.parameters(), model.parameters()):
                s_param.data = self.decay * s_param.data + (1 - self.decay) * param.data
                # hint: decay * shadow + (1 - decay) * live

    def __call__(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)


# ── Validation step ───────────────────────────────────────────────────────────
def validate(ema_model, val_loader, alphas_cumprod, device, feat_std):
    ema_model.shadow.eval()
    total_fde = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            obs   = batch["obs"].to(device)     # (B, 43, 6)
            fut   = batch["fut"].to(device)     # (B, 43, 6)
            t_rel = batch["t_rel"].to(device)   # (B, 86)

            # Run DDIM sampling with EMA model (20 steps)
            x = torch.randn_like(fut)           # start from pure noise
            T = len(alphas_cumprod)
            step_size = T // 20                 # 20 inference steps

            timesteps = list(range(0, T, step_size))[::-1]   # T→0

            for i, t_val in enumerate(timesteps):
                t_tensor = torch.full((obs.shape[0],), t_val,
                                       device=device, dtype=torch.long)

                # Predict noise
                noise_pred = ema_model(obs, x, t_tensor.float() / T, t_rel)

                # DDIM step
                a_bar      = alphas_cumprod[t_val]
                a_bar_prev = alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0)

                x0_pred = (x - torch.sqrt(1 - a_bar) * noise_pred) / torch.sqrt(a_bar)
                x       = torch.sqrt(a_bar_prev) * x0_pred + torch.sqrt(1 - a_bar_prev) * noise_pred

            # Unnormalize xy only for FDE computation
            feat_std_xy = torch.tensor(feat_std[:2], device=device)   # (2,)
            pred_xy     = x[:, -1, :2]   * feat_std_xy                # last timestep xy
            true_xy     = fut[:, -1, :2] * feat_std_xy

            fde         = torch.norm(pred_xy - true_xy, dim=-1).mean()
            total_fde  += fde.item()
            n_batches  += 1

    return total_fde / n_batches   # mean FDE in metres


# ── Main training loop ────────────────────────────────────────────────────────
def train(
    nc_path,
    output_dir    = "checkpoints",
    T             = 1000,
    epochs        = 100,
    batch_size    = 64,
    lr            = 1e-4,
    weight_decay  = 0.01,
    grad_clip     = 1.0,
    warmup_steps  = 1000,
    d_model       = 256,
    n_heads       = 8,
    n_layers      = 6,
    device        = "cuda",
    subset        = None,    # set to e.g. 100_000 to train on subset
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        nc_path,
        batch_size=batch_size,
        subset=subset,
    )
    # grab feat_std for unnormalizing during validation
    import netCDF4 as nc
    ds       = nc.Dataset(nc_path, "r")
    feat_std = np.array(ds.feature_std)
    ds.close()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TrajectoryDiT(d_model=d_model, n_heads=n_heads, n_layers=n_layers).to(device)
    ema   = EMA(model, decay=0.9999)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Noise schedule ────────────────────────────────────────────────────────
    betas, alphas, alphas_cumprod = make_cosine_schedule(T=T)
    alphas_cumprod = alphas_cumprod.to(device)

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_fde   = float("inf")
    global_step = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            obs   = batch["obs"].to(device)     # (B, 43, 6)
            fut   = batch["fut"].to(device)     # (B, 43, 6)
            t_rel = batch["t_rel"].to(device)   # (B, 86)

            # Step 1: sample random timestep for each sample in batch
            t = torch.randint(0, T, (obs.shape[0],), device=device)

            # Step 2: forward diffusion — add noise to future trajectory
            x_t, noise = forward_diffusion(fut, t, alphas_cumprod)
            # hint: (fut, t, alphas_cumprod)

            # Step 3: predict noise with model
            noise_pred = model(obs, x_t, t.float() / T, t_rel)
            # hint: (obs, x_t, t.float() / T, t_rel)

            # Step 4: MSE loss between predicted and actual noise
            loss = nn.functional.mse_loss(noise_pred, noise)
            # hint: (noise_pred, noise)

            # Step 5: backprop
            optimizer.zero_grad()
            loss.backward()        # hint: loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()        # hint: optimizer.step()

            # Step 6: linear warmup for learning rate
            if global_step < warmup_steps:
                lr_scale = (global_step + 1) / warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = lr * lr_scale

            # Step 7: update EMA
            ema.update(model)   # hint: model

            total_loss  += loss.item()
            n_batches   += 1
            global_step += 1

        avg_loss = total_loss / n_batches
        scheduler.step()

        # ── Validation every epoch ─────────────────────────────────────────
        val_fde = validate(ema, val_loader, alphas_cumprod, device, feat_std)

        print(f"Epoch {epoch+1:03d} | loss {avg_loss:.4f} | val FDE {val_fde:.1f}m")

        # ── Save checkpoints ───────────────────────────────────────────────
        checkpoint = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "ema_state":       ema.shadow.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_fde":         val_fde,
        }

        # Always save latest
        torch.save(checkpoint, output_dir / "last.pt")

        # Save best if val FDE improved
        if val_fde < best_fde:
            best_fde = val_fde
            torch.save(checkpoint, output_dir / "best.pt")
            print(f"  ✓ best model saved (FDE {best_fde:.1f}m)")


if __name__ == "__main__":
    train(
        nc_path    = r"D:\trajectories_adsblol_seq86_stage2.nc",
        output_dir = "checkpoints",
        epochs     = 50,
        batch_size = 64,
        d_model    = 256,
        n_layers   = 6,
        subset     = 100_000,   # remove this line to train on full dataset
    )