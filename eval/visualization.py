import torch
import numpy as np
import folium
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from DataLoaders.ADSBdataset import get_dataloaders
from models.dit import TrajectoryDiT
from models.ddim import make_cosine_schedule


def xy_to_latlon(x, y):
    R       = 6_371_000.0
    lat_ref = np.radians(37.6213)
    lon_ref = np.radians(-122.3790)
    lat = np.degrees(y / R + lat_ref)
    lon = np.degrees(x / (R * np.cos(lat_ref)) + lon_ref)
    return lat, lon


def ddim_sample(model, obs, t_rel, alphas_cumprod, n_samples=20, n_steps=20, device="cuda"):
    B  = obs.shape[0]
    T  = len(alphas_cumprod)

    obs_rep  = obs.unsqueeze(0).expand(n_samples, -1, -1, -1).reshape(n_samples * B, 43, 6)
    trel_rep = t_rel.unsqueeze(0).expand(n_samples, -1, -1).reshape(n_samples * B, 86)

    x = torch.randn(n_samples * B, 43, 6, device=device)

    step_size = T // n_steps
    timesteps = list(range(0, T, step_size))[::-1]

    model.eval()
    with torch.no_grad():
        for i, t_val in enumerate(timesteps):
            t_tensor   = torch.full((n_samples * B,), t_val, device=device, dtype=torch.long)
            noise_pred = model(obs_rep, x, t_tensor.float() / T, trel_rep)
            a_bar      = alphas_cumprod[t_val]
            a_bar_prev = alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0)
            x0_pred    = (x - torch.sqrt(1 - a_bar) * noise_pred) / torch.sqrt(a_bar)
            x          = torch.sqrt(a_bar_prev) * x0_pred + torch.sqrt(1 - a_bar_prev) * noise_pred

    return x.reshape(n_samples, B, 43, 6)


def denormalize(traj_norm, feat_mean, feat_std):
    return traj_norm * feat_std + feat_mean


def make_map(obs_raw, fut_raw, preds_raw, map_idx, n_show=8, obs_tail=12):
    B = min(n_show, obs_raw.shape[0])

    all_x = fut_raw[:B, :, 0].mean()
    all_y = fut_raw[:B, :, 1].mean()
    center_lat, center_lon = xy_to_latlon(all_x, all_y)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="CartoDB"
    )

    for i in range(B):
        # Observed context (last obs_tail steps)
        obs_xy = obs_raw[i, -obs_tail:, :2]
        obs_ll = [xy_to_latlon(x, y) for x, y in obs_xy]
        folium.PolyLine(
            obs_ll, color="#2166ac", weight=2, opacity=0.7,
            tooltip=f"obs #{i}"
        ).add_to(m)

        # Ground truth future
        last_obs = xy_to_latlon(obs_raw[i, -1, 0], obs_raw[i, -1, 1])
        fut_xy   = fut_raw[i, :, :2]
        fut_ll   = [last_obs] + [xy_to_latlon(x, y) for x, y in fut_xy]
        folium.PolyLine(
            fut_ll, color="#1a9641", weight=2.5, opacity=0.8,
            tooltip=f"gt #{i}"
        ).add_to(m)

        # Predicted fan
        K = preds_raw.shape[0]
        for k in range(K):
            pred_xy = preds_raw[k, i, :, :2]
            pred_ll = [last_obs] + [xy_to_latlon(x, y) for x, y in pred_xy]
            folium.PolyLine(
                pred_ll, color="#d7191c", weight=1, opacity=0.12
            ).add_to(m)

    legend = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: white; padding: 12px 16px; border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.2); font-family: sans-serif; font-size: 13px;">
        <b>Trajectory Prediction — Sample {idx}</b><br>
        <span style="color:#2166ac">■</span> Observation (last 12 steps)<br>
        <span style="color:#1a9641">■</span> Ground truth future<br>
        <span style="color:#d7191c">■</span> Model predictions (K=20)<br>
        N={N} aircraft
    </div>
    """.format(idx=map_idx + 1, N=B)
    m.get_root().html.add_child(folium.Element(legend))

    return m


def visualize_all(
    nc_path,
    ckpt_path,
    output_dir  = "predictions",
    n_maps      = 6,
    n_show      = 8,
    n_samples   = 20,
    d_model     = 256,
    n_layers    = 6,
    n_heads     = 8,
    device      = "cuda",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    Path(output_dir).mkdir(exist_ok=True)

    # ── Load stats ────────────────────────────────────────────────────────
    import netCDF4 as nc
    ds        = nc.Dataset(nc_path, "r")
    feat_mean = np.array(ds.feature_mean, dtype=np.float32)
    feat_std  = np.array(ds.feature_std,  dtype=np.float32)
    ds.close()

    fm = torch.tensor(feat_mean, device=device)
    fs = torch.tensor(feat_std,  device=device)

    # ── Load data ONCE ────────────────────────────────────────────────────
    print("Loading data...")
    _, val_loader, _ = get_dataloaders(nc_path, batch_size=n_show, subset=10_000)
    loader_iter = iter(val_loader)

    # ── Load model ONCE ───────────────────────────────────────────────────
    print("Loading model...")
    model = TrajectoryDiT(d_model=d_model, n_heads=n_heads, n_layers=n_layers).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("ema_state", ckpt.get("model_state"))
    model.load_state_dict(state)
    model.eval()

    # ── Noise schedule ONCE ───────────────────────────────────────────────
    _, _, alphas_cumprod = make_cosine_schedule(T=1000)
    alphas_cumprod = alphas_cumprod.to(device)

    # ── Generate n_maps maps ──────────────────────────────────────────────
    for map_idx in range(n_maps):
        print(f"Generating map {map_idx + 1}/{n_maps}...")
        batch = next(loader_iter)

        obs   = batch["obs"].to(device)
        fut   = batch["fut"].to(device)
        t_rel = batch["t_rel"].to(device)

        # Sample predictions
        preds_norm = ddim_sample(model, obs, t_rel, alphas_cumprod,
                                  n_samples=n_samples, device=str(device))

        # Denormalize
        obs_raw   = denormalize(obs,        fm, fs).cpu().numpy()
        fut_raw   = denormalize(fut,        fm, fs).cpu().numpy()
        preds_raw = denormalize(preds_norm, fm, fs).cpu().numpy()

        # Build and save map
        m = make_map(obs_raw, fut_raw, preds_raw, map_idx, n_show=n_show)
        out_path = f"{output_dir}/predictions_{map_idx + 1}.html"
        m.save(out_path)
        print(f"  Saved → {out_path}")

    print(f"\nDone! Open any predictions/predictions_N.html in your browser.")


if __name__ == "__main__":
    visualize_all(
        nc_path    = r"D:\trajectories_adsblol_seq86_stage2.nc",
        ckpt_path  = r"checkpoints\best.pt",
        output_dir = "predictions_ep100_100k",
        n_maps     = 6,
        n_show     = 8,
        n_samples  = 20,
        d_model    = 256,
        n_layers   = 6,
    )