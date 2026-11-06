"""
plot_predictions.py - for the SAME flight, show what each model thinks the future looks like.

For every chosen trajectory it draws one figure with a panel per model plus an overlay panel:
  * grey line   = observed history (43 pings)
  * black line  = ground-truth future, star = true final position
  * thin lines  = some of the K sampled futures
  * colour cloud = kernel-density estimate of where the samples END UP (the predictive distribution)
  * contours    = smallest regions holding 50% (solid) and 90% (dashed) of the probability
  * overlay panel puts every model's 50% / 90% regions on one plot so you can compare them directly

All models see identical inputs and start from identical noise (same seed), so differences
come from the model, not from luck.

Put it in your training folder next to eval_ablation.py and run from the project root, e.g.

    python training\\plot_predictions.py --nc_path C:\\dev\\work\\AviationDiffusionModelsKishore\\trajectories_adsblol_seq86_stage2.nc --subset 100000 --pick disagree --n_plots 6

Needs matplotlib:  pip install matplotlib
"""
import argparse, importlib, sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent.parent))

# Models that are missing on disk are skipped automatically, so Time2Vec shows up once it is trained.
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
    "Time2Vec": ("models.dit_time2vec", {
        "best": "checkpoints_cfm_time2vec/best.pt",
        "last": "checkpoints_cfm_time2vec/last.pt"}),
}
COLORS = {"Sinusoidal": "#2B59C3", "RoPE-OG": "#D9702A", "RoPE-TS": "#2E9C93", "Time2Vec": "#7A4FB5"}
INK, MUTED, GRID = "#14213D", "#5E6878", "#DDD8CB"

torch = None
euler_sample = None


def _lazy_imports():
    global torch, euler_sample
    import torch as _t
    from models.cfm import euler_sample as _e
    torch, euler_sample = _t, _e


# ----------------------------------------------------------------------------- model / data helpers
def load_model(module_name, ckpt_path, device):
    mod = importlib.import_module(module_name)
    model = mod.TrajectoryDiT(d_model=256, n_heads=8, n_layers=6).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["ema_state"])          # EMA weights, same as the eval script
    model.eval()
    return model, int(ckpt["epoch"]) + 1


def get_batch(ds, idxs, device):
    items = [ds[int(i)] for i in idxs]
    return {k: torch.stack([torch.as_tensor(it[k]) for it in items]).float().to(device)
            for k in ("obs", "fut", "t_rel")}


def predict(model, batch, K, steps, seed, std, device):
    """-> (K, B, 43, 2) predicted future x,y in metres."""
    torch.manual_seed(seed)                            # identical starting noise for every model
    with torch.no_grad():
        p = euler_sample(model, batch["obs"], batch["t_rel"], n_samples=K, n_steps=steps, device=str(device))
    return (p[..., :2] * std).cpu().numpy()


# ----------------------------------------------------------------------------- choosing flights
def pick_turning(ds, pool_idx, n, std, device):
    """Flights whose heading changes most between the end of the history and the end of the future."""
    scores = []
    for s in range(0, len(pool_idx), 256):
        b = get_batch(ds, pool_idx[s:s + 256], device)
        o = (b["obs"][..., :2] * std).cpu().numpy()
        f = (b["fut"][..., :2] * std).cpu().numpy()
        vo, vf = o[:, -1] - o[:, -10], f[:, -1] - f[:, -10]
        ang = np.degrees(np.arctan2(vf[:, 1], vf[:, 0]) - np.arctan2(vo[:, 1], vo[:, 0]))
        ang = np.abs((ang + 180) % 360 - 180)
        moving = (np.linalg.norm(vo, axis=1) > 500) & (np.linalg.norm(vf, axis=1) > 500)
        scores.append(np.where(moving, ang, 0.0))
    scores = np.concatenate(scores)
    return [int(pool_idx[i]) for i in np.argsort(-scores)[:n]]


def pick_disagree(models, ds, pool_idx, n, std, device, steps, seed):
    """Flights where the best and worst model differ most in mean final error."""
    scores = []
    for s in range(0, len(pool_idx), 64):
        b = get_batch(ds, pool_idx[s:s + 64], device)
        gt_end = (b["fut"][:, -1, :2] * std).cpu().numpy()
        per_model = []
        for m in models.values():
            p = predict(m, b, 8, steps, seed, std, device)
            per_model.append(np.linalg.norm(p[:, :, -1] - gt_end[None], axis=-1).mean(0))
        per_model = np.stack(per_model)
        scores.append(per_model.max(0) - per_model.min(0))
    scores = np.concatenate(scores)
    return [int(pool_idx[i]) for i in np.argsort(-scores)[:n]]


# ----------------------------------------------------------------------------- density maths (numpy only)
def kde_eval(pts, qx, qy, h):
    """Gaussian KDE (unnormalised, peak-comparable) of pts (n,2) evaluated at query points qx, qy (any shape)."""
    q = np.stack([qx.ravel(), qy.ravel()], 1).astype(np.float32)
    out = np.zeros(len(q), dtype=np.float32)
    for s in range(0, len(pts), 100):
        c = pts[s:s + 100].astype(np.float32)
        d2 = ((q[:, None, :] - c[None]) ** 2).sum(-1)
        out += np.exp(-d2 / (2 * h * h)).sum(1)
    return out.reshape(qx.shape)


def bandwidth(pts, span, mode):
    if mode == "path":
        return max(0.03, 0.025 * span)
    sd = np.sqrt(pts.var(0).mean())
    return max(0.03, sd * len(pts) ** (-1 / 6))        # Scott's rule in 2-D, floor of 30 m


def hdr_level(Z, mass):
    """Density threshold such that {Z >= level} holds `mass` of the total probability."""
    flat = np.sort(Z.ravel())[::-1]
    c = np.cumsum(flat)
    return float(flat[min(np.searchsorted(c, mass * c[-1]), len(flat) - 1)])


# ----------------------------------------------------------------------------- plotting
def style_axis(ax, lim=None):
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_flight(idx, split, obs, fut, samples, K, args, out_path, ckpt_epochs):
    """obs (43,2), fut (43,2), samples {name: (K,43,2)}; all in km relative to the last observed ping."""
    names = list(samples)
    # shared square limits so every panel is directly comparable
    obs_view = obs[-1:] if args.zoom == "future" else obs      # "future": frame the prediction, not the history
    allp = [obs_view, fut] + [s[:, ::4].reshape(-1, 2) for s in samples.values()] + [s[:, -1] for s in samples.values()]
    exact = (obs_view, fut)
    lo = np.min([a.min(0) if any(a is e for e in exact) else np.percentile(a, 0.5, axis=0) for a in allp], axis=0)
    hi = np.max([a.max(0) if any(a is e for e in exact) else np.percentile(a, 99.5, axis=0) for a in allp], axis=0)
    c, half = (lo + hi) / 2, max((hi - lo).max() / 2 * 1.12, 0.5)
    xs, ys = np.linspace(c[0] - half, c[0] + half, 170), np.linspace(c[1] - half, c[1] + half, 170)
    gx, gy = np.meshgrid(xs, ys)

    n_panels = len(names) + 1
    if args.layout == "grid":
        ncols = 2
        nrows = int(np.ceil(n_panels / 2))
        figsize = (6.2 * ncols, 6.4 * nrows)
    else:
        ncols, nrows, figsize = n_panels, 1, (5.6 * n_panels, 6.2)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True, facecolor=args.bg)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n_panels:]:
        ax.axis("off")

    fields = {}
    for ax, name in zip(axes, names):
        col = COLORS.get(name, "#444444")
        S = samples[name]                                              # (K,43,2)
        pts = S[:, -1] if args.density == "end" else S[:, ::4].reshape(-1, 2)
        h = bandwidth(pts, 2 * half, args.density)
        Z = kde_eval(pts, gx, gy, h)
        l50, l90 = hdr_level(Z, 0.5), hdr_level(Z, 0.9)
        fields[name] = (Z, l50, l90)

        r, g, b = to_rgb(col)
        cmap = LinearSegmentedColormap.from_list(name, [(r, g, b, 0.0), (r, g, b, 0.9)])
        style_axis(ax, (half, half))
        ax.imshow((Z / Z.max()) ** 0.6, extent=(xs[0], xs[-1], ys[0], ys[-1]), origin="lower",
                  cmap=cmap, vmin=0, vmax=1, zorder=1, interpolation="bilinear")
        ax.contour(xs, ys, Z, levels=[l90], colors=[col], linestyles="--", linewidths=1.2, zorder=3)
        ax.contour(xs, ys, Z, levels=[l50], colors=[col], linestyles="-", linewidths=1.8, zorder=3)
        for k in range(min(args.paths, K)):
            ax.plot(S[k, :, 0], S[k, :, 1], color=col, alpha=0.22, lw=0.8, zorder=2)
        ax.plot(S[:, -1, 0], S[:, -1, 1], ".", color=col, ms=2.5, alpha=0.5, zorder=3)
        ax.plot(obs[:, 0], obs[:, 1], color="#6B7280", lw=2.2, zorder=4)
        ax.plot(*obs[-1], "o", color="#6B7280", ms=6, zorder=5)
        ax.plot(fut[:, 0], fut[:, 1], color="black", lw=2.0, zorder=5)
        ax.plot(*fut[-1], "*", color="black", ms=15, mec="white", mew=0.8, zorder=6)

        ends = S[:, -1]
        fde = np.linalg.norm(ends - fut[-1], axis=1)
        spread = np.sqrt(ends.var(0).sum())
        inside = kde_eval(pts, np.array([fut[-1, 0]]), np.array([fut[-1, 1]]), h)[0] >= l90
        ax.set_title(f"{name}  (epoch {ckpt_epochs[name]})", color=col, fontsize=14, fontweight="bold", loc="left")
        ax.text(0.02, 0.02,
                f"best-of-{K} FDE  {fde.min():.2f} km\nmean FDE  {fde.mean():.2f} km\n"
                f"spread  {spread:.2f} km\ntruth in 90% region: {'yes' if inside else 'NO'}",
                transform=ax.transAxes, fontsize=9.5, color=INK, va="bottom", zorder=10,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID, alpha=0.92))

    # overlay panel
    ax = axes[len(names)]
    style_axis(ax, (half, half))
    for name in names:
        Z, l50, l90 = fields[name]
        col = COLORS.get(name, "#444444")
        ax.contourf(xs, ys, Z, levels=[l50, Z.max() * 1.01], colors=[col], alpha=0.18, zorder=1)
        ax.contour(xs, ys, Z, levels=[l90], colors=[col], linestyles="--", linewidths=1.6, zorder=3)
        ax.contour(xs, ys, Z, levels=[l50], colors=[col], linestyles="-", linewidths=2.4, zorder=3)
    ax.plot(obs[:, 0], obs[:, 1], color="#6B7280", lw=2.2, zorder=4)
    ax.plot(*obs[-1], "o", color="#6B7280", ms=6, zorder=5)
    ax.plot(fut[:, 0], fut[:, 1], color="black", lw=2.0, zorder=5)
    ax.plot(*fut[-1], "*", color="black", ms=15, mec="white", mew=0.8, zorder=6)
    ax.set_title("All models, where the probability sits", color=INK, fontsize=14, fontweight="bold", loc="left")
    handles = [Line2D([], [], color=COLORS.get(n, "#444"), lw=3, label=n) for n in names]
    handles += [Line2D([], [], color=MUTED, lw=2.4, label="50% region"),
                Line2D([], [], color=MUTED, lw=1.6, ls="--", label="90% region"),
                Line2D([], [], color="#6B7280", lw=2.2, label="observed history"),
                Line2D([], [], color="black", lw=2, marker="*", ms=11, label="true future")]
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.92, edgecolor=GRID)

    for ax in axes[:n_panels]:          # identical limits on every panel (stray samples must not rescale one)
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(ys[0], ys[-1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x from last observed ping (km)", color=MUTED, fontsize=10)
        ax.set_ylabel("y from last observed ping (km)", color=MUTED, fontsize=10)
    fig.suptitle(f"{split} flight #{idx}   |   {K} samples per model, same inputs, same noise seed",
                 color=INK, fontsize=15, x=0.01, ha="left")
    fig.savefig(out_path, dpi=args.dpi, facecolor=args.bg)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc_path", required=True)
    ap.add_argument("--subset", type=int, default=None, help="use the SAME value as in training")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--indices", default="", help="comma-separated dataset indices, e.g. 12,845,3011")
    ap.add_argument("--pick", default="disagree", choices=["disagree", "turning", "random"],
                    help="how to choose flights when --indices is not given")
    ap.add_argument("--n_plots", type=int, default=6)
    ap.add_argument("--pool", type=int, default=1000, help="how many flights to scan when picking")
    ap.add_argument("--K", type=int, default=100, help="samples per model")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--ckpt", default="last", choices=["best", "last"])
    ap.add_argument("--models", default="", help="comma list, default = every model whose checkpoint exists")
    ap.add_argument("--density", default="end", choices=["end", "path"],
                    help="end = distribution of final positions; path = of all positions along the future")
    ap.add_argument("--paths", type=int, default=30, help="how many sample paths to draw per panel")
    ap.add_argument("--layout", default="row", choices=["row", "grid"])
    ap.add_argument("--zoom", default="all", choices=["all", "future"],
                    help="all = frame history + future; future = zoom in on the predicted region")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--bg", default="#F7F5F0", help="figure background colour (use white for print)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out", default="flight_plots")
    a = ap.parse_args()

    _lazy_imports()
    import netCDF4 as nc
    from DataLoaders.ADSBdataset import get_dataloaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    d = nc.Dataset(a.nc_path, "r")
    std = torch.tensor(np.array(d.feature_std)[:2], device=device, dtype=torch.float32)
    d.close()

    _, va, te = get_dataloaders(a.nc_path, batch_size=a.batch_size, subset=a.subset)
    ds = (va if a.split == "val" else te).dataset
    print(f"{a.split}: {len(ds)} flights")

    wanted = [m for m in a.models.split(",") if m] or list(RUNS)
    models, epochs = {}, {}
    for name in wanted:
        mod, files = RUNS[name]
        path = Path(files[a.ckpt])
        if not path.exists():
            print("skipping", name, "- missing", path)
            continue
        models[name], epochs[name] = load_model(mod, path, device)
        print("loaded", name, "epoch", epochs[name])
    if not models:
        sys.exit("no checkpoints found - run from the project root")

    rng = np.random.default_rng(a.seed)
    if a.indices:
        chosen = [int(i) for i in a.indices.split(",")]
    else:
        pool = np.sort(rng.choice(len(ds), min(a.pool, len(ds)), replace=False))
        if a.pick == "random":
            chosen = [int(i) for i in rng.choice(pool, min(a.n_plots, len(pool)), replace=False)]
        elif a.pick == "turning":
            chosen = pick_turning(ds, pool, a.n_plots, std, device)
        else:
            if len(models) < 2:
                sys.exit("--pick disagree needs at least two models; use --pick turning or --indices")
            chosen = pick_disagree(models, ds, pool, a.n_plots, std, device, a.steps, a.seed)
    print("flights:", chosen, "  (reuse with --indices", ",".join(map(str, chosen)) + ")")

    batch = get_batch(ds, chosen, device)
    obs = (batch["obs"][..., :2] * std).cpu().numpy()
    fut = (batch["fut"][..., :2] * std).cpu().numpy()
    preds = {n: predict(m, batch, a.K, a.steps, a.seed, std, device) for n, m in models.items()}   # (K,B,43,2)

    for j, idx in enumerate(chosen):
        anchor = obs[j, -1]
        o, f = (obs[j] - anchor) / 1000, (fut[j] - anchor) / 1000          # km, relative to last ping
        s = {n: (p[:, j] - anchor) / 1000 for n, p in preds.items()}
        path = out / f"{a.split}_flight_{idx}.png"
        plot_flight(idx, a.split, o, f, s, a.K, a, path, epochs)
        print("saved", path)


if __name__ == "__main__":
    main()