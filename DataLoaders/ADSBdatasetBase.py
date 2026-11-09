"""
adsbdatasetBase.py

Drop-in replacement for ADSBdataset.py (same get_dataloaders interface), but the
train/val/test split now reproduces TrajectoryForecastDatasetV3 (mae_dataset.py)
exactly:

    flights = np.unique(icao24)
    np.random.default_rng(42).shuffle(flights)      # PCG64, NOT RandomState
    n_test = int(n * 0.05); n_val = int(n * 0.10)
    train = flights[: n - n_val - n_test]
    val   = flights[n - n_val - n_test : n - n_test]
    test  = flights[n - n_test :]

Same .nc file + same split logic => same test flights as the mentor's pipeline.
Do NOT pass subset when you want a comparable split (it changes the flight set).

Usage in a training script:
    from DataLoaders.adsbdatasetBase import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(nc_path, batch_size=64)
    feat_std = train_loader.dataset.std      # same values as ds.feature_std
"""

from __future__ import annotations

import netCDF4 as nc
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

VAL_FRACTION  = 0.10
TEST_FRACTION = 0.05
SPLIT_SEED    = 42


def load_netcdf(filepath):
    with nc.Dataset(filepath, "r") as ds:
        trajectory = np.array(ds.variables["trajectory"][:])                      # (N, T, 6)
        timestamps = np.array(ds.variables["timestamps"][:], dtype=np.float64)    # (N, T)
        icao24     = np.array(ds.variables["icao24"][:])                          # (N,)
        stats = {
            "feature_mean": np.array(ds.feature_mean, dtype=np.float32),
            "feature_std":  np.array(ds.feature_std,  dtype=np.float32),
            "t_rel_mean":   float(ds.t_rel_mean),
            "t_rel_std":    float(ds.t_rel_std),
        }
    return trajectory, timestamps, icao24, stats


def split_flights(icao24, val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION,
                  seed=SPLIT_SEED):
    """Split by aircraft, identical to TrajectoryForecastDatasetV3."""
    flights = np.unique(icao24)
    rng = np.random.default_rng(seed)
    rng.shuffle(flights)
    n        = len(flights)
    n_test_f = int(n * test_fraction)
    n_val_f  = int(n * val_fraction)
    return {
        "train": flights[: n - n_val_f - n_test_f],
        "val":   flights[n - n_val_f - n_test_f : n - n_test_f],
        "test":  flights[n - n_test_f :],
    }


def split_indices(icao24, **kwargs):
    flights = split_flights(icao24, **kwargs)
    return {name: np.where(np.isin(icao24, f))[0] for name, f in flights.items()}


class TrajectoryDataset(Dataset):
    """Absolute z-score obs/fut plus normalised t_rel, like TrajectoryForecastDatasetV3."""

    def __init__(self, trajectory, timestamps, icao24, indices, stats,
                 obs_len=43, fut_len=43):
        seq_len = obs_len + fut_len
        assert trajectory.shape[1] >= seq_len, (
            f"NetCDF has T={trajectory.shape[1]} but obs_len+fut_len={seq_len}"
        )
        self.obs_len = obs_len
        self.fut_len = fut_len
        self.mean    = stats["feature_mean"]
        self.std     = stats["feature_std"]

        raw = trajectory[indices, :seq_len, :]
        ts  = timestamps[indices, :seq_len]

        z = ((raw - self.mean) / (self.std + 1e-8)).astype(np.float32)
        self.obs = torch.from_numpy(z[:, :obs_len, :].copy())
        self.fut = torch.from_numpy(z[:, obs_len:, :].copy())

        t_rel   = (ts - ts[:, 0:1]).astype(np.float32)
        t_rel_n = ((t_rel - stats["t_rel_mean"]) / (stats["t_rel_std"] + 1e-8)).astype(np.float32)
        self.t_rel = torch.from_numpy(t_rel_n.copy())

        self.icao24 = icao24[indices]      # per-window aircraft id, for per-flight analysis

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return {"obs": self.obs[idx], "fut": self.fut[idx], "t_rel": self.t_rel[idx]}


def get_datasets(filepath, obs_len=43, fut_len=43, seed=SPLIT_SEED, subset=None):
    trajectory, timestamps, icao24, stats = load_netcdf(filepath)

    if subset is not None:
        print(f"WARNING: subset={subset} changes the set of flights, so the split "
              "will NOT match the mentor's full-file split.")
        trajectory, timestamps, icao24 = trajectory[:subset], timestamps[:subset], icao24[:subset]

    idx = split_indices(icao24, seed=seed)
    ds = {
        name: TrajectoryDataset(trajectory, timestamps, icao24, idx[name], stats,
                                obs_len, fut_len)
        for name in ("train", "val", "test")
    }
    for name, d in ds.items():
        print(f"[adsbdatasetBase] {name}: {len(d):,} windows, "
              f"{len(np.unique(d.icao24)):,} flights")
    return ds["train"], ds["val"], ds["test"]


def get_dataloaders(filepath, batch_size=64, obs_len=43, seed=SPLIT_SEED, subset=None,
                    fut_len=43, num_workers=4):
    train_ds, val_ds, test_ds = get_datasets(filepath, obs_len, fut_len, seed, subset)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader