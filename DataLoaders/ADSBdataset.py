import netCDF4 as nc
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ── Step 1: Read the NetCDF file ──────────────────────────────────────────────
def load_netcdf(filepath):
    ds = nc.Dataset(filepath, "r")
    
    trajectory = ds["trajectory"][:]
    timestamps = ds["timestamps"][:]
    icao24 = list(ds["icao24"][:])
    
    # BUG FIX: "ds.array()" doesn't exist — it's np.array() for all of these
    # IMPORTANT: his NetCDF uses feature_mean/std (one array for all 6 features)
    # not separate pos_delta/z/vel stats like yours — so we read those here
    stats = {
        "feature_mean": np.array(ds.feature_mean),   # (6,) covers all features
        "feature_std":  np.array(ds.feature_std),     # (6,)
        "t_rel_mean":   float(ds.t_rel_mean),
        "t_rel_std":    float(ds.t_rel_std),
        "dt_mean":      float(ds.dt_mean),
        "dt_std":       float(ds.dt_std),
    }
    
    ds.close()
    return trajectory, timestamps, icao24, stats


# ── Step 2: Split indices by ICAO ─────────────────────────────────────────────
def split_by_icao(icao24, train_frac=0.85, val_frac=0.10):
    unique_icao = list(set(icao24))
    np.random.shuffle(unique_icao)
    
    n       = len(unique_icao)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    
    train_icao = set(unique_icao[:n_train])
    val_icao   = set(unique_icao[n_train : n_train + n_val])
    test_icao  = set(unique_icao[n_train + n_val:])
    
    train_idx = [i for i, ic in enumerate(icao24) if ic in train_icao]
    val_idx   = [i for i, ic in enumerate(icao24) if ic in val_icao]
    test_idx  = [i for i, ic in enumerate(icao24) if ic in test_icao]
    
    return train_idx, val_idx, test_idx


# ── Step 3: Normalize a trajectory ───────────────────────────────────────────
def normalize(trajectory, stats):
    # trajectory: (86, 6) — his file is absolute x,y,z,vx,vy,vz
    # so we just z-score all 6 features at once using feature_mean/std
    traj = trajectory.copy().astype(np.float32)
    traj = (traj - stats["feature_mean"]) / (stats["feature_std"] + 1e-8)
    return traj


# ── Step 4: The Dataset class ─────────────────────────────────────────────────
class TrajectoryDataset(Dataset):
    def __init__(self, trajectory, timestamps, indices, stats, obs_len=43):
        self.trajectory = trajectory
        self.timestamps = timestamps
        self.indices    = indices
        self.stats      = stats
        self.obs_len    = obs_len

    def __len__(self):
        # BUG FIX: was len(___) — should be self.indices
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        traj = self.trajectory[real_idx]      # (86, 6)
        traj = normalize(traj, self.stats)    # (86, 6) normalized
        
        obs = traj[:self.obs_len]             # (43, 6) first half
        fut = traj[self.obs_len:]             # (43, 6) second half

        # t_rel: seconds since start of window
        ts    = self.timestamps[real_idx]     # (86,) unix timestamps
        t_rel = ts - ts[0]                    # BUG FIX: subtract first timestamp

        # Normalize t_rel
        t_rel = (t_rel - self.stats["t_rel_mean"]) / (self.stats["t_rel_std"] + 1e-8)
        
        return {
            "obs":   torch.tensor(obs,   dtype=torch.float32),
            "fut":   torch.tensor(fut,   dtype=torch.float32),
            "t_rel": torch.tensor(t_rel, dtype=torch.float32),
        }


# ── Step 5: Convenience function ──────────────────────────────────────────────
def get_dataloaders(filepath, batch_size=64, obs_len=43, seed=42, subset=None):
    np.random.seed(seed)
    trajectory, timestamps, icao24, stats = load_netcdf(filepath)

    if subset is not None:
        trajectory = trajectory[:subset]
        timestamps = timestamps[:subset]
        icao24     = icao24[:subset]
    
    trajectory, timestamps, icao24, stats = load_netcdf(filepath)
    train_idx, val_idx, test_idx = split_by_icao(icao24)
    
    train_ds = TrajectoryDataset(trajectory, timestamps, train_idx, stats, obs_len)
    val_ds   = TrajectoryDataset(trajectory, timestamps, val_idx,   stats, obs_len)
    test_ds  = TrajectoryDataset(trajectory, timestamps, test_idx,  stats, obs_len)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = get_dataloaders(r"D:\trajectories_adsblol_seq86_stage2.nc")
batch = next(iter(train_loader))
print(batch["obs"].shape)    # expect (64, 43, 6)
print(batch["fut"].shape)    # expect (64, 43, 6)
print(batch["t_rel"].shape)  # expect (64, 86)