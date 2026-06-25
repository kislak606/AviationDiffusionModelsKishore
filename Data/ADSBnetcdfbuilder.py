import sys
from pathlib import Path
from typing import List, Tuple, Optional

import netCDF4 as nc
import numpy as np
import pandas as pd
import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import config

def load_file(filepath):
    df = pd.read_csv(filepath)
    df_filtered = df[['time_position', 'icao24', 'callsign', 'longitude', 'latitude', 'geo_altitude', 'velocity', 'true_track', 'vertical_rate', 'on_ground']].copy()

    df_filtered["time_position"] = pd.to_numeric(df_filtered["time_position"], errors="coerce")
    df_filtered = df_filtered.dropna(subset=["time_position", "velocity"])
    df_filtered["geo_altitude"]  = df_filtered["geo_altitude"].fillna(0)
    df_filtered["vertical_rate"] = df_filtered["vertical_rate"].fillna(0)

    df_filtered["callsign"] = df_filtered.groupby("icao24")["callsign"].ffill().bfill()
    df_filtered = df_filtered.dropna(subset=['callsign'])

    df_cleaned = df_filtered.drop_duplicates(subset=['icao24', 'time_position'])

    return df_cleaned

def convertToCartesian(df):
    radius = 6371000.0
    originLong = np.radians(-122.3754)
    originLat = np.radians(37.6188)
    longitudeVal = np.radians(df['longitude'].to_numpy())
    latitudeVal = np.radians(df['latitude'].to_numpy())
    altitude = df['geo_altitude'].to_numpy()
    track_rad = np.radians(df["true_track"].to_numpy())

    df_copy = df.copy()

    #Velocities
    df_copy["vx"] = df["velocity"] * np.sin(track_rad)   # East component
    df_copy["vy"] = df["velocity"] * np.cos(track_rad)   # North component
    df_copy["vz"] = df["vertical_rate"]

    #Positions
    df_copy["x"] = (radius + altitude) * np.cos(latitudeVal) * (longitudeVal - originLong)
    df_copy["y"] = radius * (latitudeVal - originLat)
    df_copy["z"] = altitude

    df_copy = df_copy.drop(columns=["latitude", "longitude", "geo_altitude"])
    return df_copy

def segment_aircraft(group, seq_len):
    n = len(group)
    segments = []
    start = 0
    for i in range(1,n):
        dt = group["time_position"].iat[i] - group["time_position"].iat[i - 1]
        same_callsign = group["callsign"].iat[i] == group["callsign"].iat[i - 1]
        if dt > 120 or not same_callsign:
            seg = group.iloc[start:i]
            if len(seg) >= seq_len and not seg["on_ground"].all():
                segments.append((start, i))
            start = i
    seg = group.iloc[start:n]
    if len(seg) >= seq_len and not seg["on_ground"].all():
        segments.append((start, n))
    return segments

def extract_windows(df, seq_len, offset):
    df = df.sort_values(["icao24", "time_position"]).reset_index(drop=True)
    sequences, icao_list, callsign_list, timestamps_list = [], [], [], []

    for icao, group in df.groupby("icao24", sort=False):
        group = group.reset_index(drop=True)
        if len(group) < seq_len:
            continue

        for seg_start, seg_end in segment_aircraft(group, seq_len):
            seg = group.iloc[seg_start:seg_end].reset_index(drop=True)
            for ws in range(0, len(seg) - seq_len + 1, offset):
                window = seg.iloc[ws: ws + seq_len]
                if window["on_ground"].all():
                    continue
                features = window[["x","y","z","vx","vy","vz"]].values.astype(np.float32)
                if (features[:, 2] < -50).any():   # bad altitude check
                    continue
                sequences.append(features)
                icao_list.append(icao)
                callsign_list.append(window["callsign"].iat[0])
                timestamps_list.append(window["time_position"].values)

    return sequences, icao_list, callsign_list, timestamps_list

def to_delta_xy(data):
    N, seq_len, _ = data.shape
    delta_xy = np.zeros((N, seq_len, 2), dtype=np.float32)
    delta_xy[:, 1:, :] = data[:, 1:, :2] - data[:, :-1, :2]

    traj = np.concatenate([delta_xy, data[:, :, 2:]], axis=-1)
    origin = data[:, 0, :2].astype(np.float32)
    return traj, origin

def compute_stats(traj, origin, times):
    stats = {}

    xy_diffs = traj[:, 1:, :2].reshape(-1, 2).astype(np.float64)
    stats["pos_delta_mean"] = xy_diffs.mean(axis=0)
    stats["pos_delta_std"]  = xy_diffs.std(axis=0)

    z_flat = traj[:, :, 2].ravel().astype(np.float64)
    stats["z_mean"] = z_flat.mean()
    stats["z_std"]  = z_flat.std()

    vel_flat = traj[:, :, 3:].reshape(-1, 3).astype(np.float64)
    stats["vel_mean"] = vel_flat.mean(axis=0)
    stats["vel_std"]  = vel_flat.std(axis=0)

    return stats

def write_netcdf(output_path, traj, origin, icao_list, callsign_list, times, stats, seq_len, offset):
    N = traj.shape[0]
    ds = nc.Dataset(output_path, "w", format="NETCDF4")
    try:
        ds.createDimension("sample", N)
        ds.createDimension("sequence", seq_len)
        ds.createDimension("feature", 6)
        ds.createDimension("xy", 2)

        ds.createVariable("icao24", str, ("sample",))[:] = np.array(icao_list, dtype="S")
        ds.createVariable("callsign", str, ("sample",))[:] = np.array(callsign_list, dtype="S")
        ds.createVariable("timestamps", "f8", ("sample", "sequence"))[:] = times

        ds.createVariable("trajectory", "f4", ("sample", "sequence", "feature"), zlib=True, complevel=4)[:] = traj
        ds.createVariable("origin", "f4", ("sample", "xy"))[:] = origin

        for key, val in stats.items():
            setattr(ds, key, val.tolist() if hasattr(val, "tolist") else val)

        ds.seq_len = seq_len
        ds.offset = offset
    finally:
        ds.close()


def build(file_paths, seq_len, offset, output_path):
    dfs = [convertToCartesian(load_file(fp)) for fp in file_paths]
    full_df = pd.concat(dfs, ignore_index=True)

    sequences, icao_list, callsign_list, timestamps_list = extract_windows(full_df, seq_len, offset)
    data  = np.array(sequences)
    times = np.array(timestamps_list)

    traj, origin = to_delta_xy(data)
    stats = compute_stats(traj, origin, times)

    write_netcdf(output_path, traj, origin, icao_list, callsign_list, times, stats, seq_len, offset)