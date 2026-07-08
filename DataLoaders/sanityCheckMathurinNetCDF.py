import netCDF4 as nc

def inspect_netcdf(filepath):
    ds = nc.Dataset(filepath, "r")
    print("=== VARIABLES ===")
    for name, var in ds.variables.items():
        print(f"  {name}: {var.shape} ({var.dtype})")
    print("\n=== ATTRIBUTES ===")
    for attr in ds.ncattrs():
        val = getattr(ds, attr)
        print(f"  {attr}: {val}")
    ds.close()

inspect_netcdf(r"D:\trajectories_adsblol_seq86_stage2.nc")

import torch
print(torch.cuda.is_available())        # should print True
print(torch.cuda.get_device_name(0))    # should print "NVIDIA GeForce RTX 2060"
print(torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")  # should print ~6.0