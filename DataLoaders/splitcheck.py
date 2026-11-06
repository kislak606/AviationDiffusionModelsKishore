import numpy as np, netCDF4 as nc

ds = nc.Dataset("trajectories_adsblol_seq86_stage2.nc"); icao = np.array(ds["icao24"][:]); ds.close()
uniq = np.unique(icao); n = len(uniq)

# his split
a = uniq.copy(); np.random.default_rng(42).shuffle(a)
his_test  = set(a[n - int(n*0.05):])
his_train = set(a[:n - int(n*0.10) - int(n*0.05)])

# your split
b = sorted(set(icao)); np.random.RandomState(42).shuffle(b)
nt, nv = int(n*0.85), int(n*0.10)
my_train, my_test = set(b[:nt]), set(b[nt+nv:])

print(len(his_test), len(my_test))
print("your test flights in his train:", len(my_test & his_train) / len(my_test))
print("his test flights in your train:", len(his_test & my_train) / len(his_test))