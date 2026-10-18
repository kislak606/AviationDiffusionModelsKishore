import torch

for name, path in [
    ("Sinusoidal CFM", r"checkpoints_cfm\best_cfm.pt"),
    ("RoPE-A",         r"checkpoints_cfm_rope_og\best_rope_og.pt"),
    ("RoPE-B",         r"checkpoints_cfm_rope_ts\best_rope_ts.pt"),
]:
    ckpt = torch.load(path, map_location="cpu")
    print(f"{name}: epoch {ckpt['epoch']+1}, val FDE {ckpt['val_fde']:.1f}m")


ckpt = torch.load(r"checkpoints_cfm_rope_og\last_rope_og.pt", map_location="cpu")
print("Last saved epoch:", ckpt["epoch"] + 1)
print("val_fde:", ckpt["val_fde"])