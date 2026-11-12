"""
dit_RoPETS_impose.py - RoPE-timestamps DiT with token imposition.

Identical to models/dit_RoPE_timestamps.py::TrajectoryDiT (same RoPE attention driven by the real
t_rel timestamps x pos_scale, same AdaLN blocks, same t * 1000 flow-time embedding, same
block-causal mask) except that each future token also gets a learned flag embedding saying whether
it is imposed (known) or free. An imposed token's TIME is still given by RoPE through t_rel, so the
model knows both what the known point is and when it happens.

    model(x_obs, x_t, t, t_rel, impose_mask=None)

    x_obs        (B, 43, 6)   normalised observed context
    x_t          (B, 43, 6)   interpolated future; at imposed positions it must already hold the
                              clean value (forward_cfm_impose / euler_sample_impose do this)
    t            (B,)         flow time in (0, 1)
    t_rel        (B, 86)      normalised relative timestamps
    impose_mask  (B, 43) bool True = this future token is imposed. None = nothing imposed, i.e. the
                              ordinary model, so models.cfm.euler_sample works with it (k=0 eval).

The class is called TrajectoryDiT on purpose, so eval scripts that do
importlib.import_module(...).TrajectoryDiT work with this module unchanged.
Not state_dict-compatible with RoPE-TS checkpoints (extra `impose_emb.weight`).

Place in models/ next to dit_RoPE_timestamps.py. Self-test (CPU, a few seconds), from the repo root:
    python -m models.dit_RoPETS_impose
"""
import torch
import torch.nn as nn

from models.dit_RoPE_timestamps import AdaLNBlock, sinusoidal_embedding


class TrajectoryDiT(nn.Module):
    def __init__(self,
                 obs_len   = 43,
                 fut_len   = 43,
                 in_dim    = 6,
                 d_model   = 256,
                 n_heads   = 8,
                 n_layers  = 6,
                 dropout   = 0.1,
                 pos_scale = 30.0):
        super().__init__()
        # same rescaling as dit_RoPE_timestamps.py: ~1 RoPE unit per average sample gap
        self.pos_scale = pos_scale
        self.obs_len   = obs_len
        self.fut_len   = fut_len
        seq_len        = obs_len + fut_len

        self.input_proj = nn.Linear(in_dim, d_model)

        # NEW: 0 = observed / free future token, 1 = imposed (known) future token
        self.impose_emb = nn.Embedding(2, d_model)

        self.tflow_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.blocks = nn.ModuleList([
            AdaLNBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, in_dim)

        # obs tokens cannot attend to future tokens (True = ignore); future tokens, imposed or
        # not, attend to everything, which is how imposed points inform the free ones
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        mask[:obs_len, obs_len:] = True
        self.register_buffer("attn_mask", mask)

    def forward(self, x_obs, x_t, t, t_rel, impose_mask=None):
        B      = x_obs.shape[0]
        device = x_obs.device

        tokens = torch.cat([
            self.input_proj(x_obs),
            self.input_proj(x_t),
        ], dim=1)                                                        # (B, 86, d_model)

        # which future tokens are imposed (observed tokens are always flag 0)
        if impose_mask is None:
            impose_mask = torch.zeros(B, self.fut_len, dtype=torch.bool, device=device)
        flags = torch.cat([
            torch.zeros(B, self.obs_len, dtype=torch.long, device=device),
            impose_mask.long(),
        ], dim=1)                                                        # (B, 86)
        tokens = tokens + self.impose_emb(flags)

        positions = t_rel * self.pos_scale                               # (B, 86) RoPE positions

        cond = self.tflow_mlp(
            sinusoidal_embedding(t.float() * 1000.0, tokens.shape[-1])
        )                                                                # (B, d_model)

        for block in self.blocks:
            tokens = block(tokens, cond, positions, mask=self.attn_mask)

        fut_tokens = self.output_norm(tokens[:, self.obs_len:, :])
        return self.output_head(fut_tokens)                              # predicted velocity (B, 43, 6)


if __name__ == "__main__":
    from models.cfm_impose import (sample_impose_mask, forward_cfm_impose,
                                   cfm_loss_impose, euler_sample_impose)
    torch.manual_seed(0)
    B = 4
    model = TrajectoryDiT(d_model=64, n_heads=4, n_layers=2)
    print(f"Parameters (small test model): {sum(p.numel() for p in model.parameters()):,}")

    obs, fut = torch.randn(B, 43, 6), torch.randn(B, 43, 6)
    t_rel, t = torch.randn(B, 86), torch.rand(B)

    mask = sample_impose_mask(B, 43, 3, device="cpu")
    x_t, _, v_tgt = forward_cfm_impose(fut, t, mask)
    out = model(obs, x_t, t, t_rel, mask)
    assert out.shape == (B, 43, 6)
    assert model(obs, x_t, t, t_rel).shape == (B, 43, 6)               # no mask also works

    cfm_loss_impose(out, v_tgt, mask).backward()
    assert model.impose_emb.weight.grad is not None                    # flag embedding trains

    # flipping the flags changes the output, i.e. the model actually sees them
    model.eval()
    with torch.no_grad():
        a = model(obs, x_t, t, t_rel, torch.zeros(B, 43, dtype=torch.bool))
        b = model(obs, x_t, t, t_rel, torch.ones(B, 43, dtype=torch.bool))
    assert not torch.allclose(a, b)

    # the sampler keeps imposed positions exactly at their known values
    m3 = sample_impose_mask(B, 43, 3, device="cpu", k=3, allow_last=False)
    s  = euler_sample_impose(model, obs, t_rel, fut, m3, n_samples=5, n_steps=4, device="cpu")
    assert s.shape == (5, B, 43, 6)
    assert torch.equal(s[:, m3], fut[None].expand(5, -1, -1, -1)[:, m3])

    print("dit_RoPETS_impose self-test OK")