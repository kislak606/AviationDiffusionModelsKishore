"""
cfm_impose.py - conditional flow matching with token imposition (self-contained).

For each training sample, k ~ Uniform{0..max_k} of the 43 future positions are "imposed": the
model sees the CLEAN value there at every flow time, is told which positions they are, and is
not trained on them. At inference you supply up to max_k known future points and they stay
clamped to their known values during Euler integration.

Same convention as models/cfm.py: x_t = (1 - t) * noise + t * data, target v = data - noise,
Euler from t=0 (noise) to t=1 (data).

Place in models/ next to cfm.py. Self-test (CPU, a few seconds), from the repo root:
    python -m models.cfm_impose
"""
import torch


def sample_flow_time(B, device="cuda", t_mean=1.0, t_std=1.0):
    """t ~ sigmoid(N(t_mean, t_std)), identical to models/cfm.py."""
    u = torch.randn(B, device=device) * t_std + t_mean
    return torch.sigmoid(u)


def sample_impose_mask(B, fut_len=43, max_k=3, device="cuda", allow_last=True, k=None):
    """
    Random imposition mask, (B, fut_len) bool, True = imposed.

    k=None       -> each sample imposes k_i ~ Uniform{0, ..., max_k} positions (k_i=0 allowed, so
                    the model still works with nothing imposed).
    k=int        -> exactly k positions for every sample (k=0: no imposition; k=1..3: evaluation).
    allow_last   -> False never imposes the final future step. Use it for validation/eval of FDE,
                    otherwise a revealed last point makes FDE trivially zero for that sample.
    """
    if k is None:
        k_i = torch.randint(0, max_k + 1, (B,), device=device)
    else:
        assert 0 <= k <= max_k, f"k={k} must be in [0, {max_k}]"
        k_i = torch.full((B,), int(k), dtype=torch.long, device=device)

    scores = torch.rand(B, fut_len, device=device)
    if not allow_last:
        scores[:, -1] = -1.0                              # lowest score: never in the top max_k
    top  = scores.topk(max_k, dim=1).indices              # (B, max_k) distinct random positions
    keep = torch.arange(max_k, device=device)[None, :] < k_i[:, None]
    mask = torch.zeros(B, fut_len, dtype=torch.bool, device=device)
    mask.scatter_(1, top, keep)
    return mask


def forward_cfm_impose(x_0, t, impose_mask):
    """
    CFM forward process with imposed tokens.

    x_0:         (B, 43, 6) clean future trajectory
    t:           (B,) flow time in (0, 1)
    impose_mask: (B, 43) bool

    Returns:
        x_t:   (B, 43, 6) model input; equals the clean x_0 at imposed positions
        noise: (B, 43, 6)
        v_tgt: (B, 43, 6) target velocity (meaningless at imposed positions, masked in the loss)
    """
    noise = torch.randn_like(x_0)
    tb    = t.view(-1, 1, 1)
    x_t   = (1 - tb) * noise + tb * x_0
    x_t   = torch.where(impose_mask[..., None], x_0, x_t)
    v_tgt = x_0 - noise
    return x_t, noise, v_tgt


def cfm_loss_impose(v_pred, v_tgt, impose_mask, weights=None):
    """
    Velocity MSE over the FREE (non-imposed) future tokens only.

    weights: optional (fut_len,) per-step weights, e.g. 2.0 on the last quarter. The weighted sum
    is divided by the number of free tokens, so with nothing imposed this equals the baseline
    (mse_loss(..., reduction="none") * weights).mean() and the loss scale stays comparable.
    """
    sq   = (v_pred - v_tgt).pow(2).mean(dim=-1)           # (B, 43) per-token MSE
    free = (~impose_mask).float()
    w    = free if weights is None else free * weights[None, :]
    return (sq * w).sum() / free.sum().clamp(min=1.0)


@torch.no_grad()
def euler_sample_impose(model, obs, t_rel, known, impose_mask,
                        n_samples=20, n_steps=20, device=None):
    """
    Euler ODE sampling with imposed tokens held at their known values.

    obs:         (B, 43, 6)
    t_rel:       (B, 86)
    known:       (B, 43, 6) clean values; only entries where impose_mask is True are used
    impose_mask: (B, 43) bool (all False = ordinary sampling)

    Returns (n_samples, B, 43, 6); imposed positions equal `known` exactly.
    """
    B, L, F = known.shape
    device  = obs.device if device is None else torch.device(device)
    dt      = 1.0 / n_steps

    def rep(x):   # (B, ...) -> (n_samples * B, ...), sample-major like models/cfm.py
        return x.unsqueeze(0).expand(n_samples, *x.shape).reshape(n_samples * B, *x.shape[1:])

    obs_rep, trel_rep   = rep(obs), rep(t_rel)
    known_rep, mask_rep = rep(known), rep(impose_mask)
    m = mask_rep[..., None]                               # (n*B, 43, 1)

    x = torch.randn(n_samples * B, L, F, device=device)
    x = torch.where(m, known_rep, x)

    model.eval()
    for step in range(n_steps):
        t_tensor = torch.full((n_samples * B,), step * dt, device=device, dtype=torch.float32)
        v = model(obs_rep, x, t_tensor, trel_rep, mask_rep)
        x = x + v * dt
        x = torch.where(m, known_rep, x)                  # re-clamp imposed tokens every step

    return x.reshape(n_samples, B, L, F)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, F = 4096, 43, 6

    m = sample_impose_mask(B, L, 3, device="cpu")
    cnt = m.sum(1)
    assert cnt.min() >= 0 and cnt.max() <= 3
    print("k distribution for k=0..3 (expect ~0.25 each):",
          [round(v, 3) for v in (torch.bincount(cnt, minlength=4).float() / B).tolist()])
    print("every future position gets imposed sometimes:", bool(m.any(0).all()))

    for k in range(4):
        assert (sample_impose_mask(B, L, 3, device="cpu", k=k).sum(1) == k).all(), k
    assert not sample_impose_mask(B, L, 3, device="cpu", allow_last=False)[:, -1].any()
    assert not sample_impose_mask(B, L, 3, device="cpu", allow_last=False, k=3)[:, -1].any()

    x0 = torch.randn(B, L, F)
    t  = torch.rand(B)
    x_t, noise, v_tgt = forward_cfm_impose(x0, t, m)
    tb = t.view(-1, 1, 1)
    assert torch.equal(x_t[m], x0[m])                                    # imposed = clean
    assert torch.allclose(x_t[~m], ((1 - tb) * noise + tb * x0)[~m])     # free = interpolated
    assert torch.allclose(v_tgt, x0 - noise)

    # with nothing imposed the loss equals the baseline loss
    vp = torch.randn(B, L, F)
    w  = torch.ones(L)
    w[-L // 4:] = 2.0
    base = (torch.nn.functional.mse_loss(vp, v_tgt, reduction="none") * w[None, :, None]).mean()
    new  = cfm_loss_impose(vp, v_tgt, torch.zeros(B, L, dtype=torch.bool), w)
    assert torch.allclose(base, new, rtol=1e-5), (base.item(), new.item())

    # imposed tokens contribute nothing to the loss
    vp2 = vp.clone()
    vp2[m] += 100.0
    assert torch.allclose(cfm_loss_impose(vp, v_tgt, m, w), cfm_loss_impose(vp2, v_tgt, m, w))

    print("cfm_impose self-test OK")