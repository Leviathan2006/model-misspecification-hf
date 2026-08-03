"""Validate probe.py against operators with KNOWN structure.

If the probe reads the right exponent off a k^p operator, flags a spatially varying
operator as non-equivariant, and flags a quadratic operator as state-dependent, the
probing math is sound and the per-PDE scripts are greenlit.
"""

import torch
from probe import probe_operator

torch.manual_seed(0)


def spectral(p, alpha, dim):
    """M(u) = u + alpha * (-Delta)^{p/2} u  -> correction symbol alpha*|k|^p."""
    def M(u):
        if dim == 1:
            n = u.shape[-1]
            k = torch.arange(n // 2 + 1, dtype=torch.float32)
            sym = alpha * k ** p
            return u + torch.fft.irfft(torch.fft.rfft(u, dim=-1) * sym, n=n, dim=-1)
        h, w = u.shape[-2], u.shape[-1]
        ky = torch.fft.fftfreq(h, d=1.0 / h).abs()
        kx = torch.arange(w // 2 + 1, dtype=torch.float32)
        kk = torch.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
        sym = alpha * kk ** p
        return u + torch.fft.irfft2(torch.fft.rfft2(u, dim=(-2, -1)) * sym, s=(h, w), dim=(-2, -1))
    return M


def heterogeneous(alpha, dim):
    """M(u) = u + alpha * mask(x) * u  -> non-equivariant."""
    def M(u):
        grid = u.shape[1:]
        coords = [torch.linspace(0, 1, n) for n in grid]
        if dim == 1:
            mask = torch.sin(3.14159 * coords[0])
        else:
            mask = torch.sin(3.14159 * coords[0])[:, None] * torch.sin(3.14159 * coords[1])[None, :]
        return u + alpha * mask[None] * u
    return M


def quadratic(alpha, dim):
    """M(u) = u + alpha * u^2  -> state-dependent J."""
    return lambda u: u + alpha * u ** 2


def smooth_states(S, grid, cutoff=8):
    x = torch.randn(S, *grid)
    if len(grid) == 1:
        Xf = torch.fft.rfft(x, dim=-1)
        Xf[:, cutoff:] = 0
        return torch.fft.irfft(Xf, n=grid[0], dim=-1)
    Xf = torch.fft.rfft2(x, dim=(-2, -1))
    Xf[:, cutoff:, :] = 0
    Xf[:, :, cutoff:] = 0
    return torch.fft.irfft2(Xf, s=grid, dim=(-2, -1))


def run(name, M, grid, expect):
    S = 6
    base = smooth_states(S, grid)
    r = probe_operator(M, base, n_rand=3)
    print(f"{name:28s} exp={r['exponent']:+.2f} mag={r['magnitude']:.3f} "
          f"loc={r['locality']:.1f} equiv={r['equivariance']:.3f} svar={r['state_var']:.3f}")
    return r


print("=== 1D ===")
g1 = (128,)
r_id = run("identity", lambda u: u, g1, None)
r_p2 = run("k^2 (viscosity)", spectral(2.0, 3e-3, 1), g1, 2.0)
r_p15 = run("k^1.5 (fractional)", spectral(1.5, 1e-2, 1), g1, 1.5)
r_p3 = run("k^3 (dispersive)", spectral(3.0, 1e-4, 1), g1, 3.0)
r_het = run("heterogeneous", heterogeneous(0.5, 1), g1, None)
r_nl = run("quadratic (nonlinear)", quadratic(0.3, 1), g1, None)

print("=== 2D ===")
g2 = (48, 48)
r2_p2 = run("k^2 2D", spectral(2.0, 3e-3, 2), g2, 2.0)
r2_het = run("heterogeneous 2D", heterogeneous(0.5, 2), g2, None)

print("\n=== checks ===")
ok = True


def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg)
    ok = ok and cond


check(abs(r_p2["exponent"] - 2.0) < 0.4, f"1D exponent~2 (got {r_p2['exponent']:.2f})")
check(abs(r_p15["exponent"] - 1.5) < 0.4, f"1D exponent~1.5 (got {r_p15['exponent']:.2f})")
check(abs(r_p3["exponent"] - 3.0) < 0.5, f"1D exponent~3 (got {r_p3['exponent']:.2f})")
check(abs(r2_p2["exponent"] - 2.0) < 0.5, f"2D exponent~2 (got {r2_p2['exponent']:.2f})")
check(r_id["magnitude"] < 1e-3, f"identity magnitude~0 (got {r_id['magnitude']:.1e})")
check(r_het["equivariance"] > 5 * r_p2["equivariance"], "heterogeneous flagged non-equivariant")
check(r2_het["equivariance"] > 5 * r2_p2["equivariance"], "heterogeneous 2D flagged non-equivariant")
check(r_nl["state_var"] > 5 * r_p2["state_var"], "quadratic flagged state-dependent")
print("\n" + ("ALL PROBE CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
