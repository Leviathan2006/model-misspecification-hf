"""Universal operator-probing toolkit.

Given a field-to-field map M: u_theta -> u (a trained neural operator wrapped as a
callable on fields), probe its local linearization J = dM/du_theta and read five
features of the correction operator (J - I) that fingerprint the misspecification:

    magnitude    ||J - I||           how far the prior is from the truth
    exponent     slope of |T(k)|     differential order of the discrepancy (k^p)
    locality     impulse spread      local vs nonlocal
    equivariance ||[J, shift]||       constant-coefficient vs heterogeneous
    state_var    CoV of gain         linear vs state-dependent (nonlinearity)

Everything is a finite-difference Jacobian-vector product, so the pipeline is
identical for 1D and 2D fields and for linear and nonlinear operators. J varying
across base states is not a special case -- it is measured (state_var).
"""

import torch

EPS_REL = 1e-2


def _rms(x):
    return x.pow(2).mean().clamp_min(1e-30).sqrt()


def jvp(M, u0, d, eps):
    """Central finite-difference J @ d at base state u0 (both shape (1, *grid))."""
    return (M(u0 + eps * d) - M(u0 - eps * d)) / (2.0 * eps)


def _unit(x):
    return x / x.norm().clamp_min(1e-30)


def _impulse(grid, device):
    d = torch.zeros((1, *grid), device=device)
    if len(grid) == 1:
        d[0, grid[0] // 2] = 1.0
    else:
        d[0, grid[0] // 2, grid[1] // 2] = 1.0
    return d


def _radial_spectrum(h):
    """|FFT(h)| reduced to a 1D function of wavenumber. h: (1, *grid)."""
    g = h[0]
    if g.dim() == 1:
        H = torch.fft.rfft(g)
        k = torch.arange(H.shape[0], device=g.device).float()
        return k, H.abs()
    H = torch.fft.rfft2(g)
    ny, nxh = H.shape
    ky = torch.fft.fftfreq(ny, d=1.0 / ny, device=g.device).abs()
    kx = torch.arange(nxh, device=g.device).float()
    KY = ky[:, None].expand(ny, nxh)
    KX = kx[None, :].expand(ny, nxh)
    kr = torch.sqrt(KX ** 2 + KY ** 2).reshape(-1)
    mag = H.abs().reshape(-1)
    kmax = int(min(ny // 2, nxh - 1))
    kb = torch.arange(kmax + 1, device=g.device).float()
    out = torch.zeros_like(kb)
    cnt = torch.zeros_like(kb)
    idx = kr.round().long().clamp(max=kmax)
    out.index_add_(0, idx, mag)
    cnt.index_add_(0, idx, torch.ones_like(mag))
    return kb, out / cnt.clamp_min(1.0)


def _fit_slope(k, T, klo, khi):
    m = (k >= klo) & (k <= khi) & (T > 0)
    if m.sum() < 3:
        return float("nan")
    lk = torch.log(k[m]); lt = torch.log(T[m])
    lk = lk - lk.mean(); lt = lt - lt.mean()
    return float((lk * lt).sum() / (lk * lk).sum().clamp_min(1e-30))


def _corr_impulse(M, u0, eps):
    d = _unit(_impulse(tuple(u0.shape[1:]), u0.device))
    h = jvp(M, u0, d, eps) - d
    return h


def probe_operator(M, base_states, n_rand=3, seed=0):
    """base_states: (S, *grid) tensor of representative u_theta fields."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = base_states.device
    grid = tuple(base_states.shape[1:])
    S = base_states.shape[0]

    mags, spectra, radii, equis, gains = [], [], [], [], []
    for s in range(S):
        u0 = base_states[s:s + 1]
        eps = EPS_REL * _rms(u0)

        # magnitude: mean ||(J-I) d|| / ||d|| over random unit directions
        acc = 0.0
        for _ in range(n_rand):
            d = _unit(torch.randn((1, *grid), generator=g).to(dev))
            corr = jvp(M, u0, d, eps) - d
            acc += float(corr.norm() / d.norm().clamp_min(1e-30))
        mags.append(acc / n_rand)

        # spectral transfer of the correction from the impulse response
        h = _corr_impulse(M, u0, eps)
        k, T = _radial_spectrum(h)
        spectra.append((k, T))
        gains.append(float(T.sum()))

        # locality: normalized spatial radius of |impulse response|
        a = h[0].abs()
        coords = [torch.arange(n, device=dev).float() - n / 2 for n in grid]
        if len(grid) == 1:
            r2 = coords[0] ** 2
        else:
            r2 = coords[0][:, None] ** 2 + coords[1][None, :] ** 2
        w = a / a.sum().clamp_min(1e-30)
        radii.append(float((w * r2).sum().sqrt()))

        # equivariance: ||J(shift d) - shift(J d)|| / ||J d|| for a random d
        d = _unit(torch.randn((1, *grid), generator=g).to(dev))
        sh = tuple(n // 4 for n in grid)
        dims = tuple(range(1, 1 + len(grid)))
        Jd = jvp(M, u0, d, eps)
        Jsd = jvp(M, u0, torch.roll(d, shifts=sh, dims=dims), eps)
        equis.append(float((Jsd - torch.roll(Jd, shifts=sh, dims=dims)).norm()
                           / Jd.norm().clamp_min(1e-30)))

    # aggregate; exponent from the mean spectrum
    kref = spectra[0][0]
    Tstack = torch.stack([T for _, T in spectra], 0)
    Tmean = Tstack.mean(0)
    kmax = float(kref.max())
    p = _fit_slope(kref, Tmean, klo=2.0, khi=max(4.0, kmax / 8.0))

    g_t = torch.tensor(gains)
    state_var = float(g_t.std() / g_t.mean().abs().clamp_min(1e-30))

    return {
        "magnitude": float(torch.tensor(mags).mean()),
        "exponent": p,
        "locality": float(torch.tensor(radii).mean()),
        "equivariance": float(torch.tensor(equis).mean()),
        "state_var": state_var,
        "k": kref.cpu().numpy(),
        "transfer": Tmean.cpu().numpy(),
    }
