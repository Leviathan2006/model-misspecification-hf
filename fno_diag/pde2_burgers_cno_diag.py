"""
P2 (viscous Burgers): diagnose misspecification by probing a learned correction operator.

Pipeline per misspecification level {low, med, high, max}:
  1. train the physics prior (label-free) on the assumed residual -> u_theta = prior(v)
  2. train a CONVOLUTIONAL NEURAL OPERATOR (CNO) M_psi: u_theta -> u
  3. probe M_psi's local Jacobian J = dM_psi/du_theta ON the data manifold and read:
        magnitude     ||J - I||                          (severity)
        mag_slope     |g(k)-1| ~ k^p   (real transfer)   (dissipative order: viscosity k^2, ...)
        phase_slope   arg g(k) ~ k^q   (phase transfer)  (advection q~1, dispersion/KdV q~3)
        locality      impulse-response spread            (local vs nonlocal)
        heterogeneity local response varies with x       (constant vs varying coefficient)
        state_var     transfer varies across states      (linear vs nonlinear)
        k_support     wavenumber the data actually excites (identifiability limit)

CNO is adapted from camlab-ethz/ConvolutionalNeuralOperator (CNO2d_simplified), generalized
to rectangular grids. It does NOT truncate Fourier modes (unlike FNO), so the spectral
signatures survive. Probing uses in-distribution directions (not broadband impulses) and the
COMPLEX transfer (magnitude AND phase) -- the two fixes over the earlier FNO probe.

Run: set DATA_PATH to your Kaggle dataset .npz (keys: x, t, {train,val,test}_{v,u}).
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# >>> SET THIS to your uploaded Kaggle dataset (the pde2 burgers .npz) <<<
DATA_PATH = os.environ.get("MM_DATA", "/kaggle/input/FILL-ME/pde2_burgers.npz")
# ============================================================================
OUT_DIR = os.environ.get("MM_OUT_DIR", "/kaggle/working" if os.path.isdir("/kaggle/working") else ".")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
PI = math.pi

NU_TRUE, DELTA = 0.01, 1e-3
MISSPECS = os.environ.get("MM_LEVELS", "low,med,high,max").split(",")

# ---- prior (physics-informed DeepONet) hyperparameters ----
M_FOUR = 16
PRIOR_BH, PRIOR_P, PRIOR_TH = [256, 256], 128, [128, 128, 128]
LR, REL_TOL, PRIOR_REL_TOL = 1e-3, 5e-3, 1e-4
STEPS_PER_EPOCH, PHYS_BATCH, N_COLLOC = 16, 32, 3000
PHYS_EPOCHS, PHYS_PATIENCE, W_IC = 800, 50, 20.0

# ---- CNO (correction operator M_psi) hyperparameters ----
CNO_LAYERS, CNO_CH, CNO_RES, CNO_RES_NECK = 3, 32, 4, 4
CNO_EPOCHS, CNO_PATIENCE, CNO_BATCH, CNO_STEPS = 300, 30, 5, 16

# ---- probe ----
N_BASE, EPS_REL, N_RAND = 6, 1e-2, 4

# ---- smoke mode: tiny synthetic data + tiny budgets, to dry-run the wiring on CPU ----
SMOKE = bool(int(os.environ.get("MM_SMOKE", "0")))
if SMOKE:
    MISSPECS = ["low", "max"]
    M_FOUR = 6
    PHYS_EPOCHS, PHYS_PATIENCE, STEPS_PER_EPOCH, N_COLLOC = 3, 3, 2, 200
    CNO_EPOCHS, CNO_PATIENCE, CNO_STEPS, CNO_LAYERS, CNO_CH = 3, 3, 2, 2, 8
    N_BASE, N_RAND = 3, 2


# ============================== data ==============================
if SMOKE:
    _rng = np.random.default_rng(0)
    Nx, Nt = 48, 32
    x_np = np.linspace(0, 1, Nx, endpoint=False).astype(np.float32)
    t_np = np.linspace(0, 0.5, Nt).astype(np.float32)

    def _syn(n):
        v = _rng.standard_normal((n, Nx)).astype(np.float32)
        vf = np.fft.rfft(v, axis=-1); vf[:, 6:] = 0
        v = np.fft.irfft(vf, n=Nx, axis=-1).astype(np.float32)
        u = np.repeat(v[:, None, :], Nt, axis=1) + 0.1 * _rng.standard_normal((n, Nt, Nx)).astype(np.float32)
        u[:, 0] = v
        return v, u
    tr_v_n, tr_u_n = _syn(40); va_v_n, va_u_n = _syn(8); te_v_n, te_u_n = _syn(8)
else:
    d = np.load(DATA_PATH)
    x_np, t_np = d["x"].astype(np.float32), d["t"].astype(np.float32)
    Nx, Nt = x_np.shape[0], t_np.shape[0]
    tr_v_n, tr_u_n = d["train_v"].astype(np.float32), d["train_u"].astype(np.float32)
    va_v_n, va_u_n = d["val_v"].astype(np.float32), d["val_u"].astype(np.float32)
    te_v_n, te_u_n = d["test_v"].astype(np.float32), d["test_u"].astype(np.float32)
Ng, T_END = Nt * Nx, float(t_np.max())

V_MEAN, V_STD = float(tr_v_n.mean()), float(tr_v_n.std())
U_MEAN, U_STD = float(tr_u_n.mean()), float(tr_u_n.std())
C0 = float(np.sqrt((tr_v_n ** 2).mean()))

to = lambda a: torch.tensor(a, device=DEVICE)
tr_v, tr_u = to(tr_v_n), to(tr_u_n)
va_v, va_u = to(va_v_n), to(va_u_n)
te_v, te_u = to(te_v_n), to(te_u_n)
ntr, nval, nte = tr_v.shape[0], va_v.shape[0], te_v.shape[0]

x_row, t_row = to(x_np), to(t_np)
TT, XX = torch.meshgrid(t_row, x_row, indexing="ij")
grid_full = torch.stack([XX.reshape(-1), TT.reshape(-1)], dim=-1)
ic_coords = torch.stack([x_row, torch.zeros_like(x_row)], dim=-1)
X2D, T2D = XX, TT
ms = torch.arange(1, M_FOUR + 1, device=DEVICE).float()
FEAT = 2 * M_FOUR + 1
VAL_COLL = torch.rand(PHYS_BATCH, N_COLLOC, 2, device=DEVICE); VAL_COLL[..., 1] *= T_END


def vnorm(v): return (v - V_MEAN) / V_STD
def gnorm(g): return (g - U_MEAN) / U_STD


def features(c):
    x = c[..., 0:1]; t = c[..., 1:2]
    ang = 2 * PI * x * ms
    return torch.cat([torch.sin(ang), torch.cos(ang), t / T_END], dim=-1)


# ============================== physics prior ==============================
class MLP(nn.Module):
    def __init__(self, sizes, act=nn.Tanh):
        super().__init__()
        L = []
        for i in range(len(sizes) - 1):
            L.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                L.append(act())
        self.net = nn.Sequential(*L)

    def forward(self, x): return self.net(x)


class DeepONetST(nn.Module):
    def __init__(self, extra, branch_in, bh, P, th):
        super().__init__()
        self.branch = MLP([branch_in] + bh + [P])
        self.trunk = MLP([FEAT + extra] + th + [P])
        self.b0 = nn.Parameter(torch.zeros(1)); self.extra = extra

    def eval_grid(self, vin, coords):
        b = self.branch(vin)
        return b @ self.trunk(features(coords)).t() + self.b0

    def pointwise(self, vin, coords):
        b = self.branch(vin)
        return torch.einsum("bp,bnp->bn", b, self.trunk(features(coords))) + self.b0


def residual(name, u, ux, uxx, uxxx, ut):
    if name == "low":
        return ut + u * ux - 0.013 * uxx
    if name == "med":
        return ut + u * ux - 0.03 * uxx
    if name == "high":
        return ut + C0 * ux - NU_TRUE * uxx
    return ut + u * ux + DELTA * uxxx


def derivs(model, vin, coords):
    u = model.pointwise(vin, coords)
    g1 = torch.autograd.grad(u.sum(), coords, create_graph=True)[0]
    ux, ut = g1[..., 0], g1[..., 1]
    g2 = torch.autograd.grad(ux.sum(), coords, create_graph=True)[0]
    uxx = g2[..., 0]
    g3 = torch.autograd.grad(uxx.sum(), coords, create_graph=True)[0]
    uxxx = g3[..., 0]
    return u, ux, uxx, uxxx, ut


def clone(m): return {k: v.detach().clone() for k, v in m.state_dict().items()}


def fit(model, step, val, epochs, patience, rel_tol):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, state, wait = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for _ in range(CNO_STEPS if model.__class__.__name__ == "CNO2d" else STEPS_PER_EPOCH):
            opt.zero_grad(); loss = step(); loss.backward(); opt.step()
        model.eval(); vl = val()
        if vl < best * (1 - rel_tol):
            best, state, wait = vl, clone(model), 0
        else:
            wait += 1
        if wait >= patience:
            break
    model.load_state_dict(state); return model


def train_prior(name):
    prior = DeepONetST(0, Nx, PRIOR_BH, PRIOR_P, PRIOR_TH).to(DEVICE)

    def step():
        idx = torch.randint(0, ntr, (PHYS_BATCH,), device=DEVICE)
        coll = torch.rand(PHYS_BATCH, N_COLLOC, 2, device=DEVICE); coll[..., 1] *= T_END
        coll = coll.requires_grad_(True); vin = vnorm(tr_v[idx])
        u, ux, uxx, uxxx, ut = derivs(prior, vin, coll)
        ic = ((prior.eval_grid(vin, ic_coords) - tr_v[idx]) ** 2).mean()
        return (residual(name, u, ux, uxx, uxxx, ut) ** 2).mean() + W_IC * ic

    def val():
        idx = torch.arange(min(PHYS_BATCH, nval), device=DEVICE)
        coll = VAL_COLL[:idx.shape[0]].clone().requires_grad_(True); vin = vnorm(va_v[idx])
        u, ux, uxx, uxxx, ut = derivs(prior, vin, coll)
        ic = ((prior.eval_grid(vin, ic_coords) - va_v[idx]) ** 2).mean()
        return ((residual(name, u, ux, uxx, uxxx, ut) ** 2).mean() + W_IC * ic).item()

    fit(prior, step, val, PHYS_EPOCHS, PHYS_PATIENCE, PRIOR_REL_TOL)
    for p in prior.parameters():
        p.requires_grad_(False)
    prior.eval(); return prior


@torch.no_grad()
def prior_full(prior, v):
    n = v.shape[0]; out = torch.empty(n, Ng, device=DEVICE)
    for b in range(0, n, 64):
        e = min(b + 64, n); vin = vnorm(v[b:e])
        for c in range(0, Ng, 8192):
            out[b:e, c:c + 8192] = prior.eval_grid(vin, grid_full[c:c + 8192])
    return out.reshape(n, Nt, Nx)


# ============================== CNO (rectangular) ==============================
class CNO_LReLu(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__(); self.i, self.o = in_size, out_size; self.act = nn.LeakyReLU()

    def forward(self, x):
        x = F.interpolate(x, size=(2 * self.i[0], 2 * self.i[1]), mode="bicubic", antialias=True)
        x = self.act(x)
        return F.interpolate(x, size=(self.o[0], self.o[1]), mode="bicubic", antialias=True)


class CNOBlock(nn.Module):
    def __init__(self, ic, oc, in_size, out_size, use_bn=True):
        super().__init__()
        self.conv = nn.Conv2d(ic, oc, 3, padding=1)
        self.bn = nn.BatchNorm2d(oc) if use_bn else nn.Identity()
        self.act = CNO_LReLu(in_size, out_size)

    def forward(self, x): return self.act(self.bn(self.conv(x)))


class LiftProject(nn.Module):
    def __init__(self, ic, oc, size, latent=64):
        super().__init__()
        self.b = CNOBlock(ic, latent, size, size, use_bn=False)
        self.conv = nn.Conv2d(latent, oc, 3, padding=1)

    def forward(self, x): return self.conv(self.b(x))


class ResidualBlock(nn.Module):
    def __init__(self, ch, size, use_bn=True):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1); self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b1 = nn.BatchNorm2d(ch) if use_bn else nn.Identity()
        self.b2 = nn.BatchNorm2d(ch) if use_bn else nn.Identity()
        self.act = CNO_LReLu(size, size)

    def forward(self, x):
        o = self.act(self.b1(self.c1(x))); o = self.b2(self.c2(o)); return x + o


class ResNet(nn.Module):
    def __init__(self, ch, size, n, use_bn=True):
        super().__init__(); self.net = nn.Sequential(*[ResidualBlock(ch, size, use_bn) for _ in range(n)])

    def forward(self, x): return self.net(x)


class CNO2d(nn.Module):
    def __init__(self, in_dim, out_dim, size, N_layers, N_res=4, N_res_neck=4, cm=32, use_bn=True):
        super().__init__()
        self.N = N_layers; lift = cm // 2
        enc_f = [lift] + [2 ** i * cm for i in range(N_layers)]
        dec_in = enc_f[1:][::-1]; dec_out = enc_f[:-1][::-1]
        for i in range(1, N_layers):
            dec_in[i] = 2 * dec_in[i]

        def sz(i): return (size[0] // 2 ** i, size[1] // 2 ** i)
        enc_s = [sz(i) for i in range(N_layers + 1)]
        dec_s = [sz(N_layers - i) for i in range(N_layers + 1)]

        self.lift = LiftProject(in_dim, enc_f[0], size)
        self.project = LiftProject(enc_f[0] + dec_out[-1], out_dim, size)
        self.encoder = nn.ModuleList([CNOBlock(enc_f[i], enc_f[i + 1], enc_s[i], enc_s[i + 1], use_bn) for i in range(N_layers)])
        self.ED = nn.ModuleList([CNOBlock(enc_f[i], enc_f[i], enc_s[i], dec_s[N_layers - i], use_bn) for i in range(N_layers + 1)])
        self.decoder = nn.ModuleList([CNOBlock(dec_in[i], dec_out[i], dec_s[i], dec_s[i + 1], use_bn) for i in range(N_layers)])
        self.res = nn.ModuleList([ResNet(enc_f[l], enc_s[l], N_res, use_bn) for l in range(N_layers)])
        self.res_neck = ResNet(enc_f[N_layers], enc_s[N_layers], N_res_neck, use_bn)

    def forward(self, x):
        x = self.lift(x); skip = []
        for i in range(self.N):
            skip.append(self.res[i](x)); x = self.encoder[i](x)
        x = self.res_neck(x)
        for i in range(self.N):
            if i == 0:
                x = self.ED[self.N - i](x)
            else:
                x = torch.cat((x, self.ED[self.N - i](skip[-i])), 1)
            x = self.decoder[i](x)
        x = torch.cat((x, self.ED[0](skip[0])), 1)
        return self.project(x)


def make_Mfun(cno):
    xb = X2D[None, None]; tb = T2D[None, None]

    def Mfun(uth):                       # uth: (B, Nt, Nx) -> (B, Nt, Nx)
        B = uth.shape[0]
        inp = torch.cat([gnorm(uth)[:, None], xb.expand(B, 1, Nt, Nx), tb.expand(B, 1, Nt, Nx)], dim=1)
        return cno(inp).squeeze(1)
    return Mfun


# ============================== universal probe ==============================
def _rms(x): return x.pow(2).mean().clamp_min(1e-30).sqrt()


def _jvp(M, u0, dvec, eps): return (M(u0 + eps * dvec) - M(u0 - eps * dvec)) / (2 * eps)


def _slope(k, y, klo, khi):
    m = (k >= klo) & (k <= khi) & (y > 0)
    if int(m.sum()) < 3:
        return float("nan")
    lk, ly = torch.log(k[m]), torch.log(y[m])
    lk, ly = lk - lk.mean(), ly - ly.mean()
    return float((lk * ly).sum() / (lk * lk).sum().clamp_min(1e-30))


def data_support(fields, thresh=0.01):
    P = torch.zeros(Nx // 2 + 1, device=fields.device)
    for i in range(fields.shape[0]):
        P += torch.fft.rfft(fields[i], dim=-1).abs().mean(0)
    P /= fields.shape[0]; P = P / P.max()
    k = torch.arange(P.shape[0], device=P.device)
    sup = k[P > thresh]
    return int(sup.max()) if sup.numel() else 2


def complex_transfer(M, u0, kmax):
    xg = x_row[None]                                  # (1, Nx)
    ck = torch.stack([torch.cos(2 * PI * kk * xg) for kk in range(1, kmax + 1)])  # (K,1,Nx)
    sk = torch.stack([torch.sin(2 * PI * kk * xg) for kk in range(1, kmax + 1)])
    eps = EPS_REL * _rms(u0)
    mag = torch.zeros(kmax); pha = torch.zeros(kmax)
    for j in range(kmax):
        d = ck[j].expand(Nt, Nx)[None]; d = d / d.norm()
        r = _jvp(M, u0, d, eps)[0]                    # (Nt, Nx)
        c = (r * torch.cos(2 * PI * (j + 1) * xg)).sum() / (torch.cos(2 * PI * (j + 1) * xg) ** 2).sum() / Nt
        s = (r * torch.sin(2 * PI * (j + 1) * xg)).sum() / (torch.sin(2 * PI * (j + 1) * xg) ** 2).sum() / Nt
        g = torch.complex(c, s)
        mag[j] = (g - 1).abs(); pha[j] = g.angle().abs()
    return torch.arange(1, kmax + 1).float(), mag, pha


def probe(M, base, gen):
    kmax = min(data_support(base), Nx // 4)
    kmax = max(kmax, 6)
    # complex transfer (mean + across-state variance)
    slopes_m, slopes_p, magspec = [], [], []
    for s in range(base.shape[0]):
        k, mag, pha = complex_transfer(M, base[s:s + 1], kmax)
        slopes_m.append(_slope(k, mag, 2, max(4, kmax)))
        slopes_p.append(_slope(k, pha, 2, max(4, kmax)))
        magspec.append(float(mag.sum()))
    magspec = torch.tensor(magspec)
    state_var = float(magspec.std() / magspec.mean().abs().clamp_min(1e-30))

    # magnitude on in-distribution directions (differences of prior fields)
    mags = []
    for s in range(base.shape[0]):
        u0 = base[s:s + 1]; eps = EPS_REL * _rms(u0)
        for _ in range(N_RAND):
            j = int(torch.randint(0, base.shape[0], (1,), generator=gen).item())
            dvec = base[j:j + 1] - u0
            if dvec.norm() < 1e-20:
                continue
            dvec = dvec / dvec.norm()
            mags.append(float((_jvp(M, u0, dvec, eps) - dvec).norm()))

    # heterogeneity + locality from localized bumps at several x-positions
    u0 = base[0:1]; eps = EPS_REL * _rms(u0)
    resp = []
    for x0 in torch.linspace(0.15, 0.85, 5):
        bump = torch.exp(-((x_row - x0) ** 2) / (2 * (0.03 ** 2)))[None].expand(Nt, Nx)[None]
        bump = bump / bump.norm()
        h = (_jvp(M, u0, bump, eps) - bump)[0].abs().mean(0)     # (Nx,) response profile
        resp.append(h)
    resp = torch.stack(resp)
    prof = resp.mean(0)
    ctr = int((torch.arange(Nx, device=DEVICE) * prof).sum() / prof.sum().clamp_min(1e-30))
    rad = float(((torch.arange(Nx, device=DEVICE).float() - ctr) ** 2 * prof).sum() / prof.sum().clamp_min(1e-30)).__pow__(0.5)
    # heterogeneity: how much each response deviates from the shifted mean shape
    shifts = torch.tensor([0.15, 0.325, 0.5, 0.675, 0.85])
    aligned = torch.stack([torch.roll(resp[i], shifts=int((0.5 - shifts[i]) * Nx)) for i in range(5)])
    het = float((aligned.std(0).sum() / aligned.mean(0).sum().clamp_min(1e-30)))

    return {
        "magnitude": float(torch.tensor(mags).mean()) if mags else float("nan"),
        "mag_slope": float(np.nanmean(slopes_m)),
        "phase_slope": float(np.nanmean(slopes_p)),
        "locality": rad,
        "heterogeneity": het,
        "state_var": state_var,
        "k_support": float(kmax),
    }


# ============================== main ==============================
def main():
    print(f"device {DEVICE}  grid {Nt}x{Nx}  ntr {ntr}", flush=True)
    rows = []
    for name in MISSPECS:
        prior = train_prior(name)
        G_tr, G_va, G_te = prior_full(prior, tr_v), prior_full(prior, va_v), prior_full(prior, te_v)
        prior_l2 = float((torch.linalg.norm((G_te - te_u).reshape(nte, -1), dim=1)
                          / torch.linalg.norm(te_u.reshape(nte, -1), dim=1).clamp_min(1e-30)).mean())

        cno = CNO2d(3, 1, (Nt, Nx), CNO_LAYERS, CNO_RES, CNO_RES_NECK, CNO_CH).to(DEVICE)
        Mfun = make_Mfun(cno)

        def step():
            idx = torch.randint(0, ntr, (CNO_BATCH,), device=DEVICE)
            return ((Mfun(G_tr[idx]) - tr_u[idx]) ** 2).mean()

        def val():
            with torch.no_grad():
                tot = 0.0
                for b in range(0, nval, CNO_BATCH):
                    e = min(b + CNO_BATCH, nval)
                    tot += ((Mfun(G_va[b:e]) - va_u[b:e]) ** 2).sum().item()
                return tot / (nval * Ng)

        fit(cno, step, val, CNO_EPOCHS, CNO_PATIENCE, REL_TOL)
        cno.eval()
        with torch.no_grad():
            fit_err = float((torch.linalg.norm((Mfun(G_te[:64]) - te_u[:64]).reshape(-1, Ng), dim=1)
                             / torch.linalg.norm(te_u[:64].reshape(-1, Ng), dim=1).clamp_min(1e-30)).mean())
            gen = torch.Generator(device="cpu").manual_seed(SEED)
            feats = probe(Mfun, G_te[:N_BASE].detach(), gen)

        feats["prior_l2"] = prior_l2; feats["cno_fit_err"] = fit_err
        print(f"[{name}] " + "  ".join(f"{k}={v:+.3f}" for k, v in feats.items()), flush=True)
        for k, v in feats.items():
            rows.append((name, k, v))
        del prior, cno, G_tr, G_va, G_te
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "pde2_cno_diag_metrics.txt")
    with open(path, "w") as f:
        f.write("misspec\tfeature\tvalue\n")
        for a, b, c in rows:
            f.write(f"{a}\t{b}\t{c:.6e}\n")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
