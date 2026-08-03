"""
Misspecification assay + filters-as-stencils, on an ADDITIVE CNO corrector.

For a panel of (PDE, injected-error) cells -- one error we WANT to detect (a local
diffusive error, ~u_xx) and one we do NOT (a control) -- we:

  1. train a physics prior on the misspecified residual   -> u_theta
  2. train an ADDITIVE CNO corrector  u_hat = u_theta + CNO(u_theta)
     (so CNO's output is exactly the correction field u_hat - u_theta)
  3. ASSAY: regress the correction against the discrete curvature u_xx of u_theta,
     report R^2 (the diffusive-error test statistic) and cross-reactivity R^2 against
     {u_x, u, u u_x, u_xxx} (specificity: the assay must fire on u_xx only)
  4. FILTERS-AS-STENCILS: project the CNO's first-layer conv filters (acting on the
     u_theta channel) onto discrete differential-operator stencils {I, d_x, d_t, d_xx,
     d_tt, laplacian, d_xt} -- does the network build a Laplacian internally?

Cells here span P2 (Burgers) and P3 (advection-diffusion), each with a diffusive
positive and a non-diffusive control. Extending to P1/P4/P5 is a config + residual add
(see NOTE at bottom).  Data: Hugging Face zip of .npy files.

Env: MM_SMOKE=1 for a tiny CPU dry-run (synthetic data, no download).
"""

import os
import math
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

HF_REPO = "alexanderthegreat69420/Model_misspecification"
CACHE_ROOT = os.environ.get("MM_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mm_data"))
OUT_DIR = os.environ.get("MM_OUT_DIR", "/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
PI = math.pi
SMOKE = bool(int(os.environ.get("MM_SMOKE", "0")))

# ---------- panel: (pde, hf_zip, level, gt_diffusive) ----------
CELLS = [
    ("pde2", "pde2_burgers.zip", "low", 1),   # wrong viscosity  -> diffusive (target)
    ("pde2", "pde2_burgers.zip", "max", 0),   # KdV dispersion   -> control
    ("pde3", "pde4_advection_diffusion.zip", "diff", 1),  # wrong viscosity -> diffusive
    ("pde3", "pde4_advection_diffusion.zip", "low", 0),   # wrong speed     -> control
]

# ---------- hyperparameters ----------
M_FOUR = 16
PRIOR_BH, PRIOR_P, PRIOR_TH = [256, 256], 128, [128, 128, 128]
LR = 1e-3
PHYS_EPOCHS, PHYS_PATIENCE, PHYS_STEPS, PHYS_BATCH, N_COLLOC, W_IC = 800, 50, 16, 32, 3000, 20.0
CNO_LAYERS, CNO_CH, CNO_RES, CNO_RES_NECK = 3, 32, 4, 4
CNO_EPOCHS, CNO_PATIENCE, CNO_BATCH, CNO_STEPS = 400, 40, 5, 16
ASSAY_N = 300
if SMOKE:
    CELLS = [("pde2", "", "low", 1), ("pde2", "", "max", 0)]
    M_FOUR = 6
    PHYS_EPOCHS, PHYS_PATIENCE, PHYS_STEPS, N_COLLOC = 3, 3, 2, 200
    CNO_EPOCHS, CNO_PATIENCE, CNO_STEPS, CNO_LAYERS, CNO_CH = 3, 3, 2, 2, 8
    ASSAY_N = 8

NU_P2, DELTA_P2 = 0.01, 1e-3
C_P3, NU_P3 = 1.0, 1e-3


# ======================= data =======================
def prepare_data(zip_name):
    ext = os.path.join(CACHE_ROOT, zip_name[:-4])
    if not os.path.isdir(ext):
        os.makedirs(ext, exist_ok=True)
        z = hf_hub_download(repo_id=HF_REPO, filename=zip_name, repo_type="dataset")
        with zipfile.ZipFile(z) as zz:
            zz.extractall(ext)
    for root, _, files in os.walk(ext):
        if "x.npy" in files:
            return root
    raise FileNotFoundError(zip_name)


def load_pde(hf_zip):
    if SMOKE:
        rng = np.random.default_rng(0); Nx, Nt = 48, 32
        x = np.linspace(0, 1, Nx, endpoint=False).astype(np.float32)
        t = np.linspace(0, 0.5, Nt).astype(np.float32)

        def syn(n):
            v = rng.standard_normal((n, Nx)).astype(np.float32)
            vf = np.fft.rfft(v, axis=-1); vf[:, 5:] = 0
            v = np.fft.irfft(vf, n=Nx, axis=-1).astype(np.float32)
            u = np.repeat(v[:, None], Nt, 1) + 0.05 * rng.standard_normal((n, Nt, Nx)).astype(np.float32)
            u[:, 0] = v
            return v, u
        d = dict(x=x, t=t)
        for s, n in (("train", 40), ("val", 8), ("test", 40)):
            d[s + "_v"], d[s + "_u"] = syn(n)
    else:
        D = prepare_data(hf_zip)
        ld = lambda nm: np.load(os.path.join(D, nm)).astype(np.float32)
        d = {k: ld(k + ".npy") for k in ["x", "t", "train_v", "train_u", "val_v", "val_u", "test_v", "test_u"]}
    return d


# ======================= physics prior =======================
class MLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        L = []
        for i in range(len(sizes) - 1):
            L.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                L.append(nn.Tanh())
        self.net = nn.Sequential(*L)

    def forward(self, x): return self.net(x)


class DeepONetST(nn.Module):
    def __init__(self, branch_in, feat, ms, t_end):
        super().__init__()
        self.branch = MLP([branch_in] + PRIOR_BH + [PRIOR_P])
        self.trunk = MLP([feat] + PRIOR_TH + [PRIOR_P])
        self.b0 = nn.Parameter(torch.zeros(1)); self.ms = ms; self.t_end = t_end

    def feats(self, c):
        x = c[..., 0:1]; t = c[..., 1:2]; ang = 2 * PI * x * self.ms
        return torch.cat([torch.sin(ang), torch.cos(ang), t / self.t_end], dim=-1)

    def eval_grid(self, vin, coords):
        return self.branch(vin) @ self.trunk(self.feats(coords)).t() + self.b0

    def pointwise(self, vin, coords):
        return torch.einsum("bp,bnp->bn", self.branch(vin), self.trunk(self.feats(coords))) + self.b0


def residual(pde, name, u, ux, uxx, uxxx, ut):
    if pde == "pde2":
        if name == "low":
            return ut + u * ux - 0.013 * uxx          # wrong viscosity  (DIFFUSIVE)
        return ut + u * ux + DELTA_P2 * uxxx           # KdV              (control)
    if name == "diff":
        return ut + C_P3 * ux - 2e-3 * uxx             # wrong viscosity  (DIFFUSIVE)
    return ut + 1.05 * ux - NU_P3 * uxx                # wrong speed      (control)


def derivs(model, vin, coords):
    u = model.pointwise(vin, coords)
    g1 = torch.autograd.grad(u.sum(), coords, create_graph=True)[0]; ux, ut = g1[..., 0], g1[..., 1]
    uxx = torch.autograd.grad(ux.sum(), coords, create_graph=True)[0][..., 0]
    uxxx = torch.autograd.grad(uxx.sum(), coords, create_graph=True)[0][..., 0]
    return u, ux, uxx, uxxx, ut


def clone(m): return {k: v.detach().clone() for k, v in m.state_dict().items()}


def fit(model, step, val, epochs, patience, steps, rel_tol):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, state, wait = float("inf"), clone(model), 0
    for _ in range(epochs):
        model.train()
        for _ in range(steps):
            opt.zero_grad(); loss = step(); loss.backward(); opt.step()
        model.eval(); vl = val()
        if vl < best * (1 - rel_tol):
            best, state, wait = vl, clone(model), 0
        else:
            wait += 1
        if wait >= patience:
            break
    model.load_state_dict(state); return model


# ======================= CNO (rectangular, additive) =======================
class CNO_LReLu(nn.Module):
    def __init__(self, i, o): super().__init__(); self.i, self.o = i, o; self.act = nn.LeakyReLU()

    def forward(self, x):
        x = F.interpolate(x, size=(2 * self.i[0], 2 * self.i[1]), mode="bicubic", antialias=True)
        return F.interpolate(self.act(x), size=(self.o[0], self.o[1]), mode="bicubic", antialias=True)


class CNOBlock(nn.Module):
    def __init__(self, ic, oc, i, o, bn=True):
        super().__init__(); self.conv = nn.Conv2d(ic, oc, 3, padding=1)
        self.bn = nn.BatchNorm2d(oc) if bn else nn.Identity(); self.act = CNO_LReLu(i, o)

    def forward(self, x): return self.act(self.bn(self.conv(x)))


class LiftProject(nn.Module):
    def __init__(self, ic, oc, size, latent=64):
        super().__init__(); self.b = CNOBlock(ic, latent, size, size, bn=False)
        self.conv = nn.Conv2d(latent, oc, 3, padding=1)

    def forward(self, x): return self.conv(self.b(x))


class ResidualBlock(nn.Module):
    def __init__(self, ch, size, bn=True):
        super().__init__(); self.c1 = nn.Conv2d(ch, ch, 3, padding=1); self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b1 = nn.BatchNorm2d(ch) if bn else nn.Identity(); self.b2 = nn.BatchNorm2d(ch) if bn else nn.Identity()
        self.act = CNO_LReLu(size, size)

    def forward(self, x): return x + self.b2(self.c2(self.act(self.b1(self.c1(x)))))


class ResNet(nn.Module):
    def __init__(self, ch, size, n, bn=True):
        super().__init__(); self.net = nn.Sequential(*[ResidualBlock(ch, size, bn) for _ in range(n)])

    def forward(self, x): return self.net(x)


class CNO2d(nn.Module):
    def __init__(self, in_dim, out_dim, size, N, N_res=4, N_neck=4, cm=32):
        super().__init__(); self.N = N; lift = cm // 2
        ef = [lift] + [2 ** i * cm for i in range(N)]
        di = ef[1:][::-1]; do = ef[:-1][::-1]
        for i in range(1, N):
            di[i] = 2 * di[i]

        def sz(i): return (size[0] // 2 ** i, size[1] // 2 ** i)
        es = [sz(i) for i in range(N + 1)]; ds = [sz(N - i) for i in range(N + 1)]
        self.lift = LiftProject(in_dim, ef[0], size)
        self.project = LiftProject(ef[0] + do[-1], out_dim, size)
        self.encoder = nn.ModuleList([CNOBlock(ef[i], ef[i + 1], es[i], es[i + 1]) for i in range(N)])
        self.ED = nn.ModuleList([CNOBlock(ef[i], ef[i], es[i], ds[N - i]) for i in range(N + 1)])
        self.decoder = nn.ModuleList([CNOBlock(di[i], do[i], ds[i], ds[i + 1]) for i in range(N)])
        self.res = nn.ModuleList([ResNet(ef[l], es[l], N_res) for l in range(N)])
        self.neck = ResNet(ef[N], es[N], N_neck)

    def forward(self, x):
        x = self.lift(x); skip = []
        for i in range(self.N):
            skip.append(self.res[i](x)); x = self.encoder[i](x)
        x = self.neck(x)
        for i in range(self.N):
            x = self.ED[self.N - i](x) if i == 0 else torch.cat((x, self.ED[self.N - i](skip[-i])), 1)
            x = self.decoder[i](x)
        return self.project(torch.cat((x, self.ED[0](skip[0])), 1))


# ======================= assay + stencils =======================
def dx(f):  return (torch.roll(f, -1, -1) - torch.roll(f, 1, -1)) / 2.0
def dxx(f): return torch.roll(f, -1, -1) - 2 * f + torch.roll(f, 1, -1)
def dxxx(f):return (torch.roll(f, -2, -1) - 2 * torch.roll(f, -1, -1) + 2 * torch.roll(f, 1, -1) - torch.roll(f, 2, -1)) / 2.0


def r2(y, f):
    y = y - y.mean(); f = f - f.mean()
    vf = (f * f).mean()
    if float(vf) < 1e-20:
        return 0.0, 0.0
    c = float((y * f).mean() / vf)
    r = float((y * f).mean() / (y.std() * f.std() + 1e-20))
    return r * r, c


def stencil_report(cno):
    w = cno.lift.b.conv.weight.detach()          # (latent, in=3, 3, 3)
    f0 = w[:, 0].reshape(w.shape[0], 9)           # filters on the u_theta channel
    S = {
        "I":   [0, 0, 0, 0, 1, 0, 0, 0, 0],
        "d_x": [0, 0, 0, -.5, 0, .5, 0, 0, 0],
        "d_t": [0, -.5, 0, 0, 0, 0, 0, .5, 0],
        "d_xx": [0, 0, 0, 1, -2, 1, 0, 0, 0],
        "d_tt": [0, 1, 0, 0, -2, 0, 0, 1, 0],
        "lap": [0, 1, 0, 1, -4, 1, 0, 1, 0],
        "d_xt": [.25, 0, -.25, 0, 0, 0, -.25, 0, .25],
    }
    B = torch.tensor([S[k] for k in S], dtype=f0.dtype, device=f0.device)
    Q, _ = torch.linalg.qr(B.t())                 # orthonormal basis (9, k)
    fn = f0 / f0.norm(dim=1, keepdim=True).clamp_min(1e-20)
    coeff = (fn @ Q).abs().mean(0)                # mean |projection| per stencil
    return {k: float(coeff[i]) for i, k in enumerate(S)}


# ======================= per-cell pipeline =======================
def run_cell(pde, hf_zip, level, gt):
    d = load_pde(hf_zip)
    x_np, t_np = d["x"], d["t"]; Nx, Nt = x_np.shape[0], t_np.shape[0]; Ng = Nt * Nx; T_END = float(t_np.max())
    to = lambda a: torch.tensor(a, device=DEVICE)
    tr_v, tr_u = to(d["train_v"]), to(d["train_u"]); va_v, va_u = to(d["val_v"]), to(d["val_u"]); te_v, te_u = to(d["test_v"]), to(d["test_u"])
    ntr, nval, nte = tr_v.shape[0], va_v.shape[0], te_v.shape[0]
    V_MEAN, V_STD = float(tr_v.mean()), float(tr_v.std()); U_MEAN, U_STD = float(tr_u.mean()), float(tr_u.std())
    vnorm = lambda v: (v - V_MEAN) / V_STD; gnorm = lambda g: (g - U_MEAN) / U_STD
    x_row, t_row = to(x_np), to(t_np); TT, XX = torch.meshgrid(t_row, x_row, indexing="ij")
    grid_full = torch.stack([XX.reshape(-1), TT.reshape(-1)], -1)
    ic = torch.stack([x_row, torch.zeros_like(x_row)], -1)
    ms = torch.arange(1, M_FOUR + 1, device=DEVICE).float(); FEAT = 2 * M_FOUR + 1
    X2D, T2D = XX, TT
    VAL_COLL = torch.rand(PHYS_BATCH, N_COLLOC, 2, device=DEVICE); VAL_COLL[..., 1] *= T_END

    # ---- prior ----
    prior = DeepONetST(Nx, FEAT, ms, T_END).to(DEVICE)

    def pstep():
        idx = torch.randint(0, ntr, (PHYS_BATCH,), device=DEVICE)
        coll = torch.rand(PHYS_BATCH, N_COLLOC, 2, device=DEVICE); coll[..., 1] *= T_END; coll = coll.requires_grad_(True)
        vin = vnorm(tr_v[idx]); u, ux, uxx, uxxx, ut = derivs(prior, vin, coll)
        icl = ((prior.eval_grid(vin, ic) - tr_v[idx]) ** 2).mean()
        return (residual(pde, level, u, ux, uxx, uxxx, ut) ** 2).mean() + W_IC * icl

    def pval():
        idx = torch.arange(min(PHYS_BATCH, nval), device=DEVICE)
        coll = VAL_COLL[:idx.shape[0]].clone().requires_grad_(True); vin = vnorm(va_v[idx])
        u, ux, uxx, uxxx, ut = derivs(prior, vin, coll)
        icl = ((prior.eval_grid(vin, ic) - va_v[idx]) ** 2).mean()
        return ((residual(pde, level, u, ux, uxx, uxxx, ut) ** 2).mean() + W_IC * icl).item()

    fit(prior, pstep, pval, PHYS_EPOCHS, PHYS_PATIENCE, PHYS_STEPS, 1e-4)
    for p in prior.parameters():
        p.requires_grad_(False)
    prior.eval()

    @torch.no_grad()
    def prior_full(v):
        n = v.shape[0]; out = torch.empty(n, Ng, device=DEVICE)
        for b in range(0, n, 64):
            e = min(b + 64, n); vin = vnorm(v[b:e])
            for c in range(0, Ng, 8192):
                out[b:e, c:c + 8192] = prior.eval_grid(vin, grid_full[c:c + 8192])
        return out.reshape(n, Nt, Nx)

    G_tr, G_va, G_te = prior_full(tr_v), prior_full(va_v), prior_full(te_v)
    prior_l2 = float((torch.linalg.norm((G_te - te_u).reshape(nte, -1), dim=1)
                      / torch.linalg.norm(te_u.reshape(nte, -1), dim=1).clamp_min(1e-30)).mean())

    # ---- ADDITIVE CNO corrector: u_hat = u_theta + CNO([gnorm(u_theta), X, T]) ----
    cno = CNO2d(3, 1, (Nt, Nx), CNO_LAYERS, CNO_RES, CNO_RES_NECK, CNO_CH).to(DEVICE)
    xb, tb = X2D[None, None], T2D[None, None]

    def corr(uth):                         # CNO's correction field for a batch of prior fields
        B = uth.shape[0]
        inp = torch.cat([gnorm(uth)[:, None], xb.expand(B, 1, Nt, Nx), tb.expand(B, 1, Nt, Nx)], 1)
        return cno(inp).squeeze(1)

    def cstep():
        idx = torch.randint(0, ntr, (CNO_BATCH,), device=DEVICE)
        return ((G_tr[idx] + corr(G_tr[idx]) - tr_u[idx]) ** 2).mean()

    def cval():
        with torch.no_grad():
            tot = 0.0
            for b in range(0, nval, CNO_BATCH):
                e = min(b + CNO_BATCH, nval)
                tot += ((G_va[b:e] + corr(G_va[b:e]) - va_u[b:e]) ** 2).sum().item()
            return tot / (nval * Ng)

    fit(cno, cstep, cval, CNO_EPOCHS, CNO_PATIENCE, CNO_STEPS, 5e-3)
    cno.eval()

    # ---- assay: regress correction (CNO output) vs candidate operators on u_theta ----
    with torch.no_grad():
        m = min(ASSAY_N, nte)
        cf = torch.cat([corr(G_te[b:min(b + CNO_BATCH, m)]) for b in range(0, m, CNO_BATCH)], 0)  # (m,Nt,Nx)
        g = G_te[:m]
        y = cf.reshape(-1)
        cand = {"u_xx": dxx(g), "u_x": dx(g), "u": g, "u_ux": g * dx(g), "u_xxx": dxxx(g)}
        assay = {}
        for k, f in cand.items():
            R2, c = r2(y, f.reshape(-1)); assay["R2_" + k] = R2
            if k == "u_xx":
                assay["c_uxx"] = c
        fit_err = float((torch.linalg.norm((g + cf - te_u[:m]).reshape(m, -1), dim=1)
                         / torch.linalg.norm(te_u[:m].reshape(m, -1), dim=1).clamp_min(1e-30)).mean())

    sten = stencil_report(cno)
    out = {"prior_l2": prior_l2, "cno_fit_err": fit_err, **assay, **{"stencil_" + k: v for k, v in sten.items()}}
    del prior, cno, G_tr, G_va, G_te
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return out


def main():
    print(f"device {DEVICE}  smoke={SMOKE}", flush=True)
    rows = []
    for pde, zip_, level, gt in CELLS:
        r = run_cell(pde, zip_, level, gt)
        r["gt_diffusive"] = gt
        tag = f"{pde}/{level}"
        print(f"[{tag}] gt={gt}  R2_uxx={r['R2_u_xx']:.3f}  R2_ux={r['R2_u_x']:.3f}  "
              f"fit_err={r['cno_fit_err']:.3f}  lap_stencil={r['stencil_lap']:.3f}", flush=True)
        for k, v in r.items():
            rows.append((tag, gt, k, v))
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "assay_metrics.txt")
    with open(path, "w") as f:
        f.write("cell\tgt_diffusive\tmetric\tvalue\n")
        for a, b, c, v in rows:
            f.write(f"{a}\t{b}\t{c}\t{v:.6e}\n")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()

# NOTE: to reach 5 PDEs x 2 errors, add cells + a diffusive/control residual per PDE:
#   P1 (steady 1D): diffusive = wrong D (0.11 vs 0.1); control = linearized reaction  (needs 1D prior)
#   P5 (2D Helmholtz): diffusive = wrong Laplacian coeff a*(-Lap); control = wrong kappa^2 (needs 2D-spatial prior)
#   P4 (fractional): stress case -- true operator is nonlocal, no natural local-diffusive positive.
