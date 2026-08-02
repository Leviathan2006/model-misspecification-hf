import os, math, zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from probe import probe_operator
from huggingface_hub import hf_hub_download

HF_REPO = "alexanderthegreat69420/Model_misspecification"
HF_ZIP = "pde6_helmholtz2d.zip"
CACHE_ROOT = os.environ.get("MM_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mm_data"))


def prepare_data(zip_name):
    extract_dir = os.path.join(CACHE_ROOT, zip_name[:-4])
    if not os.path.isdir(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        zpath = hf_hub_download(repo_id=HF_REPO, filename=zip_name, repo_type="dataset")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(extract_dir)
    for root, _, files in os.walk(extract_dir):
        if "x.npy" in files:
            return root
    raise FileNotFoundError(f"x.npy not found under {extract_dir}")


DATA_DIR = prepare_data(HF_ZIP)
OUT_DIR = os.environ.get("MM_OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))

KAPPA2 = 3765.25
MISSPECS = ["low", "med", "high", "max"]

RFF_SCALES = [6.0, 12.0, 20.0]
RFF_K = 64
P_THETA = 128
TRUNK_THETA = [256, 256, 256]
SMALL_P = 64
SMALL_TH = [128, 128]
FW, FM, FL = 32, (24, 24), 4

LR = 1e-3
REL_TOL = 5e-3
PRIOR_REL_TOL = 1e-4
STEPS_PER_EPOCH = 8
PHYS_BATCH = 8
N_COLLOC = 2500
DON_BATCH = 16
N_PTS = 4096
FNO_BATCH = 8
PHYS_EPOCHS = 800
PHYS_PATIENCE = 50
DATA_EPOCHS = 200
DATA_PATIENCE = 20
CORR_BATCH = 32
EVAL_CHUNK = 8192
VAL_PTS = 8000
SEED = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED)
PI = math.pi


def load_split(name):
    v = np.load(os.path.join(DATA_DIR, name + "_v.npy")).astype(np.float32)
    u = np.load(os.path.join(DATA_DIR, name + "_u.npy")).astype(np.float32)
    return v, u


x_np = np.load(os.path.join(DATA_DIR, "x.npy")).astype(np.float32)
y_np = np.load(os.path.join(DATA_DIR, "y.npy")).astype(np.float32)
N = x_np.shape[0]
Ng = N * N
tr_v_n, tr_u_n = load_split("train")
va_v_n, va_u_n = load_split("val")
te_v_n, te_u_n = load_split("test")

V_MEAN, V_STD = float(tr_v_n.mean()), float(tr_v_n.std())
US = float(tr_u_n.std())

to = lambda a: torch.tensor(a, device=DEVICE)
tr_v, tr_u = to(tr_v_n), to(tr_u_n)
va_v, va_u = to(va_v_n), to(va_u_n)
te_v, te_u = to(te_v_n), to(te_u_n)
ntr, nval, nte = tr_v.shape[0], va_v.shape[0], te_v.shape[0]
tr_vi = tr_v[:, None]; va_vi = va_v[:, None]; te_vi = te_v[:, None]
tr_vf = tr_v.reshape(ntr, Ng); va_vf = va_v.reshape(nval, Ng); te_vf = te_v.reshape(nte, Ng)
tr_uf = tr_u.reshape(ntr, Ng); va_uf = va_u.reshape(nval, Ng); te_uf = te_u.reshape(nte, Ng)

xg = to(x_np); yg = to(y_np)
XX, YY = torch.meshgrid(xg, yg, indexing="ij")
grid_full = torch.stack([XX.reshape(-1), YY.reshape(-1)], dim=-1)
X2D = XX; Y2D = YY
BUMP2D = XX * (1 - XX) * YY * (1 - YY)
VAL_FIDX = torch.randint(0, Ng, (VAL_PTS,), device=DEVICE)

_cols = [torch.randn(2, RFF_K, device=DEVICE) * s for s in RFF_SCALES]
BMAT = torch.cat(_cols, dim=1)
RFF_DIM = 2 * BMAT.shape[1]


def vnorm(v): return (v - V_MEAN) / V_STD
def gnorm(g): return g / US


def rff(coords):
    proj = 2.0 * PI * (coords @ BMAT)
    return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


def bump(coords):
    x = coords[..., 0]; y = coords[..., 1]
    return x * (1 - x) * y * (1 - y)


class MLP(nn.Module):
    def __init__(self, sizes, act=nn.Tanh):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNNBranch(nn.Module):
    def __init__(self, P):
        super().__init__()
        chs = [1, 16, 32, 64, 64]
        blocks = []
        for i in range(4):
            blocks += [nn.Conv2d(chs[i], chs[i + 1], 3, stride=2, padding=1), nn.GELU()]
        self.conv = nn.Sequential(*blocks)
        red = N
        for _ in range(4):
            red = (red + 1) // 2
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * red * red, 256), nn.GELU(), nn.Linear(256, P))

    def forward(self, vimg):
        return self.fc(self.conv(vimg))


class DeepONet2D(nn.Module):
    def __init__(self, extra, P, th):
        super().__init__()
        self.branch = CNNBranch(P)
        self.trunk = MLP([RFF_DIM + extra] + th + [P])
        self.b0 = nn.Parameter(torch.zeros(1))
        self.extra = extra

    def code(self, vimg):
        return self.branch(vnorm(vimg))

    def eval_grid(self, b, coords, gextra=None):
        if self.extra == 0:
            raw = b @ self.trunk(rff(coords)).t() + self.b0
            return raw * bump(coords)
        f = rff(coords).unsqueeze(0).expand(b.shape[0], -1, -1)
        f = torch.cat([f, gextra.unsqueeze(-1)], dim=-1)
        raw = torch.einsum("bp,bnp->bn", b, self.trunk(f)) + self.b0
        return raw * bump(coords)

    def pointwise(self, b, coords):
        raw = torch.einsum("bp,bnp->bn", b, self.trunk(rff(coords))) + self.b0
        return raw * bump(coords)


class SpectralConv2d(nn.Module):
    def __init__(self, ic, oc, m1, m2):
        super().__init__()
        self.m1, self.m2 = m1, m2
        s = 1.0 / (ic * oc)
        self.w1 = nn.Parameter(s * torch.rand(ic, oc, m1, m2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(s * torch.rand(ic, oc, m1, m2, dtype=torch.cfloat))

    def forward(self, x):
        B, H, W = x.shape[0], x.shape[-2], x.shape[-1]
        xft = torch.fft.rfft2(x, dim=(-2, -1))
        Wf = xft.shape[-1]
        m1 = min(self.m1, H // 2); m2 = min(self.m2, Wf)
        out = torch.zeros(B, self.w1.shape[1], H, Wf, dtype=torch.cfloat, device=x.device)
        out[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", xft[:, :, :m1, :m2], self.w1[:, :, :m1, :m2])
        out[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", xft[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2])
        return torch.fft.irfft2(out, s=(H, W), dim=(-2, -1))


class FNO2d(nn.Module):
    def __init__(self, ic, width, modes, layers):
        super().__init__()
        self.fc0 = nn.Linear(ic, width)
        self.sp = nn.ModuleList([SpectralConv2d(width, width, modes[0], modes[1]) for _ in range(layers)])
        self.w = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc0(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        for s, w in zip(self.sp, self.w):
            x = F.gelu(s(x) + w(x))
        x = F.gelu(self.fc1(x.permute(0, 2, 3, 1)))
        return self.fc2(x).squeeze(-1)


def residual(name, u, ux, uy, lap, v, x0, y0):
    if name == "low":
        return -lap - 3780.0 * u - v
    if name == "med":
        return -lap - 3805.0 * u - v
    if name == "high":
        a = 1.0 + 0.2 * torch.cos(2 * PI * x0) * torch.cos(2 * PI * y0)
        ax = -0.4 * PI * torch.sin(2 * PI * x0) * torch.cos(2 * PI * y0)
        ay = -0.4 * PI * torch.cos(2 * PI * x0) * torch.sin(2 * PI * y0)
        return -(a * lap + ax * ux + ay * uy) - KAPPA2 * u - v
    r = torch.sqrt((x0 - 0.5) ** 2 + (y0 - 0.5) ** 2 + 1e-12)
    s = (0.25 - r) / 0.02
    th = torch.tanh(s); sech2 = 1.0 - th * th
    a = 1.0 + 1.5 * (1.0 + th)
    ax = -(1.5 / 0.02) * sech2 * (x0 - 0.5) / r
    ay = -(1.5 / 0.02) * sech2 * (y0 - 0.5) / r
    return -(a * lap + ax * ux + ay * uy) - KAPPA2 * u - v


def derivs(prior, b, coords):
    u = prior.pointwise(b, coords)
    g1 = torch.autograd.grad(u.sum(), coords, create_graph=True)[0]
    ux, uy = g1[..., 0], g1[..., 1]
    uxx = torch.autograd.grad(ux.sum(), coords, create_graph=True)[0][..., 0]
    uyy = torch.autograd.grad(uy.sum(), coords, create_graph=True)[0][..., 1]
    return u, ux, uy, uxx + uyy


def rel_l2(pred, u):
    return (torch.linalg.norm(pred - u, dim=1) / (torch.linalg.norm(u, dim=1) + 1e-30)).mean().item()


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def clone(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def fit(model, step, val, epochs, patience, tag, rel_tol=REL_TOL):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, state, wait = float("inf"), None, 0
    for ep in range(epochs):
        model.train(); tl = 0.0
        for _ in range(STEPS_PER_EPOCH):
            opt.zero_grad(); loss = step(); loss.backward(); opt.step(); tl += loss.item()
        model.eval(); vl = val()
        if vl < best * (1 - rel_tol):
            best, state, wait = vl, clone(model), 0
        else:
            wait += 1
        if wait >= patience:
            break
    model.load_state_dict(state)
    return model


def channels_pure(vi):
    B = vi.shape[0]
    return torch.stack([vnorm(vi[:, 0]), X2D[None].expand(B, N, N), Y2D[None].expand(B, N, N)], dim=1)


def channels_comb(vi, g):
    B = vi.shape[0]
    return torch.stack([gnorm(g), vnorm(vi[:, 0]), X2D[None].expand(B, N, N), Y2D[None].expand(B, N, N)], dim=1)


@torch.no_grad()
def fno_pure_full(model, vi):
    n = vi.shape[0]; out = torch.empty(n, N, N, device=DEVICE)
    for b in range(0, n, FNO_BATCH):
        e = min(b + FNO_BATCH, n); out[b:e] = US * (model(channels_pure(vi[b:e])) * BUMP2D)
    return out


@torch.no_grad()
def fno_comb_full(model, vi, g_field):
    n = vi.shape[0]; out = torch.empty(n, N, N, device=DEVICE)
    for b in range(0, n, FNO_BATCH):
        e = min(b + FNO_BATCH, n); out[b:e] = g_field[b:e] + US * (model(channels_comb(vi[b:e], g_field[b:e])) * BUMP2D)
    return out


def don_pure_full(model, vi):
    n = vi.shape[0]; out = torch.empty(n, Ng, device=DEVICE)
    for b in range(0, n, CORR_BATCH):
        e = min(b + CORR_BATCH, n); code = model.code(vi[b:e])
        for c in range(0, Ng, EVAL_CHUNK):
            out[b:e, c:c + EVAL_CHUNK] = US * model.eval_grid(code, grid_full[c:c + EVAL_CHUNK])
    return out.reshape(n, N, N)


def don_comb_full(model, vi, g_field):
    n = vi.shape[0]; g_flat = g_field.reshape(n, Ng); out = torch.empty(n, Ng, device=DEVICE)
    for b in range(0, n, CORR_BATCH):
        e = min(b + CORR_BATCH, n); code = model.code(vi[b:e])
        for c in range(0, Ng, EVAL_CHUNK):
            cc = grid_full[c:c + EVAL_CHUNK]; gpts = g_flat[b:e, c:c + EVAL_CHUNK]
            out[b:e, c:c + EVAL_CHUNK] = gpts + US * model.eval_grid(code, cc, gnorm(gpts))
    return out.reshape(n, N, N)


@torch.no_grad()
def prior_full(prior, vi):
    n = vi.shape[0]; out = torch.empty(n, Ng, device=DEVICE)
    for b in range(0, n, CORR_BATCH):
        e = min(b + CORR_BATCH, n); code = prior.code(vi[b:e])
        for c in range(0, Ng, EVAL_CHUNK):
            out[b:e, c:c + EVAL_CHUNK] = prior.eval_grid(code, grid_full[c:c + EVAL_CHUNK])
    return out.reshape(n, N, N)


def train_prior(name):
    prior = DeepONet2D(0, P_THETA, TRUNK_THETA).to(DEVICE)
    vfix = torch.randint(0, Ng, (N_COLLOC,), device=DEVICE)

    def ploss(vi, vf, idx, fidx):
        b = prior.code(vi[idx])
        coords = grid_full[fidx].view(1, -1, 2).expand(idx.shape[0], -1, 2).clone().requires_grad_(True)
        u, ux, uy, lap = derivs(prior, b, coords)
        return (residual(name, u, ux, uy, lap, vf[idx][:, fidx], coords[..., 0], coords[..., 1]) ** 2).mean()

    def step():
        idx = torch.randint(0, ntr, (PHYS_BATCH,), device=DEVICE)
        return ploss(tr_vi, tr_vf, idx, torch.randint(0, Ng, (N_COLLOC,), device=DEVICE))

    def val():
        idx = torch.arange(min(PHYS_BATCH, nval), device=DEVICE)
        return ploss(va_vi, va_vf, idx, vfix).item()

    fit(prior, step, val, PHYS_EPOCHS, PHYS_PATIENCE, f"{name} prior", rel_tol=PRIOR_REL_TOL)
    for p in prior.parameters():
        p.requires_grad_(False)
    prior.eval()
    return prior


def phys_residual(prior, name):
    tot, cnt = 0.0, 0
    for b in range(0, nte, PHYS_BATCH):
        e = min(b + PHYS_BATCH, nte); code = prior.code(te_vi[b:e])
        fidx = torch.randint(0, Ng, (N_COLLOC,), device=DEVICE)
        coords = grid_full[fidx].view(1, -1, 2).expand(e - b, -1, 2).clone().requires_grad_(True)
        u, ux, uy, lap = derivs(prior, code, coords)
        r = residual(name, u, ux, uy, lap, te_vf[b:e][:, fidx], coords[..., 0], coords[..., 1])
        tot += (r ** 2).sum().item(); cnt += r.numel()
    return tot / cnt


MW, MM, ML = 64, (32, 32), 4


def Mchan(u):
    B = u.shape[0]
    return torch.stack([gnorm(u), X2D[None].expand(B, N, N), Y2D[None].expand(B, N, N)], dim=1)


metric_rows = []
transfers = {}

for name in MISSPECS:
    prior = train_prior(name)
    G_tr = prior_full(prior, tr_vi); G_va = prior_full(prior, va_vi); G_te = prior_full(prior, te_vi)
    Mpsi = FNO2d(3, MW, MM, ML).to(DEVICE)
    Mfun = lambda u: US * (Mpsi(Mchan(u)) * BUMP2D)

    def s():
        idx = torch.randint(0, ntr, (FNO_BATCH,), device=DEVICE)
        return (((Mfun(G_tr[idx]) - tr_u[idx]) / US) ** 2).mean()

    def v():
        with torch.no_grad():
            tot = 0.0
            for b in range(0, nval, FNO_BATCH):
                e = min(b + FNO_BATCH, nval)
                tot += (((Mfun(G_va[b:e]) - va_u[b:e]) / US) ** 2).sum().item()
            return tot / (nval * N * N)

    fit(Mpsi, s, v, DATA_EPOCHS, DATA_PATIENCE, "")
    Mpsi.eval()
    base = G_te[:8].detach()
    with torch.no_grad():
        feats = probe_operator(Mfun, base, n_rand=3)
    for key in ("magnitude", "exponent", "locality", "equivariance", "state_var"):
        metric_rows.append((name, key, feats[key]))
    transfers[f"{name}_k"] = feats["k"]; transfers[f"{name}_T"] = feats["transfer"]
    del prior, Mpsi, G_tr, G_va, G_te; torch.cuda.empty_cache()

os.makedirs(OUT_DIR, exist_ok=True)
mpath = os.path.join(OUT_DIR, "pde5_diag_metrics.txt")
with open(mpath, "w") as f:
    f.write("misspec\tfeature\tvalue\n")
    for a, b, c in metric_rows:
        f.write(f"{a}\t{b}\t{c:.6e}\n")
tpath = os.path.join(OUT_DIR, "pde5_diag_transfer.npz")
np.savez_compressed(tpath, **transfers)
print(f"wrote {mpath} and {tpath}", flush=True)
