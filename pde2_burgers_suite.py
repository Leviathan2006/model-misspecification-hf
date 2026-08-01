import os, math, zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

HF_REPO = "alexanderthegreat69420/Model_misspecification"
HF_ZIP = "pde2_burgers.zip"
CACHE_ROOT = os.environ.get("MM_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mm_data"))


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

NU_TRUE = 0.01
DELTA = 1e-3
MISSPECS = ["low", "med", "high", "max"]

M_FOUR = 16
PRIOR_BH = [256, 256]
PRIOR_P = 128
PRIOR_TH = [128, 128, 128]
SMALL_BH = [128, 128]
SMALL_P = 64
SMALL_TH = [64, 64, 64]
FW, FM, FL = 12, (8, 8), 2

LR = 1e-3
REL_TOL = 5e-3
STEPS_PER_EPOCH = 16
PHYS_BATCH = 32
N_COLLOC = 3000
DON_BATCH = 64
N_PTS = 4000
FNO_BATCH = 16
PHYS_EPOCHS = 200
PHYS_PATIENCE = 15
DATA_EPOCHS = 200
DATA_PATIENCE = 20
W_IC = 20.0
CORR_BATCH = 64
EVAL_CHUNK = 8192
VAL_PTS = 8000
SEED = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED)
PI = math.pi
print(f"device {DEVICE}", flush=True)


def load_split(name):
    v = np.load(os.path.join(DATA_DIR, name + "_v.npy")).astype(np.float32)
    u = np.load(os.path.join(DATA_DIR, name + "_u.npy")).astype(np.float32)
    return v, u


x_np = np.load(os.path.join(DATA_DIR, "x.npy")).astype(np.float32)
t_np = np.load(os.path.join(DATA_DIR, "t.npy")).astype(np.float32)
Nx, Nt = x_np.shape[0], t_np.shape[0]
Ng = Nt * Nx
T_END = float(t_np.max())
tr_v_n, tr_u_n = load_split("train")
va_v_n, va_u_n = load_split("val")
te_v_n, te_u_n = load_split("test")

V_MEAN, V_STD = float(tr_v_n.mean()), float(tr_v_n.std())
U_MEAN, U_STD = float(tr_u_n.mean()), float(tr_u_n.std())
C0 = float(np.sqrt((tr_v_n ** 2).mean()))
print(f"C0 {C0:.4f} DELTA {DELTA:.4e}", flush=True)

to = lambda a: torch.tensor(a, device=DEVICE)
tr_v, tr_u = to(tr_v_n), to(tr_u_n)
va_v, va_u = to(va_v_n), to(va_u_n)
te_v, te_u = to(te_v_n), to(te_u_n)
ntr, nval, nte = tr_v.shape[0], va_v.shape[0], te_v.shape[0]
tr_uf = tr_u.reshape(ntr, Ng); va_uf = va_u.reshape(nval, Ng); te_uf = te_u.reshape(nte, Ng)

x_row = to(x_np); t_row = to(t_np)
TT, XX = torch.meshgrid(t_row, x_row, indexing="ij")
grid_full = torch.stack([XX.reshape(-1), TT.reshape(-1)], dim=-1)
ic_coords = torch.stack([x_row, torch.zeros_like(x_row)], dim=-1)
X2D = XX; T2D = TT
ms = torch.arange(1, M_FOUR + 1, device=DEVICE).float()
FEAT = 2 * M_FOUR + 1
VAL_FIDX = torch.randint(0, Ng, (VAL_PTS,), device=DEVICE)
VAL_COLL = torch.rand(PHYS_BATCH, N_COLLOC, 2, device=DEVICE); VAL_COLL[..., 1] *= T_END


def vnorm(v): return (v - V_MEAN) / V_STD
def gnorm(g): return (g - U_MEAN) / U_STD


def features(c):
    x = c[..., 0:1]; t = c[..., 1:2]
    ang = 2 * PI * x * ms
    return torch.cat([torch.sin(ang), torch.cos(ang), t / T_END], dim=-1)


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


class DeepONetST(nn.Module):
    def __init__(self, extra, branch_in, bh, P, th):
        super().__init__()
        self.branch = MLP([branch_in] + bh + [P])
        self.trunk = MLP([FEAT + extra] + th + [P])
        self.b0 = nn.Parameter(torch.zeros(1))
        self.extra = extra

    def eval_grid(self, vin, coords, gextra=None):
        b = self.branch(vin)
        if self.extra == 0:
            return b @ self.trunk(features(coords)).t() + self.b0
        f = features(coords).unsqueeze(0).expand(b.shape[0], -1, -1)
        f = torch.cat([f, gextra.unsqueeze(-1)], dim=-1)
        return torch.einsum("bp,bnp->bn", b, self.trunk(f)) + self.b0

    def pointwise(self, vin, coords):
        b = self.branch(vin)
        return torch.einsum("bp,bnp->bn", b, self.trunk(features(coords))) + self.b0


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


def rel_l2(pred, u):
    return (torch.linalg.norm(pred - u, dim=1) / (torch.linalg.norm(u, dim=1) + 1e-12)).mean().item()


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def clone(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def fit(model, step, val, epochs, patience, tag):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, state, wait = float("inf"), None, 0
    for ep in range(epochs):
        model.train(); tl = 0.0
        for _ in range(STEPS_PER_EPOCH):
            opt.zero_grad(); loss = step(); loss.backward(); opt.step(); tl += loss.item()
        model.eval(); vl = val()
        if vl < best * (1 - REL_TOL):
            best, state, wait = vl, clone(model), 0
        else:
            wait += 1
        print(f"{tag} ep {ep} train {tl:.3e} val {vl:.3e} best {best:.3e} wait {wait}", flush=True)
        if wait >= patience:
            break
    model.load_state_dict(state)
    return model


def channels_pure(v):
    B = v.shape[0]
    vb = vnorm(v)[:, None, :].expand(B, Nt, Nx)
    return torch.stack([vb, X2D[None].expand(B, Nt, Nx), T2D[None].expand(B, Nt, Nx)], dim=1)


def channels_comb(v, g):
    B = v.shape[0]
    vb = vnorm(v)[:, None, :].expand(B, Nt, Nx)
    return torch.stack([gnorm(g), vb, X2D[None].expand(B, Nt, Nx), T2D[None].expand(B, Nt, Nx)], dim=1)


def fno_pure_full(model, v):
    n = v.shape[0]; out = torch.empty(n, Nt, Nx, device=DEVICE)
    for b in range(0, n, FNO_BATCH):
        e = min(b + FNO_BATCH, n); out[b:e] = model(channels_pure(v[b:e]))
    return out


def fno_comb_full(model, v, g_field):
    n = v.shape[0]; out = torch.empty(n, Nt, Nx, device=DEVICE)
    for b in range(0, n, FNO_BATCH):
        e = min(b + FNO_BATCH, n); out[b:e] = g_field[b:e] + model(channels_comb(v[b:e], g_field[b:e]))
    return out


def don_pred_pure(model, v):
    n = v.shape[0]; out = torch.empty(n, Ng, device=DEVICE)
    for b in range(0, n, CORR_BATCH):
        e = min(b + CORR_BATCH, n); vin = vnorm(v[b:e])
        for c in range(0, Ng, EVAL_CHUNK):
            out[b:e, c:c + EVAL_CHUNK] = model.eval_grid(vin, grid_full[c:c + EVAL_CHUNK])
    return out


def don_pred_comb(model, v, g_field):
    n = v.shape[0]; g_flat = g_field.reshape(n, Ng); out = torch.empty(n, Ng, device=DEVICE)
    for b in range(0, n, CORR_BATCH):
        e = min(b + CORR_BATCH, n); vin = vnorm(v[b:e])
        for c in range(0, Ng, EVAL_CHUNK):
            cc = grid_full[c:c + EVAL_CHUNK]; gpts = g_flat[b:e, c:c + EVAL_CHUNK]
            out[b:e, c:c + EVAL_CHUNK] = gpts + model.eval_grid(vin, cc, gnorm(gpts))
    return out


def prior_full(prior, v):
    n = v.shape[0]; out = torch.empty(n, Ng, device=DEVICE)
    for b in range(0, n, CORR_BATCH):
        e = min(b + CORR_BATCH, n); vin = vnorm(v[b:e])
        for c in range(0, Ng, EVAL_CHUNK):
            out[b:e, c:c + EVAL_CHUNK] = prior.eval_grid(vin, grid_full[c:c + EVAL_CHUNK])
    return out.reshape(n, Nt, Nx)


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

    fit(prior, step, val, PHYS_EPOCHS, PHYS_PATIENCE, f"{name} prior")
    for p in prior.parameters():
        p.requires_grad_(False)
    prior.eval()
    return prior


def phys_residual(prior, name):
    tot, cnt = 0.0, 0
    for b in range(0, nte, PHYS_BATCH):
        e = min(b + PHYS_BATCH, nte); vb = te_v[b:e]
        coll = torch.rand(vb.shape[0], N_COLLOC, 2, device=DEVICE); coll[..., 1] *= T_END
        coll = coll.requires_grad_(True)
        u, ux, uxx, uxxx, ut = derivs(prior, vnorm(vb), coll)
        r = residual(name, u, ux, uxx, uxxx, ut)
        tot += (r ** 2).sum().item(); cnt += r.numel()
    return tot / cnt


ORDER = ["phys_residual", "prior_l2", "puredata_deeponet", "puredata_fno", "combined_deeponet", "combined_fno"]
results = {}

for name in MISSPECS:
    prior = train_prior(name)
    results[(name, "phys_residual")] = phys_residual(prior, name)
    G_tr = prior_full(prior, tr_v); G_va = prior_full(prior, va_v); G_te = prior_full(prior, te_v)
    G_trf = G_tr.reshape(ntr, Ng); G_vaf = G_va.reshape(nval, Ng)
    results[(name, "prior_l2")] = rel_l2(G_te.reshape(nte, Ng), te_uf)
    print(f"{name} phys_residual {results[(name, 'phys_residual')]:.6e} prior_l2 {results[(name, 'prior_l2')]:.6e}", flush=True)

    pd_don = DeepONetST(0, Nx, SMALL_BH, SMALL_P, SMALL_TH).to(DEVICE)

    def s():
        idx = torch.randint(0, ntr, (DON_BATCH,), device=DEVICE); fidx = torch.randint(0, Ng, (N_PTS,), device=DEVICE)
        return ((pd_don.eval_grid(vnorm(tr_v[idx]), grid_full[fidx]) - tr_uf[idx][:, fidx]) ** 2).mean()

    def v():
        with torch.no_grad():
            return ((pd_don.eval_grid(vnorm(va_v), grid_full[VAL_FIDX]) - va_uf[:, VAL_FIDX]) ** 2).mean().item()

    fit(pd_don, s, v, DATA_EPOCHS, DATA_PATIENCE, f"{name} puredata_deeponet")
    results[(name, "puredata_deeponet")] = rel_l2(don_pred_pure(pd_don, te_v), te_uf)

    pd_fno = FNO2d(3, FW, FM, FL).to(DEVICE)

    def s():
        idx = torch.randint(0, ntr, (FNO_BATCH,), device=DEVICE)
        return ((pd_fno(channels_pure(tr_v[idx])) - tr_u[idx]) ** 2).mean()

    def v():
        with torch.no_grad():
            return ((fno_pure_full(pd_fno, va_v) - va_u) ** 2).mean().item()

    fit(pd_fno, s, v, DATA_EPOCHS, DATA_PATIENCE, f"{name} puredata_fno")
    results[(name, "puredata_fno")] = rel_l2(fno_pure_full(pd_fno, te_v).reshape(nte, Ng), te_uf)

    cb_don = DeepONetST(1, Nx, SMALL_BH, SMALL_P, SMALL_TH).to(DEVICE)

    def s():
        idx = torch.randint(0, ntr, (DON_BATCH,), device=DEVICE); fidx = torch.randint(0, Ng, (N_PTS,), device=DEVICE)
        gpts = G_trf[idx][:, fidx]
        pred = gpts + cb_don.eval_grid(vnorm(tr_v[idx]), grid_full[fidx], gnorm(gpts))
        return ((pred - tr_uf[idx][:, fidx]) ** 2).mean()

    def v():
        with torch.no_grad():
            gpts = G_vaf[:, VAL_FIDX]
            pred = gpts + cb_don.eval_grid(vnorm(va_v), grid_full[VAL_FIDX], gnorm(gpts))
            return ((pred - va_uf[:, VAL_FIDX]) ** 2).mean().item()

    fit(cb_don, s, v, DATA_EPOCHS, DATA_PATIENCE, f"{name} combined_deeponet")
    results[(name, "combined_deeponet")] = rel_l2(don_pred_comb(cb_don, te_v, G_te), te_uf)

    cb_fno = FNO2d(4, FW, FM, FL).to(DEVICE)

    def s():
        idx = torch.randint(0, ntr, (FNO_BATCH,), device=DEVICE)
        pred = G_tr[idx] + cb_fno(channels_comb(tr_v[idx], G_tr[idx]))
        return ((pred - tr_u[idx]) ** 2).mean()

    def v():
        with torch.no_grad():
            return ((fno_comb_full(cb_fno, va_v, G_va) - va_u) ** 2).mean().item()

    fit(cb_fno, s, v, DATA_EPOCHS, DATA_PATIENCE, f"{name} combined_fno")
    results[(name, "combined_fno")] = rel_l2(fno_comb_full(cb_fno, te_v, G_te).reshape(nte, Ng), te_uf)

    print(f"{name} params don {nparams(cb_don)} fno {nparams(cb_fno)}", flush=True)
    for m in ORDER:
        print(f"{name} {m} {results[(name, m)]:.6e}", flush=True)

os.makedirs(OUT_DIR, exist_ok=True)
path = os.path.join(OUT_DIR, "pde2_metrics.txt")
with open(path, "w") as f:
    f.write("misspec\tmetric\tvalue\n")
    for name in MISSPECS:
        for m in ORDER:
            f.write(f"{name}\t{m}\t{results[(name, m)]:.6e}\n")

print("==== FINAL ====", flush=True)
for name in MISSPECS:
    for m in ORDER:
        print(f"{name}\t{m}\t{results[(name, m)]:.6e}", flush=True)
print(f"wrote {path}", flush=True)
