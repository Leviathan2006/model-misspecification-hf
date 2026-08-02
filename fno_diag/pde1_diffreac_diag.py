import os, math, zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from probe import probe_operator
from huggingface_hub import hf_hub_download

HF_REPO = "alexanderthegreat69420/Model_misspecification"
HF_ZIP = "pde1_diffusion_reaction.zip"
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

D_TRUE, K0 = 0.1, 0.5
MISSPECS = ["low", "med", "high", "max"]

PRIOR_BH = [128, 128]
PRIOR_P = 64
PRIOR_TH = [64, 64, 64]
SMALL_BH = [128, 128]
SMALL_P = 64
SMALL_TH = [64, 64, 64]
FW, FM, FL = 24, 16, 2

BATCH = 256
STEPS_PER_EPOCH = 8
LR = 1e-3
REL_TOL = 5e-3
PRIOR_REL_TOL = 1e-4
PHYS_EPOCHS = 1500
PHYS_PATIENCE = 60
DATA_EPOCHS = 300
DATA_PATIENCE = 20
W_BC = 10.0
SEED = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED)
PI = math.pi


def load_split(name):
    v = np.load(os.path.join(DATA_DIR, name + "_v.npy")).astype(np.float32)
    u = np.load(os.path.join(DATA_DIR, name + "_u.npy")).astype(np.float32)
    return v, u


x_np = np.load(os.path.join(DATA_DIR, "x.npy")).astype(np.float32)
Nx = x_np.shape[0]
tr_v_n, tr_u_n = load_split("train")
va_v_n, va_u_n = load_split("val")
te_v_n, te_u_n = load_split("test")

V_MEAN, V_STD = float(tr_v_n.mean()), float(tr_v_n.std())
U_MEAN, U_STD = float(tr_u_n.mean()), float(tr_u_n.std())

to = lambda a: torch.tensor(a, device=DEVICE)
tr_v, tr_u = to(tr_v_n), to(tr_u_n)
va_v, va_u = to(va_v_n), to(va_u_n)
te_v, te_u = to(te_v_n), to(te_u_n)
ntr, nval, nte = tr_v.shape[0], va_v.shape[0], te_v.shape[0]
x_grid = to(x_np).view(Nx, 1)
x_row = to(x_np)


def vnorm(v): return (v - V_MEAN) / V_STD
def gnorm(g): return (g - U_MEAN) / U_STD


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


class DeepONet(nn.Module):
    def __init__(self, branch_in, bh, P, th):
        super().__init__()
        self.branch = MLP([branch_in] + bh + [P])
        self.trunk = MLP([1] + th + [P])
        self.b0 = nn.Parameter(torch.zeros(1))

    def forward(self, vin, x):
        b = self.branch(vin)
        if x.dim() == 2:
            return b @ self.trunk(x).t() + self.b0
        return torch.einsum("bp,bnp->bn", b, self.trunk(x)) + self.b0


class SpectralConv1d(nn.Module):
    def __init__(self, ic, oc, modes):
        super().__init__()
        self.modes = modes
        s = 1.0 / (ic * oc)
        self.weight = nn.Parameter(s * torch.rand(ic, oc, modes, dtype=torch.cfloat))

    def forward(self, x):
        B, N = x.shape[0], x.shape[-1]
        xft = torch.fft.rfft(x, dim=-1)
        m = min(self.modes, xft.shape[-1])
        out = torch.zeros(B, self.weight.shape[1], xft.shape[-1], dtype=torch.cfloat, device=x.device)
        out[:, :, :m] = torch.einsum("bim,iom->bom", xft[:, :, :m], self.weight[:, :, :m])
        return torch.fft.irfft(out, n=N, dim=-1)


class FNO1d(nn.Module):
    def __init__(self, ic, width, modes, layers):
        super().__init__()
        self.fc0 = nn.Linear(ic, width)
        self.sp = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(layers)])
        self.w = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc0(x.permute(0, 2, 1)).permute(0, 2, 1)
        for s, w in zip(self.sp, self.w):
            x = F.gelu(s(x) + w(x))
        x = F.gelu(self.fc1(x.permute(0, 2, 1)))
        return self.fc2(x).squeeze(-1)


def residual(name, u, ux, uxx, v, x0):
    if name == "low":
        return 0.11 * uxx - K0 * torch.exp(-u) * u - v
    if name == "med":
        return 0.15 * uxx - 0.7 * torch.exp(-u) * u - v
    if name == "high":
        return D_TRUE * uxx - K0 * u - v
    Dx = 0.1 * (1.0 + 0.5 * torch.cos(2 * PI * x0))
    Dpx = -0.1 * PI * torch.sin(2 * PI * x0)
    return Dx * uxx + Dpx * ux - K0 * torch.exp(-u) * u - v


def derivs(model, vin, xq):
    u = model(vin, xq)
    ux = torch.autograd.grad(u.sum(), xq, create_graph=True)[0][..., 0]
    uxx = torch.autograd.grad(ux.sum(), xq, create_graph=True)[0][..., 0]
    return u, ux, uxx


def rel_l2(pred, u):
    return (torch.linalg.norm(pred - u, dim=1) / (torch.linalg.norm(u, dim=1) + 1e-12)).mean().item()


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def clone(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def fit(model, step_fn, val_fn, epochs, patience, tag, rel_tol=REL_TOL):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, state, wait = float("inf"), None, 0
    for ep in range(epochs):
        model.train(); tl = 0.0
        for _ in range(STEPS_PER_EPOCH):
            idx = torch.randint(0, ntr, (BATCH,), device=DEVICE)
            opt.zero_grad(); loss = step_fn(idx); loss.backward(); opt.step(); tl += loss.item()
        model.eval(); vl = val_fn()
        if vl < best * (1 - rel_tol):
            best, state, wait = vl, clone(model), 0
        else:
            wait += 1
        if wait >= patience:
            break
    model.load_state_dict(state)
    return model


def fno_in_pure(v):
    return torch.stack([vnorm(v), x_row.unsqueeze(0).expand(v.shape[0], Nx)], dim=1)


def fno_in_comb(v, g):
    return torch.stack([vnorm(v), gnorm(g), x_row.unsqueeze(0).expand(v.shape[0], Nx)], dim=1)


def train_prior(name):
    prior = DeepONet(Nx, PRIOR_BH, PRIOR_P, PRIOR_TH).to(DEVICE)

    def step(idx):
        xq = x_grid.view(1, Nx, 1).expand(idx.shape[0], Nx, 1).contiguous().requires_grad_(True)
        u, ux, uxx = derivs(prior, vnorm(tr_v[idx]), xq)
        bc = u[:, 0] ** 2 + u[:, -1] ** 2
        return (residual(name, u, ux, uxx, tr_v[idx], xq[..., 0]) ** 2).mean() + W_BC * bc.mean()

    def val():
        xq = x_grid.view(1, Nx, 1).expand(nval, Nx, 1).contiguous().requires_grad_(True)
        u, ux, uxx = derivs(prior, vnorm(va_v), xq)
        bc = u[:, 0] ** 2 + u[:, -1] ** 2
        return ((residual(name, u, ux, uxx, va_v, xq[..., 0]) ** 2).mean() + W_BC * bc.mean()).item()

    fit(prior, step, val, PHYS_EPOCHS, PHYS_PATIENCE, f"{name} prior", rel_tol=PRIOR_REL_TOL)
    for p in prior.parameters():
        p.requires_grad_(False)
    prior.eval()
    return prior


def phys_residual(prior, name):
    tot, cnt = 0.0, 0
    for b in range(0, nte, BATCH):
        e = min(b + BATCH, nte); vb = te_v[b:e]
        xq = x_grid.view(1, Nx, 1).expand(vb.shape[0], Nx, 1).contiguous().requires_grad_(True)
        u, ux, uxx = derivs(prior, vnorm(vb), xq)
        r = residual(name, u, ux, uxx, vb, xq[..., 0])
        tot += (r ** 2).sum().item(); cnt += r.numel()
    return tot / cnt


MW, MM, ML = 64, 32, 4


def Mchan(u):
    B = u.shape[0]
    return torch.stack([gnorm(u), x_row.unsqueeze(0).expand(B, Nx)], dim=1)


metric_rows = []
transfers = {}

for name in MISSPECS:
    prior = train_prior(name)
    with torch.no_grad():
        g_tr = prior(vnorm(tr_v), x_grid); g_va = prior(vnorm(va_v), x_grid); g_te = prior(vnorm(te_v), x_grid)
    Mpsi = FNO1d(2, MW, MM, ML).to(DEVICE)
    fit(Mpsi, lambda idx: ((Mpsi(Mchan(g_tr[idx])) - tr_u[idx]) ** 2).mean(),
        lambda: ((Mpsi(Mchan(g_va)) - va_u) ** 2).mean().item(),
        DATA_EPOCHS, DATA_PATIENCE, "")
    Mpsi.eval()
    base = g_te[:8].detach()
    with torch.no_grad():
        feats = probe_operator(lambda u: Mpsi(Mchan(u)), base, n_rand=3)
    for key in ("magnitude", "exponent", "locality", "equivariance", "state_var"):
        metric_rows.append((name, key, feats[key]))
    transfers[f"{name}_k"] = feats["k"]; transfers[f"{name}_T"] = feats["transfer"]
    del prior, Mpsi, g_tr, g_va, g_te; torch.cuda.empty_cache()

os.makedirs(OUT_DIR, exist_ok=True)
mpath = os.path.join(OUT_DIR, "pde1_diag_metrics.txt")
with open(mpath, "w") as f:
    f.write("misspec\tfeature\tvalue\n")
    for a, b, c in metric_rows:
        f.write(f"{a}\t{b}\t{c:.6e}\n")
tpath = os.path.join(OUT_DIR, "pde1_diag_transfer.npz")
np.savez_compressed(tpath, **transfers)
print(f"wrote {mpath} and {tpath}", flush=True)
