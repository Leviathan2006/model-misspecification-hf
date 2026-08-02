"""End-to-end dry run: train a real FNO M_psi: u_theta -> u to imitate a KNOWN
operator, then probe the TRAINED network. If the probe reads the right exponent off
a trained FNO (not just an analytic op), the train-then-probe loop the 5 scripts use
is sound. CPU, synthetic data, no download.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from probe import probe_operator

torch.manual_seed(0)
DEV = torch.device("cpu")


class SpectralConv1d(nn.Module):
    def __init__(self, ic, oc, modes):
        super().__init__(); self.modes = modes
        s = 1.0 / (ic * oc)
        self.weight = nn.Parameter(s * torch.rand(ic, oc, modes, dtype=torch.cfloat))

    def forward(self, x):
        N = x.shape[-1]; xft = torch.fft.rfft(x, dim=-1)
        m = min(self.modes, xft.shape[-1])
        out = torch.zeros(x.shape[0], self.weight.shape[1], xft.shape[-1], dtype=torch.cfloat)
        out[:, :, :m] = torch.einsum("bim,iom->bom", xft[:, :, :m], self.weight[:, :, :m])
        return torch.fft.irfft(out, n=N, dim=-1)


class FNO1d(nn.Module):
    def __init__(self, ic, w, m, L):
        super().__init__()
        self.fc0 = nn.Linear(ic, w)
        self.sp = nn.ModuleList([SpectralConv1d(w, w, m) for _ in range(L)])
        self.w = nn.ModuleList([nn.Conv1d(w, w, 1) for _ in range(L)])
        self.fc1 = nn.Linear(w, 128); self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc0(x.permute(0, 2, 1)).permute(0, 2, 1)
        for s, w in zip(self.sp, self.w):
            x = F.gelu(s(x) + w(x))
        x = F.gelu(self.fc1(x.permute(0, 2, 1)))
        return self.fc2(x).squeeze(-1)


def smooth(S, n, cutoff=10):
    x = torch.randn(S, n); Xf = torch.fft.rfft(x, dim=-1); Xf[:, cutoff:] = 0
    return torch.fft.irfft(Xf, n=n, dim=-1)


def kp(u, p, alpha):
    n = u.shape[-1]; k = torch.arange(n // 2 + 1).float()
    return u + torch.fft.irfft(torch.fft.rfft(u, dim=-1) * (alpha * k ** p), n=n, dim=-1)


Nx = 96
xrow = torch.linspace(0, 1, Nx)
utheta = smooth(320, Nx)
utrue = kp(utheta, 1.5, 8e-3)          # target: k^1.5 correction (the P4 story)
STD = utheta.std()


def chan(u):
    B = u.shape[0]
    return torch.stack([u / STD, xrow[None].expand(B, Nx)], dim=1)


M = FNO1d(2, 48, 24, 4)
opt = torch.optim.Adam(M.parameters(), lr=1e-3)
tr_u, va_u = utheta[:256], utheta[256:]
ty, vy = utrue[:256], utrue[256:]
for ep in range(400):
    M.train(); idx = torch.randint(0, 256, (64,))
    opt.zero_grad(); loss = ((M(chan(tr_u[idx])) - ty[idx]) ** 2).mean()
    loss.backward(); opt.step()
M.eval()
with torch.no_grad():
    fit_err = (M(chan(va_u)) - vy).norm() / vy.norm()
    feats = probe_operator(lambda u: M(chan(u)), utheta[256:264].detach(), n_rand=3)

print(f"fit rel-err {fit_err:.3f}")
print(f"recovered exponent {feats['exponent']:.2f} (target 1.5)   "
      f"mag {feats['magnitude']:.3f}  loc {feats['locality']:.1f}  "
      f"equiv {feats['equivariance']:.3f}  svar {feats['state_var']:.3f}")
ok = (fit_err < 0.15) and (abs(feats["exponent"] - 1.5) < 0.5) and \
     all(torch.isfinite(torch.tensor(feats[k])).all() for k in ("magnitude", "locality", "equivariance", "state_var"))
print("DRYRUN PASS" if ok else "DRYRUN FAIL")
