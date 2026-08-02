# fno_diag

Diagnose misspecification by probing the learned correction operator.

For each PDE and misspecification level we train a **large FNO** $M_\psi: u_\theta \to u$
(prior field to truth), then probe its local linearization $J = \partial M_\psi/\partial u_\theta$
with finite-difference Jacobian-vector products and read five features of the correction
operator $(J-I)$ — identically for 1D/2D and linear/nonlinear operators:

| feature | meaning |
|---|---|
| `magnitude`    | $\lVert J-I\rVert$ — how far the prior is from truth |
| `exponent`     | slope of the correction transfer $\lvert T(k)\rvert \sim k^p$ — differential order of the discrepancy |
| `locality`     | spatial spread of the impulse response — local vs nonlocal |
| `equivariance` | $\lVert[J,\text{shift}]\rVert$ — constant-coefficient vs heterogeneous |
| `state_var`    | variation of the gain across states — linear vs nonlinear |

`probe.py` is the shared toolkit. `selftest.py` and `dryrun.py` validate it on CPU against
operators with known symbols (recovers $k^2$, $k^{1.5}$, $k^3$; flags heterogeneous and
nonlinear operators).

## Validate on CPU first (no download)
```bash
python fno_diag/selftest.py     # probe math vs analytic operators
python fno_diag/dryrun.py        # train a real FNO to imitate k^1.5, then probe it
```
Both should print `... PASSED` / `DRYRUN PASS`.

## Run the benchmark (GPU)
Data loads from Hugging Face into `../mm_data` (shared with the parent scripts). Outputs go
to `fno_diag/outputs/` (git-ignored).
```bash
for f in pde1_diffreac_diag pde2_burgers_diag pde3_advdiff_diag pde4_fractional_diag pde5_helmholtz_diag; do python fno_diag/$f.py; done
```

## Outputs (`fno_diag/outputs/`)
- `pdeN_diag_metrics.txt` — TSV `misspec  feature  value` (5 features × 4 levels).
- `pdeN_diag_transfer.npz` — `<level>_k`, `<level>_T`: the correction transfer function per level, for the transfer-function figures.

Expected signatures (validation against the designed misspecifications): wrong-viscosity → $k^2$;
fractional → $k^{1.5}$; dispersive/KdV → $k^3$; heterogeneous (`max` cases) → high `equivariance`;
nonlinear advection → high `state_var`.
