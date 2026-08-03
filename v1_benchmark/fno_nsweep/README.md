# fno_nsweep

FNO-only data-budget sweep. DeepONet data models are removed; the physics prior is still a
DeepONet (trained label-free on the assumed residual), since only it supports the pointwise
autodiff residual.

Per PDE, for each misspecification level `low/med/high/max`:
- `prior` — frozen physics prior, evaluated on the test set.
- `combined_fno` — prior + FNO correction, trained on `N` labels for `N in {32,64,128,256,512}`.

Plus, misspecification-independent:
- `puredata_fno` — FNO from scratch, trained on `N in {32,64,128,256,512, 2048(full)}`.

So each PDE records `4 priors + 4x5 combined + 6 pure = 30` experiments (150 total).
Pure and combined share the same nested training subsets (fixed permutation, seed 0), so their
curves are comparable at matched `N`.

Training is silent (no per-epoch output); same Adam / early-stopping as the suite scripts. Data
is pulled from Hugging Face into `../mm_data` (shared with the parent scripts).

## Outputs (written to `fno_nsweep/outputs/`, git-ignored)
- `pdeN_metrics.txt` — TSV: `misspec  method  N  value` (value = relative L2 on the test set;
  `N = -1` for prior/phys_residual rows).
- `pdeN_preds.npz` — grid axes, `u_true` (test targets), and one array per experiment holding the
  predicted field `u_theta` on the test set (keys `prior_<lvl>`, `combined_<lvl>_N<n>`,
  `puredata_N<n>`).

## Run
```bash
for f in pde1_diffreac_nsweep pde2_burgers_nsweep pde3_advdiff_nsweep pde4_fractional_nsweep pde5_helmholtz_nsweep; do python fno_nsweep/$f.py; done
```
