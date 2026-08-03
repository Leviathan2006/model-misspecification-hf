# model-misspecification

Current work: **`misspec_assay.py`** — additive CNO corrector + the misspecification
"assay panel" (regress the correction against candidate differential operators) + the
filters-as-stencils mechanistic check. Data loads from Hugging Face.

Everything prior (the 5-PDE FNO benchmark, the N-sweep, the FNO/CNO spectral-probe
diagnostics, the LaTeX report) is archived under [`v1_benchmark/`](v1_benchmark/).

## Run
```bash
pip install -r requirements.txt
MM_SMOKE=1 python misspec_assay.py     # CPU wiring check, no download
python misspec_assay.py                # real run (GPU; pulls data from HF)
```
Output: `assay_metrics.txt` — per-cell `R2_u_xx` (diffusive-error statistic), cross-reactivity
`R2_{u_x,u,u_ux,u_xxx}` (specificity), `cno_fit_err`, `prior_l2`, and `stencil_*` (learned-filter
projection onto {I, d_x, d_t, d_xx, d_tt, laplacian, d_xt}).
