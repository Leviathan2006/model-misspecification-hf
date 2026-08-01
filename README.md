# model-misspecification-hf

Frozen-prior PDE suites, loading data from Hugging Face instead of Kaggle.

Data: [`alexanderthegreat69420/Model_misspecification`](https://huggingface.co/datasets/alexanderthegreat69420/Model_misspecification)

Each script downloads its PDE's zip on first run, extracts it under `mm_data/`, and
writes metrics to `outputs/`. Both paths are overridable via the `MM_DATA_DIR` and
`MM_OUT_DIR` environment variables.

## Setup

```bash
pip install -r requirements.txt
```

For a Blackwell GPU install the CUDA 12.8 torch build first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Run

```bash
python pde1_diffreac_suite.py
python pde2_burgers_suite.py
python pde3_advdiff_suite.py
python pde4_fractional_suite.py
python pde5_helmholtz_suite.py
```

| Script | HF zip |
|---|---|
| `pde1_diffreac_suite.py` | `pde1_diffusion_reaction.zip` |
| `pde2_burgers_suite.py` | `pde2_burgers.zip` |
| `pde3_advdiff_suite.py` | `pde4_advection_diffusion.zip` |
| `pde4_fractional_suite.py` | `pde5_fractional_diffusion.zip` |
| `pde5_helmholtz_suite.py` | `pde6_helmholtz2d.zip` |

If the dataset is gated/private, authenticate first with `huggingface-cli login`.
