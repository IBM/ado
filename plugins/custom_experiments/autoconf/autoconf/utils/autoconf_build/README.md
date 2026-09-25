# Training the AutoConf OOM Classifier

AutoConf uses an AutoGluon binary classifier to predict whether a fine-tuning
configuration will complete without a GPU out-of-memory error. Model binaries
are not stored in ado or on Hugging Face.

The classifier is trained automatically on the first inference call when no
model is present (see [Installation and Usage](../../README.md#installation-and-usage)).
The `autoconf_build_model` CLI described here is for manual pre-training or
retraining with a custom configuration, and calls the same `build_model()`
function internally.

Build the model in the Python environment where the AutoConf recommender will
use it.

## Dataset

The measurements are hosted in the Hugging Face repository
[`ibm-research/LLMFineTuningBench`](https://huggingface.co/datasets/ibm-research/LLMFineTuningBench)
as [`ado-sfttrainer-v1-0-0.csv`](https://huggingface.co/datasets/ibm-research/LLMFineTuningBench/blob/main/ado-sfttrainer-v1-0-0.csv).
The builder downloads that file to `autoconf/data/dataset.csv` when the local
file is absent. It reuses the local file on later runs.

The classifier uses these columns:

- `model_name`
- `method`
- `number_gpus`
- `gpu_model`
- `tokens_per_sample`
- `batch_size`

The `throughput_recommender` loads the existing classifier or uses its on-demand
builder if it is absent. Its separate regressor uses this same downloaded
dataset. It trains only on successful configurations with a positive
`dataset_tokens_per_second` value where the global `batch_size` is evenly
divisible by `number_gpus` (the total GPU count). It uses the same feature
columns, excluding all metadata columns. Once trained, the regressor is saved
and loaded on later calls.

If the dataset has no explicit `is_valid` column, the builder derives it from
`train_runtime`: a recorded runtime is a successful run and a missing runtime is
a failed run. Rows rejected by AutoConf's deterministic configuration rules are
removed before fitting.

## Reproduce a Model

From the root of an ado checkout, create an environment and install AutoConf:

```terminal
uv venv --python 3.13
uv pip install -e plugins/custom_experiments/autoconf
```

Once `ado-sfttrainer-v1-0-0.csv` is available in `LLMFineTuningBench`, build
the classifier model with:

```terminal
uv run autoconf_build_model
```

The generated classifier is written to `autoconf/models/v4-0-0/`, where the
installed recommenders load it. AutoConf 2.1 pins AutoGluon 1.6.1. The
throughput recommender trains its separate regressor automatically on first
use and stores it under `autoconf/models/v4-1-0-regressor/`. Downloaded data
and model output are ignored by Git.

Use `uv run autoconf_build_model --help` to select a different local data path,
dataset URL, model root, training fraction, or AutoGluon quality preset.

For the minimal reproducibility section in the Hugging Face dataset card, use
the released package in a clean environment:

```terminal
python -m venv .venv
source .venv/bin/activate
python -m pip install "ado-autoconf==2.1.0"
autoconf_build_model
```

The package brings
in `ado-core` and pins AutoGluon 1.6.1. The command downloads the training CSV
and writes the model into the same environment that will load it.

## Minimal Computational Performance Results of Various Presets

These approaches prioritize **inference speed**, **disk usage**, and **training
time** over raw accuracy.

You can modify the presets and customize the model creation according to your
needs.
For detailed options and explanations, refer to the official AutoGluon
[documentation](https://auto.gluon.ai/stable/tutorials/tabular/).

We exclude LightGBM to avoid needing the additional dependency on `libomp` on
macOS machines.

### Current Setting

- **Preset:** `medium_quality` + `optimize_for_deployment`
- **Excluded Models:** `GBM`
- **Training Time:** ~1 minutes on ~12,000 samples
- **Model Size:** ~5 MB

```python
fit_params = {
    "presets": ["medium_quality", "optimize_for_deployment"],
    "excluded_model_types": "GBM",
}
```

---

### Option 1: Medium Quality Only

- **Preset:** `good_quality`
- **Excluded Models:** `GBM`
- **Training Time:** equal to current setting
- **Model Size:** ~300 MB

```python
fit_params = {"presets": ["medium_quality"], "excluded_model_types": "GBM"}
```

### Option 2: Good Quality + Optimize for Deployment

- **Preset:** `good_quality`, `optimize_for_deployment`
- **Excluded Models:** `GBM`
- **Training Time:** ~30× longer than current setting
- **Model Size:** ~353 MB

```python
fit_params = {
    "presets": ["good_quality", "optimize_for_deployment"],
    "excluded_model_types": "GBM",
}
```

### Option 3: Good Quality Only

- **Preset:** `good_quality`
- **Excluded Models:** `GBM`
- **Training Time:** ~30× longer than current setting
- **Model Size:** ~600 MB

```python
fit_params = {"presets": ["good_quality"], "excluded_model_types": "GBM"}
```
