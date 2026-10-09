# Training the AutoConf Models

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
as [`ado-sfttrainer.csv`](https://huggingface.co/datasets/ibm-research/LLMFineTuningBench/blob/main/ado-sfttrainer.csv).
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

Once `ado-sfttrainer.csv` is available in `LLMFineTuningBench`, build both
models with:

```terminal
uv run autoconf_build_model
```

This trains both the OOM classifier (`v4-0-0`) and the throughput regressor
(`v4-1-0-regressor`). Use `--model` to build only one:

```terminal
uv run autoconf_build_model --model classifier   # OOM classifier only
uv run autoconf_build_model --model regressor    # throughput regressor only
uv run autoconf_build_model --model all          # both (default)
```

AutoConf 2.1 pins AutoGluon 1.6.1. Downloaded data and generated models are
ignored by Git.

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

## Default Presets

These presets are recommended for the default dataset
([`ado-sfttrainer.csv`](https://huggingface.co/datasets/ibm-research/LLMFineTuningBench/blob/main/ado-sfttrainer.csv)).
LightGBM (`GBM`) is excluded to avoid the `libomp` dependency on macOS.
For other options see the AutoGluon [documentation](https://auto.gluon.ai/stable/tutorials/tabular/).

| Model | Preset | Training time (~15 k samples) |
| --- | --- | --- |
| Classifier (`v4-0-0`) | `medium_quality` + `optimize_for_deployment` | ~2 min |
| Regressor (`v4-1-0-regressor`) | `good` (15 min time limit, 5-fold bagging) | up to ~15 min |

> **Note:** The regressor default preset (`good`, up to 15 min) is more expensive
> than the classifier default (`medium_quality`, ~2 min) by design — it targets
> prediction accuracy over disk size, which matters for throughput estimates.

Use `--preset-quality` to override the preset for both models at once.
The `good` shorthand is equivalent to `good_quality` for the regressor.
