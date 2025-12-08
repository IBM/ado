# Training a Classifier for OOM Jobs

In this folder, you will find an **IPython script** that demonstrates a possible
approach to train a classifier for OOM (Out-Of-Memory) jobs.

This approach prioritizes **inference speed**, **disk usage**, and **training
time** over raw accuracy.

You can modify the presets and customize the model creation according to your
needs.

For detailed options and explanations, refer to the official AutoGluon
[documentation](https://auto.gluon.ai/stable/tutorials/tabular/).

---

## Minimal Computational Performance Results of Various Presets

We exclude LightGBM to avoid needing the additional dependency on `libomp` on
macOS machines.

### Current Setting

* **Preset:** `medium_quality` + `optimize_for_deployment`
* **Excluded Models:** `GBM`
* **Training Time:** ~1 minutes on ~12,000 samples
* **Model Size:** ~5 MB

```python
fit_params = {"presets":["medium_quality",
"optimize_for_deployment"],
"excluded_model_types": "GBM"}
```

---

### Option 1: Medium Quality Only

* **Preset:** `good_quality`
* **Excluded Models:** `GBM`
* **Training Time:** equal to current setting
* **Model Size:** ~300 MB

```python
fit_params = {"presets": ["medium_quality"], "excluded_model_types": "GBM"}
```

### Option 2: Good Quality + Optimize for Deployment

* **Preset:** `good_quality`, `optimize_for_deployment`
* **Excluded Models:** `GBM`
* **Training Time:** ~30× longer than current setting
* **Model Size:** ~353 MB

```python
fit_params = {
    "presets": ["good_quality", "optimize_for_deployment"],
    "excluded_model_types": "GBM"
}
```

### Option 3: Good Quality Only

* **Preset:** `good_quality`
* **Excluded Models:** `GBM`
* **Training Time:** ~30× longer than current setting
* **Model Size:** ~600 MB

```python
fit_params = {"presets": ["good_quality"], "excluded_model_types": "GBM"}
```
