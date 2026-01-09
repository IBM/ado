# TRIM EXAMPLE — Quickstart

## Install (from repo root)

```bash
uv pip install plugins/operators/trim
uv pip install -e examples/trim_custom_experiments
```

## Create the space

```bash
cd examples/trim_custom_experiments/trim_custom_experiments/configs
ado create space -f space_pressure.yaml --new-sample-store
```

> **Note:** The custom experiment `calculate_pressure_ideal_gas`
> is defined following  
> <https://ibm.github.io/ado/actuators/creating-custom-experiments/#decorating-your-custom-experiment-function>

---

In the same folder you can run

```bash
 ado create operation -f op_pressure.yaml --use-latest space
```

You will see that the operation bootstraps the parameter exploration
with a no-priors characterization, then it proceeds with iterative
modeling until a stopping criterion is met.
