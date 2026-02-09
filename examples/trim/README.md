# TRIM EXAMPLE — Quickstart

## Install (from repository root)

```bash
pip install plugins/operators/trim/
pip install -e examples/trim/custom_experiments/
```

> **Note:** All commands below assume you are running them from
the **top-level of the repository**, not from inside `examples/trim_custom_experiments/`.

---

## Create the space

You can create the space without changing directories by specifying
the full path to the config file:

```bash
ado create space -f examples/trim/configs/space_pressure.yaml --new-sample-store
```

> **Tip:** The custom experiment `calculate_pressure_ideal_gas`  
> is defined following  
> [ADO documentation on decorating custom experiment functions](https://ibm.github.io/ado/actuators/creating-custom-experiments/#decorating-your-custom-experiment-function).

---

## Create the operation

Similarly, you can create the operation without `cd`:

```bash
ado create operation -f examples/trim/configs/op_pressure.yaml --use-latest space
```

This operation bootstraps the parameter exploration with a no-priors characterization,
then proceeds with iterative modeling until a stopping criterion is met.
