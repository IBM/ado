<!-- markdownlint-disable code-block-style -->

# Build a surrogate model with TRIM

!!! abstract "Advanced tutorial"

    This walkthrough assumes you are comfortable with the core `ado`
    workflow — custom experiments, discovery spaces, and operations — and that
    you have already run at least one operation end-to-end. If you are new to
    `ado`, work through
    [Your first ado experiment](tutorials/density-example.md) first.

When evaluating points in a parameter space is expensive — a scientific
simulation, a machine learning training run, or a physical experiment — you
often cannot afford to measure every configuration. The `trim` operator
(**Transfer Refined Iterative Modeling**) solves this by building a predictive
model that can estimate outcomes at unmeasured points. It gathers data
intelligently, training and refining its model at each step, and halts once
further sampling stops improving model accuracy.

This walkthrough uses the `calculate_pressure_ideal_gas` custom experiment — a
simple Ideal Gas Law calculation — so the focus stays on `ado` and TRIM
mechanics rather than the domain.

We will:

1. Install the `trim` operator and the example custom experiment
2. Inspect the experiment interface
3. Define a discovery space for pressure across temperature, volume, and moles
4. Run TRIM to build a surrogate model
5. Inspect the sampled measurements and the saved model

!!! warning "Prerequisites"

    - `ado-core` installed (`pip install ado-core`)
    - The example package cloned from GitHub (the wheel is not published to
      PyPI)

    Clone the repository if you have not already done so:

    ```bash
    git clone https://github.com/ibm/ado.git
    cd ado
    ```

    Then install the operator and experiment packages:

    ```bash
    pip install ado-trim
    pip install -e examples/trim/custom_experiments/
    ```

    !!! danger "Python version compatibility"

        The `trim` operator is not available on Python 3.14 due to a dependency
        on `autogluon==1.5.0`, which requires `pyarrow==20.0.0` (incompatible
        with Python 3.14). Use Python 3.10–3.13.

    All commands in this walkthrough are run from the **repository root**
    (`ado/`).

!!! example "TL;DR"

    Once the packages are installed:

    <!-- markdownlint-disable MD013 -->
    ```bash
    ado create space -f examples/trim/example_yamls/space_pressure.yaml --new-sample-store
    ado create operation -f examples/trim/example_yamls/op_pressure.yaml --use-latest space
    ado show related space --use-latest
    ado show measurements space --use-latest
    ```
    <!-- markdownlint-enable MD013 -->

## Step 1 — Install the required packages

### The `trim` operator

The `trim` operator is distributed as a separate package:

```bash
pip install ado-trim
```

Confirm it is registered:

```bash
ado get operators
```

You should see `trim` listed:

```text
Available operators by type:
┌───────┬─────────────┬─────────┬────────────┐
│ INDEX │ OPERATOR    │ VERSION │ TYPE       │
├───────┼─────────────┼─────────┼────────────┤
│ 0     │ random_walk │ 2.0.0   │ explore    │
│ 1     │ ray_tune    │ 2.0.0   │ explore    │
│ 2     │ rifferla    │ 2.0.0   │ modify     │
│ 3     │ trim        │ 2.0.3   │ characterize│
└───────┴─────────────┴─────────┴────────────┘
```

### The `calculate_pressure_ideal_gas` custom experiment

The example ships a custom experiment that computes gas pressure from the
[Ideal Gas Law](https://en.wikipedia.org/wiki/Ideal_gas_law): `P = nRT/V`.

Install it from the example directory:

```bash
pip install -e examples/trim/custom_experiments/
```

Confirm `ado` can see it:

```bash
ado get experiments
```

You should see `calculate_pressure_ideal_gas` listed under `custom_experiments`.

Inspect the experiment interface:

```bash
ado describe experiment calculate_pressure_ideal_gas
```

```terminaloutput
Identifier: custom_experiments.calculate_pressure_ideal_gas

Required Inputs:

   Constitutive Properties:
    ─────────────────────────────────────────────────────────────────
     Identifier: mol
     Domain:
         Type: CONTINUOUS_VARIABLE_TYPE
         Range: [0.01, 10]
    ─────────────────────────────────────────────────────────────────
    ─────────────────────────────────────────────────────────────────
     Identifier: temperature
     Domain:
         Type: CONTINUOUS_VARIABLE_TYPE
         Range: [1, 400]
    ─────────────────────────────────────────────────────────────────
    ─────────────────────────────────────────────────────────────────
     Identifier: volume
     Domain:
         Type: CONTINUOUS_VARIABLE_TYPE
         Range: [1, 100]
    ─────────────────────────────────────────────────────────────────

Outputs:
 ──────────────────────────────────────────────────────────────────────────────
   calculate_pressure_ideal_gas-pressure
 ──────────────────────────────────────────────────────────────────────────────
```

The three required inputs (`mol`, `temperature`, `volume`) map to entity-space
dimensions. The single output, `pressure`, is the target property TRIM will
model.

## Step 2 — Define a discovery space

The file `examples/trim/example_yamls/space_pressure.yaml` defines a
three-dimensional discrete space spanning temperature (270–300 K), volume (1–10
m³), and moles (0.1–1.0 mol), with `calculate_pressure_ideal_gas` as the
measurement:

```bash
ado create space -f examples/trim/example_yamls/space_pressure.yaml --new-sample-store
```

```text
Success! Created space with identifier: space-bfed2d-19b49a
```

Inspect the space to confirm the entity and measurement spaces are correct:

<!-- markdownlint-disable MD013 -->

```bash
ado describe space --use-latest
```

```terminaloutput
Identifier: 'space-bfed2d-19b49a'

Entity Space:

   Number of entities: 1215

   Discrete properties:

      name        ┃ range      ┃ interval ┃ values
     ━━━━━━━━━━━━━╋━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━
      temperature ┃ [270, 300] ┃ 2.0      ┃ None
      volume      ┃ [1, 10]    ┃ 1.0      ┃ None
      mol         ┃ [0.1, 1]   ┃ 0.1      ┃ None


Measurement Space:

   Experiments:

      base identifier                                    ┃ required major version ┃ parameterization
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━
      custom_experiments.calculate_pressure_ideal_gas   ┃ None                   ┃ None


Sample Store identifier: 19b49a
```

<!-- markdownlint-enable MD013 -->

!!! tip

    The entity space has **discrete** dimensions, so `ado` can enumerate all
    possible entities in advance. TRIM leverages this to guide its sampling
    strategy — it knows exactly which points are unmeasured and selects the
    most informative ones to probe next.

## Step 3 — Run TRIM

The file `examples/trim/example_yamls/op_pressure.yaml` configures a TRIM
characterization run:

<!-- markdownlint-disable MD013 -->

```bash
ado create operation -f examples/trim/example_yamls/op_pressure.yaml --use-latest space
```

<!-- markdownlint-enable MD013 -->

TRIM logs its progress to the terminal as it runs. There are three distinct
stages to watch for.

### Stage 1 — No-priors characterization

Since the sample store starts empty, TRIM cannot immediately build a model. It
logs this and begins an initial characterization phase using Concatenated Latin
Hypercube Sampling (`clhs`) to collect a representative baseline:

<!-- markdownlint-disable MD013 -->

```text
2026-07-09 15:35:48,452 WARNING MainThread trim.samplers.no_priors_utils: No measured properties found in the discovery space
2026-07-09 15:35:48,470 WARNING MainThread trim.operator: Only 0 points in the source space.
Starting with no-prior characterization operation, it will sample 18 points.
```

<!-- markdownlint-enable MD013 -->

You will see output for each point being submitted and completed:

<!-- markdownlint-disable MD013 -->

```text
(RandomWalk pid=74822) Continuous batching: SUBMIT EXPERIMENT. Submitted experiment
custom_experiments.calculate_pressure_ideal_gas for mol.0.7-temperature.272-volume.1.
(RandomWalk pid=74822) Continuous Batching: EXPERIMENT COMPLETION. Received finished notification for
experiment in measurement request in group 1: request-4f70cf-...
```

<!-- markdownlint-enable MD013 -->

### Stage 2 — Iterative modeling

Once the baseline is collected, TRIM enters its main loop. In each iteration it
samples a new point guided by the current model, retrains `AutoGluon`, and
evaluates whether accuracy is still improving:

```text
(RandomWalk pid=76621) AutoGluon training complete, total runtime = 1.49s ...
Best model: WeightedEnsemble_L2 | Estimated inference throughput: 508.6 rows/s
```

After every `iterationSize` iterations (5 in `op_pressure.yaml`), TRIM checks
the stopping criterion:

<!-- markdownlint-disable MD013 -->

```text
(RandomWalk pid=10736) Testing stopping criterion after measuring 14 points, mean_ratio=... std_ratio=...
(RandomWalk pid=10736) Stopping not triggered for i=14
```

<!-- markdownlint-enable MD013 -->

### Stage 3 — Finalizing

When the stopping criterion is met or the budget is exhausted, TRIM trains one
high-quality model on all data collected under `outputDirectory_finalized/`
alongside a `model_card.json`. The `stopping_criteria_satisfied` field in that
file is `true` if TRIM converged, `false` if the budget ran out first.

<!-- markdownlint-disable MD013 -->

```text
(RandomWalk pid=10736) Stopping criteria hit after measuring 22 entities.
(RandomWalk pid=10736) Finalizing the predictive model: Fitting AutoGluon TabularPredictor on full Source Space data of 42 rows. Model will be saved in: trim_models_finalized
(RandomWalk pid=10736) Final model root_mean_squared_error=-48.72586662062896. Saving predicted model to: trim_models_finalized.
```

<!-- markdownlint-enable MD013 -->

The operation ends with a success message:

<!-- markdownlint-disable MD013 -->

```text
Success! Created operation with identifier operation-trim@2.0.3-cb3448b3 and it finished successfully.
```

<!-- markdownlint-enable MD013 -->

!!! note "Key TRIM parameters"

    The `op_pressure.yaml` file controls TRIM's behaviour:

    - **`targetOutput`** — the experiment output property to model (`pressure`)
    - **`outputDirectory`** — base directory for model artefacts; the final model
      is saved to `{outputDirectory}_finalized/`. Defaults to `trim_models`
      relative to where you run `ado create operation`.
    - **`iterationSize`** — how many points to sample before checking the
      stopping criterion
    - **`stoppingCriterion.meanThreshold` / `stdThreshold`** — model-quality
      thresholds that trigger early stopping
    - **`autoGluonArgs`** — passed directly to `AutoGluon`'s `TabularPredictor`

    See the [TRIM operator reference](../operators/trim.md) for the
    full parameter list.

## Step 4 — Inspect the results

### Sampled measurements

To see every point TRIM evaluated, along with its measured pressure value:

```bash
ado show measurements space --use-latest
```

<!-- markdownlint-disable line-length -->

```text
┌───────┬────────────────┬────────────────┬────────────────┬─────────────┬────────┬─────┬───────────────┐
│ INDEX │ identifier     │ generatorid    │ experiment_id  │ temperature │ volume │ mol │ pressure      │
├───────┼────────────────┼────────────────┼────────────────┼─────────────┼────────┼─────┼───────────────┤
│ 0     │ mol.0.1-tempe… │ no_priors_cha… │ custom_experi… │ 270         │ 1      │ 0.1 │ 224.49049068… │
│ 1     │ mol.0.1-tempe… │ no_priors_cha… │ custom_experi… │ 274         │ 2      │ 0.1 │ 113.90813786… │
│ 2     │ mol.0.1-tempe… │ no_priors_cha… │ custom_experi… │ 280         │ 3      │ 0.1 │ 77.601651101… │
│ ...   │ ...            │ ...            │ ...            │ ...         │ ...    │ ... │ ...           │
│ 72    │ mol.0.9-tempe… │ no_priors_cha… │ custom_experi… │ 298         │ 9      │ 0.9 │ 247.77098601… │
└───────┴────────────────┴────────────────┴────────────────┴─────────────┴────────┴─────┴───────────────┘
```

<!-- markdownlint-enable line-length -->

The `generatorid` column shows which sub-operation produced each measurement.
TRIM runs two internal sub-operations: one for no-priors characterization
(`no_priors_characterization`) and one for iterative modeling. You can inspect
the full sub-operation hierarchy with:

```bash
ado show related space --use-latest
```

### The saved surrogate model

TRIM saves the final `AutoGluon` model to `{outputDirectory}_finalized/`
(`trim_models_finalized/` by default). Load it in Python to make predictions
at unmeasured points:

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor.load("trim_models_finalized")

# Predict pressure for an unmeasured configuration
result = predictor.predict({"mol": 0.5, "temperature": 285, "volume": 4})
print(result)
```

!!! tip

    The surrogate model covers the entire entity space, including the points
    that were never directly measured. This is the core value of TRIM: **you pay
    for a small fraction of measurements, but get predictions everywhere**.

## Summary

| Step | What you did                                                   | `ado` concept                         |
| :--- | :------------------------------------------------------------- | :------------------------------------ |
| 1    | Installed `trim` and the custom experiment                     | Operator / custom experiment          |
| 2    | Defined a three-dimensional discrete discovery space           | Discovery space                       |
| 3    | Ran TRIM to characterize the space and train a surrogate model | Operation / `trim` operator           |
| 4    | Retrieved the sampled measurements and loaded the saved model  | `ado show measurements` / `AutoGluon` |

## Going further

Try extending this example:

- **Tune the stopping thresholds** — tighten `meanThreshold` / `stdThreshold`
  for a more accurate model (at the cost of more samples), or loosen them for a
  faster run; see `examples/trim/example_yamls/quick_exploration.yaml` for a
  pre-configured fast variant and `high_quality_characterization.yaml` for a
  high-accuracy variant
- **Budget the sampling** — add a `samplingBudget` block with `minPoints` and
  `maxPoints` to set hard limits on how many measurements TRIM can make
- **Switch the initial sampler** — set `noPriorParameters.sampling_strategy` to
  `sobol` instead of `clhs` to use Sobol sequences for the baseline
- **Improve final-model quality** — configure `finalModelAutoGluonArgs`
  separately from `autoGluonArgs` to give the final fit more time and better
  presets than the intermediate models
- **Bring your own data** — if you already have measurements in a sample store
  from a previous `random_walk` operation, TRIM will skip the no-priors phase
  and move directly to iterative modeling
- **Apply TRIM to a real workload** — replace `calculate_pressure_ideal_gas`
  with an actuator-based experiment (e.g. SFT Trainer throughput) to build a
  surrogate model for a genuinely expensive system

See the [TRIM operator reference](../operators/trim.md) for the full
configuration reference and debugging guidance.

## What's next

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable no-inline-html -->
<!-- markdownlint-disable MD046 -->
<!-- prettier-ignore-start -->

<div class="grid cards" markdown>

- :octicons-graph-24:{ .lg .middle } **Discovering important entity space dimensions**

    ---

    Use `ado` to identify which entity space dimensions most influence a
    target metric — a natural complement to TRIM's surrogate model.

    [Identify the important dimensions of a space :octicons-arrow-right-24:](lhu.md)

- :octicons-rocket-24:{ .lg .middle } **Search a space with an optimizer**

    ---

    Use `ray_tune` to drive an optimizer over a continuous space and locate the
    best configuration.

    [Search a space with an optimizer :octicons-arrow-right-24:](best-configuration-search.md)

</div>

<!-- prettier-ignore-end -->

<!-- markdownlint-enable MD046 -->
<!-- markdownlint-enable no-inline-html -->
<!-- markdownlint-enable line-length -->
