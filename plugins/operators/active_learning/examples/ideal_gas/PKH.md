<!-- Copyright IBM Corporation 2025, 2026 -->
<!-- SPDX-License-Identifier: MIT -->
<!-- markdownlint-disable code-block-style -->

# Build a representative surrogate with PKH

!!! abstract "Advanced tutorial"

    This walkthrough assumes you are comfortable with the core `ado`
    workflow — custom experiments, discovery spaces, and operations — and that
    you have already run at least one operation end-to-end. If you are new to
    `ado`, work through
    [Your first ado experiment][first-experiment] first.

When each measurement is expensive — benchmarking an LLM deployment, a
training run, or a physical experiment — you can usually afford to measure only
a small fraction of a configuration space. The `pkh` operator (**Predictive
Kernel Herding**) chooses that fraction so that it represents the whole space.
It trains a Random Forest on the measurements collected so far and, one
configuration at a time, measures where the sample is most under-represented
according to the forest. When the budget is spent, it saves a predictor that
estimates the target at every configuration you did not measure.
PKH was initially introduced to predict LLM serving performance.[^pkh]

!!! tip "PKH, FLORA, or TRIM?"

    All three operators build a surrogate model from a subset of measurements.
    They differ in what drives the next measurement:

    - **PKH** measures where the sample is most under-represented. Choose it
      when you need measurements whose distribution mirrors the whole space.
    - **[FLORA][flora-guide]** prioritizes measurements using an estimated
      risk-reduction score based on forest disagreement. Choose it when you
      want to target predictive accuracy for one target with a fixed budget.
    - **[TRIM][trim-guide]** retrains an AutoGluon model as it samples and can
      stop early when its holdout-score stability test is satisfied. Choose it
      when you would rather not fix the number of measurements in advance.

This walkthrough uses the `ideal_gas_pressure` custom experiment — a simple
Ideal Gas Law calculation — so the focus stays on `ado` and PKH mechanics
rather than the domain. You will measure **20 of 1,215 configurations** and use
the saved predictor to estimate pressure for the **other 1,195**.

We will:

1. Install the `pkh` operator and the example custom experiment
2. Inspect the experiment interface
3. Define a discovery space for pressure across temperature, volume, and moles
4. Run PKH to collect a representative sample and save a predictor
5. Inspect the measurements and their sampling order, and predict the rest

!!! warning "Prerequisites"

    - A clone of the `ado` repository — the `pkh` operator and the example
      experiment are installed from source
    - Python 3.11–3.14
    - An active `ado` context — the default `local` context works; see
      [Working with contexts][contexts]
      to create or switch contexts

    Clone the repository if you have not already done so:

    ```bash
    git clone https://github.com/ibm/ado.git
    cd ado
    ```

    Then install `ado`, the operator, and the example experiment:

    ```bash
    python -m pip install -e . \
      -e plugins/operators/active_learning \
      -e plugins/operators/active_learning/examples/ideal_gas/custom_experiments
    ```

    !!! note "Python version compatibility"

        The default Random Forest predictor works on Python 3.11–3.14. The
        optional [AutoGluon final predictor](#choose-the-final-predictor)
        requires Python 3.11–3.13.

    All commands in this walkthrough are run from the **repository root**
    (`ado/`).

!!! example "TL;DR"

    Once the packages are installed:

    <!-- markdownlint-disable MD013 -->
    ```bash
    ado create space -f plugins/operators/active_learning/examples/ideal_gas/space_pressure.yaml --new-sample-store
    ado create operation -f plugins/operators/active_learning/examples/ideal_gas/operation_pkh.yaml --use-latest space
    ado show related space --use-latest
    ado show measurements space --use-latest
    ```
    <!-- markdownlint-enable MD013 -->

## Step 1 — Install the required packages

### The `pkh` operator

The `pkh` operator is distributed in the `ado-active-learning` plugin.
The prerequisites command installed it; to install it on its own, run:

```bash
python -m pip install -e plugins/operators/active_learning
```

Confirm it is registered:

```bash
ado get operators
```

You should see `pkh` listed as a `characterize` operator:

```text
Available operators by type:
┌───────┬─────────────┬─────────┬──────────────┐
│ INDEX │ OPERATOR    │ VERSION │ TYPE         │
├───────┼─────────────┼─────────┼──────────────┤
│ 0     │ pkh         │ 0.1.0   │ characterize │
│ 1     │ random_walk │ 2.1.0   │ explore      │
└───────┴─────────────┴─────────┴──────────────┘
```

The list also includes any other operator plugins you have installed.

### The `ideal_gas_pressure` custom experiment

The example ships a custom experiment that computes gas pressure from the
[Ideal Gas Law](https://en.wikipedia.org/wiki/Ideal_gas_law): `P = nRT/V`.

Install it from the example directory:

```bash
python -m pip install -e \
  plugins/operators/active_learning/examples/ideal_gas/custom_experiments
```

Confirm `ado` can see it:

```bash
ado get experiments
```

You should see `ideal_gas_pressure` listed under `custom_experiments`.

Inspect the experiment interface:

```bash
ado describe experiment ideal_gas_pressure
```

```text
Identifier: custom_experiments.ideal_gas_pressure

Required Inputs:

   Constitutive Properties:
    ────────────────────────────────────────────────────────────────────
     Identifier: mol
     Domain:

        Type: CONTINUOUS_VARIABLE_TYPE
        Range: [0.01, 10]

    ────────────────────────────────────────────────────────────────────
    ────────────────────────────────────────────────────────────────────
     Identifier: temperature
     Domain:

        Type: CONTINUOUS_VARIABLE_TYPE
        Range: [1, 400]

    ────────────────────────────────────────────────────────────────────
    ────────────────────────────────────────────────────────────────────
     Identifier: volume
     Domain:

        Type: CONTINUOUS_VARIABLE_TYPE
        Range: [1, 100]

    ────────────────────────────────────────────────────────────────────

Outputs:
 ──────────────────────────────────────────────────────────────────────
   ideal_gas_pressure-pressure
 ──────────────────────────────────────────────────────────────────────
```

The three required inputs (`mol`, `temperature`, `volume`) map to entity-space
dimensions. The single output, `pressure` (in pascals), is the target property
PKH will sample for and model.

## Step 2 — Define a discovery space

The file
`plugins/operators/active_learning/examples/ideal_gas/space_pressure.yaml`
defines a three-dimensional discrete space with `ideal_gas_pressure` as the
measurement:

| Dimension     | Values             | Count |
| :------------ | :----------------- | ----: |
| `temperature` | 270, 272, …, 298 K |    15 |
| `volume`      | 1, 2, …, 9 m³      |     9 |
| `mol`         | 0.1, 0.2, …, 0.9   |     9 |

That makes 15 × 9 × 9 = 1,215 configurations. Create the space with a new,
empty sample store:

```bash
ado create space \
  -f plugins/operators/active_learning/examples/ideal_gas/space_pressure.yaml \
  --new-sample-store
```

```text
Success! Created space with identifier: space-3c031e-0326be
```

The outputs below are excerpts from an example run. Resource and process IDs
vary between runs; sampling choices can also vary with library versions.

Inspect the space to confirm the entity and measurement spaces are correct:

```bash
ado describe space --use-latest
```

<!-- markdownlint-disable MD013 -->

```text
Identifier: 'space-3c031e-0326be'

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

      base identifier                       ┃ required major version ┃ parameterization
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━
      custom_experiments.ideal_gas_pressure ┃ None                   ┃ None

    ────────────────────────────────── ideal_gas_pressure ──────────────────────────────────
     Expected Interface

     Inputs:

        parameter   ┃ type     ┃ value ┃ parameterized
       ━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━╋━━━━━━━━━━━━━━━
        mol         ┃ required ┃ None  ┃ na
        temperature ┃ required ┃ None  ┃ na
        volume      ┃ required ┃ None  ┃ na

     Outputs:

        target property
       ━━━━━━━━━━━━━━━━━
        pressure

    ────────────────────────────────────────────────────────────────────────────────────────


Sample Store identifier: 0326be
```

<!-- markdownlint-enable MD013 -->

The `range` column is half-open: `temperature` stops at 298 K, and `volume` and
`mol` stop at 9 and 0.9. The following commands use `--use-latest` to select
this space.

!!! tip

    The entity space has **discrete** dimensions, so `ado` can enumerate every
    configuration in advance. PKH relies on this: to find under-represented
    regions, it compares the measured sample with the full pool of 1,215
    candidates. PKH requires a finite, discrete entity space, so give any
    continuous dimension an `interval` or an explicit list of `values`.

## Step 3 — Run PKH

The file
`plugins/operators/active_learning/examples/ideal_gas/operation_pkh.yaml`
configures a PKH characterization run. These are its main settings:

```yaml
operation:
  module:
    operationType: characterize
    operatorName: pkh
  parameters:
    targetOutput: pressure # the experiment output to model
    numberEntities: 20 # new measurements, pilot included
    pilotSize: 5 # measurements before herding starts
    pilotStrategy: random
    seed: 42
    outputDirectory: pkh_models # model saved to pkh_models_finalized/
```

Start the operation:

```bash
ado create operation \
  -f plugins/operators/active_learning/examples/ideal_gas/operation_pkh.yaml \
  --use-latest space
```

PKH runs as a `characterize` operation that drives a nested `random_walk`
operation: PKH decides which configuration to measure next, and the nested
operation submits the experiment and records the result. You will see both
operations start:

```text
=========== Starting Operation operation-pkh@0.1.0-3c1128b5 ===========
...
=========== Starting Operation random_walk@2.1.0-c04584 ===========
```

There are three distinct stages to the run.

### Stage 1 — Pilot measurements

The sample store starts empty, so PKH has no measurements to train a forest
on. It first measures `pilotSize` configurations — five here — chosen
uniformly at random. Measurements are sequential: PKH waits for each result
before it requests the next configuration. You will see a submission and a
completion line for each one:

<!-- markdownlint-disable MD013 -->

```text
(_SequentialPredictiveWalk pid=49906) Continuous batching: SUBMIT EXPERIMENT. Submitted experiment
custom_experiments.ideal_gas_pressure for mol.0.2-temperature.282-volume.6. Request identifier: 97cf00
(_SequentialPredictiveWalk pid=49906) Continuous Batching: EXPERIMENT COMPLETION. Received finished
notification for experiment in measurement request in group 1: request-97cf00-...
```

<!-- markdownlint-enable MD013 -->

Pilot measurements count toward the total budget of 20. To tell pilot
measurements from the ones chosen by herding afterwards, use the
[sampling order](#sampling-order) in Step 4.

### Stage 2 — Kernel herding

With the pilot results in hand, PKH fits a Random Forest that predicts
pressure from the three inputs. Each tree partitions the space into
**leaves**: groups of configurations that the tree treats as alike for
predicting pressure. Configurations that share a leaf in many trees are close
in the forest's eyes, and this leaf co-occurrence is the kernel PKH works with.

For every tree, PKH keeps two histograms over its leaves: how the full pool of
1,215 configurations is spread across them, and how the measured sample is
spread. It measures the unmeasured configuration whose leaves are most
under-represented in the sample, averaged over all trees, adds it to the sample
histograms, and repeats. This is kernel herding, which aims to reduce the
Maximum Mean Discrepancy (MMD) between the sample and the pool under the forest
kernel. With leaf assignments cached and a fixed number of trees, scoring takes
linear time in the number of candidates, rather than comparing every pair of
configurations.

The forest is refitted every `epochLength` measurements (10 by default);
between refits, PKH only updates its sample histograms. In this run, the forest
is fitted on 5 and then 15 measurements, and sampling stops when the budget of
20 is reached.

??? info "How PKH scores a candidate"

    For each tree `t`, let `leaf_t(x)` be the leaf that contains configuration
    `x`. PKH measures next the unmeasured configuration with the highest score:

    ```text
    score(x) = mean over trees t of
               [pool share of leaf_t(x) − sample share of leaf_t(x)]
    ```

    The **pool share** of a leaf is the fraction of all configurations in the
    space that fall in it; the **sample share** is the fraction of measured
    configurations that do. A high score marks a configuration in regions the
    sample has not yet covered in proportion to their size.

### Stage 3 — Fit and save the final predictor

After the last measurement, PKH fits the final predictor on all the measured
configurations — here, a Random Forest on the 20 measurements — and saves it
under `{outputDirectory}_finalized/`:

```text
pkh_models_finalized/
├── model.pkl
└── metadata.json
```

`model.pkl` holds the predictor and its input schema; `metadata.json` records
the target, the input features, and the number of measurements supplied.

The operation ends with a success message:

<!-- markdownlint-disable MD013 -->

```text
Success! Created operation with identifier operation-pkh@0.1.0-3c1128b5 and it finished successfully.
```

<!-- markdownlint-enable MD013 -->

!!! warning "Running the operation again"

    Before you repeat the operation, set a new `outputDirectory` in the
    operation file: its `_finalized` directory must not already exist.

!!! note "Key PKH parameters"

    The `operation_pkh.yaml` file controls PKH's behaviour:

    - **`targetOutput`** — the experiment output to model (`pressure`); it
      must return a single finite number
    - **`numberEntities`** — how many new configurations to measure, pilot
      included (`20`)
    - **`pilotSize`** — how many measured configurations PKH needs before
      herding starts, existing ones included (default `5`)
    - **`pilotStrategy`** — how new pilot configurations are chosen: `random`
      (default), `clhs`, or `sobol`
    - **`epochLength`** — how many measurements to make between forest refits
      (default `10`)
    - **`finalPredictor`** — the model saved at the end: `random_forest`
      (default) or `autogluon`
    - **`outputDirectory`** — base path for the final predictor, which is
      saved to `{outputDirectory}_finalized/`
    - **`seed`** — controls randomness in the pilot, tie-breaking, the
      forests, and the final model

    See the [parameter reference](#parameter-reference) for the full list.

## Step 4 — Inspect the results

### Sampled measurements

To see every configuration PKH measured, along with its pressure:

```bash
ado show measurements space --use-latest
```

<!-- markdownlint-disable line-length -->

```text
┌───────┬──────────────────┬─────────────────┬──────────────────┬─────────────┬────────┬─────┬─────────────────┐
│ INDEX │ identifier       │ generatorid     │ experiment_id    │ temperature │ volume │ mol │ pressure        │
├───────┼──────────────────┼─────────────────┼──────────────────┼─────────────┼────────┼─────┼─────────────────┤
│ 0     │ mol.0.7-tempera… │ explicit_grid_… │ custom_experime… │ 292         │ 5      │ 0.7 │ 339.89523182384 │
│ 1     │ mol.0.8-tempera… │ explicit_grid_… │ custom_experime… │ 280         │ 3      │ 0.8 │ 620.8132088106… │
│ 2     │ mol.0.2-tempera… │ explicit_grid_… │ custom_experime… │ 288         │ 8      │ 0.2 │ 59.8641308496   │
│ ...   │ ...              │ ...             │ ...              │ ...         │ ...    │ ... │ ...             │
│ 19    │ mol.0.3-tempera… │ explicit_grid_… │ custom_experime… │ 278         │ 3      │ 0.3 │ 231.1420607804… │
└───────┴──────────────────┴─────────────────┴──────────────────┴─────────────┴────────┴─────┴─────────────────┘
```

<!-- markdownlint-enable line-length -->

Export them to CSV for further analysis:

```bash
ado show measurements space --use-latest -o csv \
  --output-file pkh-ideal-gas-measurements.csv
```

### Sampling order

The measurements table shows _what_ was measured, not _when_. The order lives
in the measurement requests, which PKH's nested `random_walk` operation made on
its behalf. List both operations with:

```bash
ado show related space --use-latest
```

```text
operation
  - operation-pkh@0.1.0-3c1128b5
  - random_walk@2.1.0-c04584
samplestore
  - 0326be
```

`operation-pkh@…` is the operation you created; `random_walk@…` is the nested
operation that submitted the measurements. Export the space's
measurement-request trace:

```bash
ado show trace space --use-latest -o json \
  --output-file pkh-ideal-gas-trace.json
```

Each request records its position in the sequence (`Request Index`), the
configuration it measured (`Entity IDs`), and its `Status`. On a fresh sample
store, the first `pilotSize` requests are the random pilot and the rest were
chosen by herding:

```python
import pandas as pd

PILOT_SIZE = 5  # pilotSize in operation_pkh.yaml

trace = pd.read_json("pkh-ideal-gas-trace.json").sort_values("Request Index")
is_pilot = trace["Request Index"] < PILOT_SIZE
trace["stage"] = is_pilot.map({True: "pilot", False: "herding"})
print(trace[["Request Index", "stage", "Entity IDs"]].to_string(index=False))
```

```text
 Request Index   stage                         Entity IDs
             0   pilot [mol.0.3-temperature.292-volume.6]
             1   pilot [mol.0.2-temperature.282-volume.6]
             ...
             4   pilot [mol.0.5-temperature.282-volume.5]
             5 herding [mol.0.7-temperature.292-volume.5]
             6 herding [mol.0.6-temperature.280-volume.2]
             ...
            19 herding [mol.0.3-temperature.278-volume.3]
```

### The saved surrogate model

PKH saved the final predictor to `pkh_models_finalized/`. Use it to estimate
pressure at the configurations you did not measure. First, export their inputs
from the same space:

```bash
ado show measurements space --use-latest --include unmeasured -o csv \
  --output-file pkh-ideal-gas-unmeasured.csv
```

Then load the predictor in the same Python environment and predict from the
original input columns:

```python
import pandas as pd

from active_learning.regression import load_predictor

inputs = pd.read_csv("pkh-ideal-gas-unmeasured.csv", index_col=0)
predictor = load_predictor("pkh_models_finalized")
inputs["predicted_pressure"] = predictor.predict(inputs)
inputs.to_csv("pkh-ideal-gas-predictions.csv", index=False)
```

`index_col=0` consumes the row index written by the CSV export, so the
predictor sees only the `temperature`, `volume`, and `mol` columns.
`pkh-ideal-gas-predictions.csv` contains the other 1,195 configurations and
their `predicted_pressure` in pascals.

!!! tip

    The predictor covers the entire entity space, including the 1,195
    configurations that were never measured: **you pay for 20 measurements and
    get an estimate everywhere**. With a sample this small, treat the estimates
    as rough — increasing `numberEntities` can improve accuracy. The
    measured values stay in `pkh-ideal-gas-measurements.csv`.

## Summary

| Step | What you did                                                       | `ado` concept                              |
| :--- | :----------------------------------------------------------------- | :----------------------------------------- |
| 1    | Installed `pkh` and the custom experiment                          | Operator / custom experiment               |
| 2    | Defined a three-dimensional discrete discovery space               | Discovery space                            |
| 3    | Ran PKH: a random pilot, kernel herding, and a final predictor fit | Operation / `pkh` operator                 |
| 4    | Inspected the measurements and their order, and predicted the rest | `ado show measurements` / `ado show trace` |

## Going further

Try extending this example:

- **Use AutoGluon for the final model** — set `finalPredictor: autogluon` to
  fit an AutoGluon ensemble on the measurements instead of a single forest; see
  [Choose the final predictor](#choose-the-final-predictor).
- **Switch the pilot design** — set `pilotStrategy` to `clhs`, which cycles
  through each input's discrete levels, or to `sobol`, a scrambled Sobol design
  mapped to the grid. Like `random`, both use `seed` and pick distinct
  unmeasured configurations.
- **Change the refit interval** — a smaller `epochLength` refits the forest
  more often, so its leaves reflect the latest measurements; a larger one saves
  fitting time on long runs.
- **Resume from existing measurements** — target measurements already in the
  sample store count toward `pilotSize`, so PKH makes fewer pilot measurements,
  or none. A resumed run adds up to `numberEntities` new measurements and fits
  the final predictor on old and new measurements together. Use a fresh sample
  store for an independent run.
- **Apply PKH to a real workload** — replace `ideal_gas_pressure` with an
  actuator-based experiment such as
  [`vllm_performance`][vllm-actuator], and set
  `targetOutput` to an output such as `output_throughput` or `mean_ttft_ms`.
  Any finite, discrete space whose target returns a single finite number works.

### Choose the final predictor

PKH always uses its own Random Forest to choose measurements. `finalPredictor`
only selects the model that is fitted and saved after measurement collection
ends.

=== "Random Forest (default)"

    `finalPredictor: random_forest` fits a Random Forest on every measured
    configuration, using the forest settings from the operation file
    (`nEstimators`, `minSamplesLeaf`, `maxFeatures`, and `criterion`) and
    `seed`. It needs no extra dependencies and works on Python 3.11–3.14.

=== "AutoGluon"

    AutoGluon requires Python 3.11–3.13 and the plugin's optional
    dependencies:

    ```bash
    python -m pip install -e 'plugins/operators/active_learning[autogluon]'
    ```

    Then set these fields under `operation.parameters` in `operation_pkh.yaml`:

    ```yaml
    finalPredictor: autogluon
    autoGluon:
      presets: medium_quality
      timeLimit: 60
      numCpus: 1
    ```

    The values shown are the defaults:

    - **`presets`** — `medium_quality`, `good_quality`, `high_quality`, or
      `best_quality`
    - **`timeLimit`** — approximate training budget, in seconds
    - **`numCpus`** — how many CPUs AutoGluon may use
    - **`hyperparameters`** — optional; choose the candidate models. For
      presets and model configurations, see the
      [AutoGluon fit options](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html).

    The operation's `seed` also controls the AutoGluon learner. Results can
    vary with library versions and time-limited training. AutoGluon needs at
    least four usable target measurements. The default
    `medium_quality` preset holds back part of the measurements for internal
    validation; bagging presets use cross-validation and may refit on all
    rows. `metadata.json` records how many rows were supplied, used for
    fitting, and used for validation.

    An AutoGluon predictor adds an `autogluon/` subdirectory to
    `pkh_models_finalized/`. Keep the whole directory together when you move
    or load it; the same `load_predictor` code works for both predictors.

### Parameter reference

Set these under `operation.parameters` in the operation file. The first table
shows the values this walkthrough uses:

| Parameter         | This example    | Description                                                                                                                                                                       |
| :---------------- | :-------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `targetOutput`    | `pressure`      | Required. Experiment output to sample for and model; it must return a single finite number.                                                                                       |
| `numberEntities`  | `20`            | Required. New configurations to measure, pilot included. A budget smaller than the pilot ends the run during the pilot; sampling also stops once every configuration is measured. |
| `pilotSize`       | `5`             | Measured configurations needed before herding starts (default `5`). Existing target measurements count toward it.                                                                 |
| `pilotStrategy`   | `random`        | Design for new pilot configurations: `random` (default), `clhs`, or `sobol`.                                                                                                      |
| `epochLength`     | `10`            | Measurements between forest refits (default `10`).                                                                                                                                |
| `seed`            | `42`            | Seed for the pilot, the forests, and the final learner.                                                                                                                           |
| `finalPredictor`  | `random_forest` | Model saved after collection: `random_forest` (default) or `autogluon`.                                                                                                           |
| `autoGluon`       | not used        | `presets`, `timeLimit`, `numCpus`, and `hyperparameters` for the AutoGluon predictor.                                                                                             |
| `outputDirectory` | `pkh_models`    | Base path for the final predictor, which is saved to `{outputDirectory}_finalized/`.                                                                                              |

The forest and scheduling settings rarely need changing. The example sets
`nJobs: 1`; its parameter default is `-1`:

| Parameter                  | Default         | Description                                                                                                     |
| :------------------------- | :-------------- | :-------------------------------------------------------------------------------------------------------------- |
| `nEstimators`              | `100`           | Trees in each forest.                                                                                           |
| `minSamplesLeaf`           | `2`             | Minimum number of measured configurations in a leaf.                                                            |
| `maxFeatures`              | `sqrt`          | Encoded features considered at each split: `sqrt`, `log2`, an integer, a fraction in (0, 1], or `null` for all. |
| `criterion`                | `squared_error` | Split criterion: `squared_error`, `absolute_error`, `friedman_mse`, or `poisson`.                               |
| `nJobs`                    | `-1`            | Parallel forest workers; `-1` uses all available CPUs.                                                          |
| `labelWaitTimeoutSeconds`  | `300.0`         | Maximum time polling for the previous target value.                                                             |
| `labelPollIntervalSeconds` | `0.1`           | Interval between checks for the previous target value.                                                          |

## What's next

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable no-inline-html -->
<!-- markdownlint-disable MD046 -->
<!-- prettier-ignore-start -->

<div class="grid cards" markdown>

- :octicons-git-compare-24:{ .lg .middle } **Target predictive accuracy with FLORA**

    ---

    Read about the sibling operator, which prioritizes measurements using an
    estimated risk-reduction score based on forest disagreement, and compare
    its approach with PKH's.

    [Build a surrogate model with FLORA :octicons-arrow-right-24:][flora-guide]

- :octicons-graph-24:{ .lg .middle } **Stop sampling automatically with TRIM**

    ---

    Let TRIM decide how many measurements to make, retraining an AutoGluon
    model until its holdout-score stability test is met.

    [Quickly building a predictive model :octicons-arrow-right-24:][trim-guide]

- :octicons-server-24:{ .lg .middle } **Benchmark LLM serving configurations**

    ---

    Measure real vLLM deployment configurations with the `vllm_performance`
    actuator: the kind of space PKH was designed for.

    [Exploring vLLM deployment configurations :octicons-arrow-right-24:][vllm-guide]

</div>

<!-- prettier-ignore-end -->
<!-- markdownlint-enable MD046 -->
<!-- markdownlint-enable no-inline-html -->
<!-- markdownlint-enable line-length -->

[^pkh]:
    E. Aydar, C. Pinto, S. Venugopal, and D. Chatzopoulos. "Sampling Where It
    Matters: Predicting LLM Serving Performance." In _Proceedings of the Sixth
    European Workshop on Machine Learning and Systems (EuroMLSys '26)_, ACM,
    2026, pp. 13–22.
    [doi:10.1145/3805621.3807633](https://doi.org/10.1145/3805621.3807633)

<!-- markdownlint-disable MD053 -->
[first-experiment]: https://ibm.github.io/ado/latest/user-guide/examples/tutorials/density-example/
[contexts]: https://ibm.github.io/ado/latest/resources/metastore/#working-with-contexts
[flora-guide]: https://ibm.github.io/ado/latest/user-guide/examples/flora/
[trim-guide]: https://ibm.github.io/ado/latest/user-guide/examples/trim/
[vllm-actuator]: https://ibm.github.io/ado/latest/user-guide/actuators/vllm-performance/
[vllm-guide]: https://ibm.github.io/ado/latest/user-guide/examples/vllm-performance-full/
<!-- markdownlint-enable MD053 -->
