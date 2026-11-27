<!-- markdownlint-disable code-block-style -->

# The PKH Operator

!!! abstract "Characterization Operator"

    The `pkh` (**Predictive Kernel Herding**) operator selects entities from a
    finite discovery space within a fixed budget. It fits a surrogate model and
    predicts the target value for the remaining entities.

!!! tip

    For a step-by-step walkthrough on an example problem, see the
    [Surrogate Modeling with PKH Tutorial](../examples/pkh.md).

## Overview

### What does the PKH operator do?

The **PKH** (Predictive Kernel Herding) operator is a *characterize* operator
that selects an informative sample from a finite discovery space within a fixed
measurement budget. If any entities remain unmeasured, PKH uses AutoGluon to
train a surrogate model on the measurements and predict the remaining values.

PKH operates in three main steps:

1. **Gathers an initial pilot sample** (or reuses existing measurements) to fit
   the first Random Forest model.
2. **Iteratively selects points** using *kernel herding*, choosing unmeasured
   entities that best reduce the discrepancy between the sample and the full
   pool in the learned forest leaf space.
3. **Completes the results table** by predicting any missing target values.

### When should you use the PKH operator?

Use the PKH operator when you want to:

- **Build a surrogate model** for an expensive experiment across a finite grid.
- **Select a representative sample** within a strictly fixed measurement
  budget.
- **Reuse existing measurements** from prior experiments to guide new sample
  selection.
- **Obtain predictions** for every unmeasured entity in your parameter space.

### Key Requirements and Constraints

- **Finite discovery space:** Define a finite grid with at least one input
  property. All candidate entities must fit in memory. Numeric coordinates
  must be finite. Store binary values as individual coordinates.
- **Single experiment:** The measurement space must contain exactly one
  experiment (which may produce multiple output properties).
- **Numeric target:** Specify a single numeric output property in
  `targetOutput`. Each sampled entity must return a finite numeric value for
  this property.

### Comparison with Other Operators

| Operator | Type | Budget | Sampling / Strategy | Primary Deliverable |
| :--- | :--- | :--- | :--- | :--- |
| **`pkh`** | `characterize` | Fixed entity budget | Match the pool in forest leaf space | Table of measured and predicted target values |
| **[`trim`](trim.md)** | `characterize` | Adaptive up to a point limit | Stop when holdout scores stabilize or the run reaches its budget | Surrogate model and predictions |
| **[`random_walk`](random-walk.md)** | `explore` | Fixed count or all entities | Sequential or random order (Sobol or CLHS with a custom sampler) | Measurements for the sampled entities |
| **[`ray_tune`](ray-tune.md)** | `explore` | Sample or time budget, with optional early stopping | Optimize a target metric | Best observed configuration and stop reason |

## Quick Start

This example uses a current checkout of `ado` and a discovery space with at
least 20 entities and a numeric `pressure` output. The
[tutorial](../examples/pkh.md) shows how to create that space.

**1. Install the core and plugin** from the repository root:

```bash
pip install -e . -e plugins/operators/active_learning
```

!!! tip

    On Python 3.11–3.13, the base installation includes AutoGluon for both
    final predictors. If you want AutoGluon to consider CatBoost, LightGBM, and
    XGBoost as well, install the extra with
    `pip install -e "plugins/operators/active_learning[autogluon]"`.

    For remote Ray runs, install the plugin and any required model backends
    in the job's Python environment. PKH checks the predictor dependencies
    when the run needs predictions.
    [Remote package installation](../advanced/remote-execution.md#installing-python-packages-on-a-remote-ray-cluster)

**2. Save this configuration** as `operation_pkh.yaml`.

Set `targetOutput` and `numberEntities`. PKH supplies defaults for the other
parameters.

```yaml
inputs:
  discoverySpace:
    identifier: <your-space-identifier>
    kind: discoveryspace
operation:
  module:
    operationType: characterize
    operatorName: pkh
  parameters:
    targetOutput: pressure   # property to model
    numberEntities: 20       # new entities, including the pilot
```

**3. Run PKH** on the most recently created space in your active context.
`--use-latest space` fills in the space identifier in the configuration.

```bash
ado create operation -f operation_pkh.yaml --use-latest space
```

When the run finishes, use the printed PKH operation identifier to
[retrieve the results table](#working-with-pkh-output).

---

## How PKH Works

PKH uses target measurements to learn which entities resemble each other. It
then selects entities that represent this learned structure. A **label** is the
measured target value for one entity.

### 1. The Pilot Sample (Bootstrap Phase)

A **pilot sample** (or *pilot*) is an initial set of entities measured at random
to train the first Random Forest before active learning begins.

When an operation starts, PKH checks the sample store for existing target
measurements:

- **If existing measurements $\ge$ `pilotSize`:** PKH reuses these measurements
  and begins kernel herding immediately without drawing random pilot points.
- **If existing measurements $<$ `pilotSize`:** PKH selects additional
  unmeasured entities uniformly at random without replacement, using `seed`,
  until it reaches `pilotSize`.

PKH requires at least two labelled entities to fit the first forest. If the
pilot ends with one label, PKH selects one more unmeasured entity uniformly at
random.

### 2. Kernel Herding and MMD Minimization

Kernel herding selects a sample to approximate a target distribution. PKH uses
a uniform target distribution $\mathbb{P}$ over the $N$ candidate entities.
The selected sample contains $S$ entities,
$\mathcal{S} = \{\mathbf{x}_1, \dots, \mathbf{x}_S\}$.

The **Maximum Mean Discrepancy (MMD)** measures the difference between the mean
feature representations of the sample and the pool:

$$
\text{MMD}^2(\mathbb{P}, \mathcal{S}) = \left\| \mu_{\mathbb{P}} - \frac{1}{S}
\sum_{s=1}^S \Phi(\mathbf{x}_s) \right\|^2
$$

where:

- $\Phi(\mathbf{x})$ is the feature representation of entity $\mathbf{x}$.
- $\mu_{\mathbb{P}} = \frac{1}{N} \sum_{i=1}^N \Phi(\mathbf{x}_i)$ is the mean
  feature vector across all $N$ entities in the space.

### 3. Predictive Kernel Herding

PKH learns its kernel from the leaf partitions of a Random Forest regressor.
A leaf is the final node that an entity reaches in a decision tree. Entities
that reach the same leaves have similar representations. The measured target
values influence these partitions through forest training.

For an ensemble of $T$ decision trees, each entity $\mathbf{x}$ maps to a
concatenated one-hot leaf vector:

$$
\Phi(\mathbf{x}) = \frac{1}{\sqrt{T}} \left[
\mathbf{e}_{\ell_1(\mathbf{x})}^\top, \, \mathbf{e}_{\ell_2(\mathbf{x})}^\top,
\, \dots, \, \mathbf{e}_{\ell_T(\mathbf{x})}^\top \right]^\top
$$

$\ell_t(\mathbf{x})$ identifies the leaf that entity $\mathbf{x}$ reaches in
tree $t$. The vector $\mathbf{e}_{\ell}$ has one at position $\ell$ and zero
elsewhere.

At step $S + 1$, PKH selects the unmeasured entity with the highest inner
product against the discrepancy between the pool and sample means:

$$
\mathbf{x}_{S+1} = \arg\max_{\mathbf{x} \in \mathcal{X}_{\text{unmeasured}}}
\left\langle \mu_{\mathbb{P}} - \frac{1}{S} \sum_{s=1}^S \Phi(\mathbf{x}_s), \,
\Phi(\mathbf{x}) \right\rangle
$$

$\mathcal{X}_{\text{unmeasured}}$ contains the remaining unmeasured entities.

For example, consider one tree with two leaves. The pool contains ten entities,
and five entities have target measurements:

| Leaf | Pool count | Measured count | Pool fraction − sample fraction |
| :--- | :--- | :--- | :--- |
| Left | 6 | 2 | `0.6 − 0.4 = 0.2` |
| Right | 4 | 3 | `0.4 − 0.6 = −0.2` |

PKH selects an unmeasured entity in the left leaf because its score is higher.
With several trees, PKH averages each entity's leaf score across the trees.
The seed controls tie-breaking between candidates with equal scores.

### 4. Multi-Epoch Refinement and Stopping

PKH refits the forest after every `epochLength` new measurements. Between
refits, it keeps the leaf partitions and updates the sample counts.

Sampling ends when the operation selects `numberEntities` new entities or
exhausts the unmeasured pool. PKH includes existing target measurements in the
pilot and training set.

If any entities remain unmeasured, PKH fits the final predictor on the measured
targets and fills in the missing values. Otherwise, the table is already
complete.

For the method and its evaluation on LLM serving workloads, see
[Sampling Where It Matters: Predicting LLM Serving Performance](https://research.ibm.com/publications/sampling-where-it-matters-predicting-llm-serving-performance)
(EuroMLSys 2026).

---

## Configuration Parameters

### Example Configuration

This example configures PKH for the ideal-gas tutorial.

<!-- prettier-ignore-start -->

```yaml
{%
  include-markdown "../../../plugins/operators/active_learning/examples/ideal_gas/operation_pkh.yaml"
%}
```
<!-- prettier-ignore-end -->

### Parameters at a Glance

| Parameter | Type | Default | Required |
| :--- | :--- | :--- | :--- |
| [`targetOutput`](#targetoutput) | `str` | — | Yes |
| [`numberEntities`](#numberentities) | `int` | — | Yes |
| [`pilotSize`](#pilotsize) | `int` | `5` | No |
| [`epochLength`](#epochlength) | `int` | `10` | No |
| [`seed`](#seed) | `int \| None` | `42` | No |
| [`nEstimators`](#nestimators) | `int` | `100` | No |
| [`minSamplesLeaf`](#minsamplesleaf) | `int` | `2` | No |
| [`maxFeatures`](#maxfeatures) | `float \| int \| "sqrt" \| "log2" \| None` | `"sqrt"` | No |
| [`criterion`](#criterion) | `str` | `"squared_error"` | No |
| [`nJobs`](#njobs) | `int` | `1` | No |
| [`finalPredictor`](#finalpredictor) | `"random_forest"`, `"autogluon"` | `"random_forest"` | No |
| [`autoGluon`](#autogluon) | object | `None` | No |

---

Counts (`numberEntities`, `pilotSize`, `epochLength`, `nEstimators`, and
`minSamplesLeaf`) must be positive integers.

### Core Parameters

#### `targetOutput`

Select the numeric target property to model, such as `pressure`. Use its bare
name or fully qualified observed-property identifier. For an experiment with
several outputs, choose one as the target. The results table uses its bare name.
Choose distinct names for the target and input properties. PKH reserves
`entityIdentifier` and `is_predicted` for the results table.

#### `numberEntities`

This budget counts new entities, including those PKH measures during the pilot.
For example, ten previously labelled entities and a budget of twenty give PKH
up to thirty entities for training.

Set the budget at or below the total pool size. If fewer unmeasured entities
remain, PKH measures those entities and stops.

PKH selects one entity at a time and uses `random_walk` to run the space's
experiment. The child walk must produce a finite `targetOutput` measurement
for each selected entity. PKH uses that measurement to select the next entity.

#### `pilotSize`

`pilotSize` sets the initial labelled-entity count. Existing labels count toward
the threshold. PKH samples any shortfall uniformly without replacement.
Keep it below `numberEntities` when starting from an empty store to reserve
budget for herding.

#### `epochLength`

`epochLength` sets the number of new measurements between forest refits.
Smaller values update the kernel more often and require more training.

#### `seed`

An integer from `0` to `2³² − 1`, or `None`. Reproducibility assumes the same
candidate pool and prior measurements.

### Random Forest Parameters

`nEstimators`, `minSamplesLeaf`, `maxFeatures`, and `criterion` configure the
sampling forest, so they affect which entities PKH selects for measurement.

#### `nEstimators`

`nEstimators` sets the number of trees in the sampling forest.

#### `minSamplesLeaf`

Each leaf must contain at least this number of training samples.

#### `maxFeatures`

`maxFeatures` controls the features considered per split. It accepts a positive
integer count, a fraction in `(0, 1]`, `"sqrt"`, `"log2"`, or `None` for all
features. The example uses `1.0` explicitly.

#### `criterion`

`criterion` selects the regression split rule. It accepts `"squared_error"`,
`"absolute_error"`, or `"poisson"`. Poisson requires nonnegative labels with
a positive sum.

#### `nJobs`

`nJobs` sets a nonzero parallel job count for both sampling and final
prediction. Set it to `-1` to use all available processors.

### Final Predictor Parameters

#### `finalPredictor`

By default, `random_forest` uses AutoGluon to fit a Random Forest. To compare
models, choose `autogluon`. PKH selects the model with the lowest validation
root mean squared error (RMSE). Ties favor a base model over a weighted
ensemble, then use the model name. PKH requests a full-data fit for the chosen
model’s base regressors. Some families, such as `DUMMY`, can retain the
original training split or a single training fold. When the regressors report their
training row counts, the dataset label `config.metadata.labels.predictorTrainingRows`
records the smallest count among the regressors used for prediction. PKH logs
a warning if that count is below the number of measured targets.

If any entities still need predictions after sampling, either mode requires
Python 3.11–3.13 and at least four entities with finite target measurements.
If every entity has a measurement, PKH returns the table directly.

#### `autoGluon`

Both final predictor modes accept these AutoGluon settings.

- `presets`: `"medium_quality"` (default), `"good_quality"`, `"high_quality"`,
  or `"best_quality"`.
- `timeLimit`: Positive time limit for the initial fit, in seconds (default
  `60`). AutoGluon’s full-data refit runs separately.
- `hyperparameters`: Model settings passed to `TabularPredictor.fit`
  (default `None`). If you specify model families for `random_forest`, use
  only `RF`. An explicit `KNN` model uses numeric features by default and needs
  enough rows for `n_neighbors` after the preset's default validation split.
  AutoGluon validates custom splits during fitting.

## Working with PKH Output

PKH returns one row per entity in the `predictions` table of a `DataContainer`
linked to the operation. The `is_predicted` column is `True` for a model
estimate and `False` for a measurement.

The stored table preserves integer measurements and distinguishes numeric and
text coordinates such as `1` and `"1"`. Binary coordinates use ADO's
`BLOB_VALUE_TYPE` tag and a base64-encoded `value`. When a column mixes
large integers with floating-point predictions, pandas can round the integers
during type inference. To keep their exact values, export the resource to
`pkh-results.json` and load the columns with `dtype=object`.
[Export instructions](../examples/pkh.md#read-the-predictions)

```python
from pathlib import Path

import pandas as pd

from ado.core.datacontainer.resource import DataContainerResource

resource = DataContainerResource.model_validate_json(
    Path("pkh-results.json").read_text(encoding="utf-8")
)
columns = resource.config.tabularData["predictions"].data
table = pd.DataFrame(columns, dtype=object)
```

### Accessing Full Space Predictions

Because the related `random_walk` can be the most recent operation, use the
PKH operation identifier printed by `ado create operation` to find the table.
Then copy the `datacontainer-…` identifier from the output of the first command
into the second.

```bash
OPERATION_IDENTIFIER="<your PKH operation identifier>"
ado show related operation "$OPERATION_IDENTIFIER" --max-hops 1
ado describe datacontainer "<related datacontainer identifier>"
```

### MMD Statistic

The [MMD²](#2-kernel-herding-and-mmd-minimization) measures discrepancy in the
forest leaf space. A lower value means a closer match under that kernel.
Zero means that the sample and pool have identical mean embeddings.

When at least two entities have target measurements, the PKH operation records
the final value in `metadata.pkhMmdSquared`. To read it, use the same operation
identifier.

```bash
ado get operation "$OPERATION_IDENTIFIER" -o yaml
```

!!! note

    PKH records only the final value. Each forest refit changes the kernel, so
    values from different epochs use different representations and are not
    comparable.

### Inspecting Runs from the CLI

To inspect the measurements, copy the related `random_walk` operation identifier
from the same `ado show related` output. The first command below exports the
measured target values as CSV, and the second shows counts of successful and
failed requests.

```bash
RANDOM_WALK_OPERATION_IDENTIFIER="<related random_walk identifier>"
ado show measurements operation "$RANDOM_WALK_OPERATION_IDENTIFIER" -o csv
ado show stats operation "$RANDOM_WALK_OPERATION_IDENTIFIER"
```

---

## Troubleshooting

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| `targetOutput '...' must identify exactly one observed property` | The target is missing or ambiguous. | Use a unique target name or its qualified observed-property identifier. |
| `random_forest accepts only RF hyperparameters.` | The configuration specifies another model family. | Use only `RF` in `autoGluon.hyperparameters`, or choose `finalPredictor: autogluon` to compare families. |
| PKH cannot load a requested AutoGluon model family | The model family requires an additional library. | Install the package named in the error in the environment that runs PKH. |
| The discovery space has zero or multiple experiments | PKH requires exactly one experiment in the measurement space. | Select one experiment. It can return several outputs. |
| A selected entity has a missing or invalid target value | PKH needs a finite value for the selected target. | Inspect the measurement results for the related `random_walk` operation. |
| Validation error on an unknown parameter | The configuration uses a name outside the PKH parameter list. | Check the name against [Parameters at a glance](#parameters-at-a-glance). |

---

## What's next

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable no-inline-html -->
<!-- prettier-ignore-start -->

<div class="grid cards" markdown>

- :octicons-rocket-24:{ .lg .middle } **Try the PKH Tutorial**

    ---

    Compare a predicted pressure from the results table with the ideal gas law.

    [PKH tutorial :octicons-arrow-right-24:](../examples/pkh.md)

- :octicons-workflow-24:{ .lg .middle } **Also try TRIM**

    ---

    See how TRIM uses holdout scores to decide when to stop sampling.

    [TRIM operator :octicons-arrow-right-24:](trim.md)

- :octicons-search-24:{ .lg .middle } **Learn about Random Walk**

    ---

    PKH collects its measurements through a `random_walk` operation. See how
    that operator runs experiments.

    [Random Walk operator :octicons-arrow-right-24:](random-walk.md)

</div>

<!-- prettier-ignore-end -->

<!-- markdownlint-enable no-inline-html -->
<!-- markdownlint-enable line-length -->
