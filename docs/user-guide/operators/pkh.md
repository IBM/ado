<!-- markdownlint-disable code-block-style -->

# The PKH Operator

!!! abstract "Characterization Operator"

    The `pkh` (**Predictive Kernel Herding**) operator selects entities from a
    finite discovery space within a fixed budget. It fits a surrogate model and
    predicts the target value for the remaining entities.

!!! tip

    For a step-by-step walkthrough on an example problem, see the
    [Surrogate Modeling with PKH Tutorial](../examples/pkh.md).

## Why Use PKH?

Use PKH when you need a representative sample within a fixed entity budget.
It compares entities through a Random Forest kernel, a rule based on the leaves
they reach. A representative sample under this kernel can
still produce inaccurate predictions.

PKH reuses existing target measurements. It saves a final model and a table of
measured and predicted values for the whole candidate pool. Define the entity
space as a finite grid of discrete values. Every point in the grid is a candidate.

The candidate entities, encoded features, and forest leaf assignments must fit
in the memory available to the operator.

PKH requires exactly one experiment in the measurement space. That experiment
can return several outputs. `targetOutput` selects the numeric output that
guides sampling and model training.

### Comparison with Other Operators

| Operator | Type | Budget | Sampling / Strategy | Primary Deliverable |
| :--- | :--- | :--- | :--- | :--- |
| **`pkh`** | `characterize` | Fixed entity budget | Match the pool in forest leaf space | Surrogate model and predictions |
| **[`flora`](flora.md)** | `characterize` | Fixed entity budget | Prioritize leaves with high tree disagreement and few measurements | Surrogate model and predictions |
| **[`trim`](trim.md)** | `characterize` | Adaptive up to a point limit | Stop when holdout scores stabilize or the budget is reached | Surrogate model and predictions |
| **[`random_walk`](random-walk.md)** | `explore` | Fixed count or all entities | Sequential or random order (Sobol or CLHS with a custom sampler) | Sampled entity observations (no surrogate model) |
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

    To use `finalPredictor: autogluon` on Python 3.11–3.13, install the optional
    extra with
    `pip install -e "plugins/operators/active_learning[autogluon]"`.
    The default, `random_forest`, needs no extra install.

**2. Save this configuration** as `operation_pkh.yaml`.

Only `targetOutput` and `numberEntities` are required parameters. The other
parameters have defaults.

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

When the run finishes, the model and predictions are in `pkh_models_finalized/`.
See [Working with PKH output](#working-with-pkh-output) for how to use them.

---

## How PKH Works

PKH uses target measurements to learn which entities resemble each other. It
then selects entities that represent this learned structure. A **label** is the
measured target value for one entity.

### 1. The Pilot Sample (Bootstrap Phase)

PKH first reads existing target measurements. If fewer than `pilotSize` labelled
entities are available, it selects additional unmeasured entities uniformly at
random without replacement, using `seed`. These measurements provide the initial
training data for the Random Forest. A pilot of at least two labelled entities
allows forest fitting before herding begins.

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
exhausts the unmeasured pool. Existing target measurements count toward the
pilot and training set, but they do not consume this budget.

PKH then fits the final predictor on all available target measurements. It
saves the predictor, a table of measured and predicted values, and metadata.

For the method and its evaluation on LLM serving workloads, see
[Sampling Where It Matters: Predicting LLM Serving Performance](https://research.ibm.com/publications/sampling-where-it-matters-predicting-llm-serving-performance)
(EuroMLSys 2026).

---

## Configuration Parameters

### Example Configuration

This example sets most parameters explicitly. Values shown are typical, not
required.

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
| [`outputDirectory`](#outputdirectory) | `str` | `"pkh_models"` | No |
| [`autoGluon`](#autogluon) | object | `None` | No |

---

Counts (`numberEntities`, `pilotSize`, `epochLength`, `nEstimators`, and
`minSamplesLeaf`) must be positive integers.

### Core Parameters

#### `targetOutput`

The numeric target property to model, such as `pressure`. Use its bare name or
fully qualified observed-property identifier. A single experiment can return
several outputs, but PKH models only the selected target. Exports use its bare
name.
Choose distinct names for the target and input properties. PKH reserves
`entityIdentifier` and `is_predicted` for the results table.

#### `numberEntities`

The maximum number of new entities selected for measurement, including pilot
additions. Existing target measurements do not consume this budget. Ten existing
measurements and a budget of twenty provide up to thirty training entities.

The budget cannot exceed the total pool size. Sampling ends early if fewer
unmeasured entities remain.

PKH selects one entity at a time and uses `random_walk` to run the space's
experiment. Each selected entity must produce a finite measurement of
`targetOutput` before PKH selects the next entity.

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

These parameters configure the herding kernel and the `random_forest` final
predictor.

#### `nEstimators`

Each Random Forest contains this number of trees.

#### `minSamplesLeaf`

Each leaf must contain at least this number of training samples.

#### `maxFeatures`

`maxFeatures` controls the features considered per split. It accepts a positive
integer count, a fraction in `(0, 1]`, `"sqrt"`, `"log2"`, or `None` for all
features. The example uses `1.0` explicitly.

#### `criterion`

`criterion` selects the regression split rule. It accepts `"squared_error"`,
`"absolute_error"`, `"friedman_mse"`, or `"poisson"`. Poisson requires
nonnegative labels with a positive sum.

#### `nJobs`

`nJobs` sets a nonzero parallel job count. The value `-1` uses all processors.

### Final Predictor Parameters

#### `finalPredictor`

PKH trains this regression model on all available labels. The `"autogluon"`
backend combines candidate model types and requires the optional extra on
Python 3.11–3.13. The `"random_forest"` backend uses the standard dependencies.

#### `outputDirectory`

PKH reserves `{outputDirectory}_finalized/` before collecting measurements.
Relative paths start from the current directory. An existing path raises
`FileExistsError`, even if it is empty. Earlier files remain untouched.

Choose a distinct `outputDirectory` before rerunning PKH. If a run fails, PKH
removes its empty reserved directory and retains any partial output files.

#### `autoGluon`

`autoGluon` applies only to `finalPredictor: autogluon`. PKH uses these defaults
when you omit it:

- `presets`: `"medium_quality"` (default), `"good_quality"`, `"high_quality"`, or
  `"best_quality"`.
- `timeLimit`: Positive training time limit in seconds (default `60`).
- `hyperparameters`: Model settings passed to `TabularPredictor.fit` (default `None`).

## Working with PKH Output

After sampling, PKH saves all artifacts under `{outputDirectory}_finalized/`
(by default, `pkh_models_finalized/`):

| File | Contents |
| :--- | :--- |
| `model.pkl` | Fitted preprocessing pipeline and Random Forest (when `finalPredictor: random_forest`) |
| `autogluon/` | Trained AutoGluon predictor (when `finalPredictor: autogluon`) |
| `combined_predictions.csv` | Measured and predicted values for every entity in the space |
| `metadata.json` | Model backend, target, feature names, categorical columns, measured count, creation time, PKH version, and final [MMD²](#mmd-statistic) |

### MMD Statistic

The [MMD²](#2-kernel-herding-and-mmd-minimization) measures discrepancy in the
forest leaf space. A lower value means a closer match under that kernel.
Zero means that the sample and pool have identical mean embeddings. MMD² does
not measure prediction error.

PKH reports MMD² in three places:

- Set `LOGLEVEL=INFO` to log the value at each forest refit.
- Read `mmdSquared` in `metadata.json` for the final value.
- Read `metadata.pkhMmdSquared` on the PKH operation resource.

Set this variable to the `operation-pkh@…` identifier printed by `ado create operation`:

```bash
OPERATION_IDENTIFIER="<your PKH operation identifier>"
ado get operation "$OPERATION_IDENTIFIER" -o yaml
```

Select the PKH identifier explicitly. The related measurement operation can be
the most recently created operation.

!!! note

    MMD² can increase after a selection or forest refit. Refitting also changes
    the kernel, so values from different epochs use different representations.
    The final value uses a forest fitted on all measured entities. MMD² is
    unavailable when fewer than two target measurements exist.

### Loading the Trained Model

Both backends accept raw feature values in a DataFrame with the training
property names. The saved Random Forest pipeline includes its fitted
categorical preprocessing:

=== "Random Forest (`model.pkl`)"

    ```python
    import pickle

    import pandas as pd

    with open("pkh_models_finalized/model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    configuration = pd.DataFrame([{"temperature": 280.0, "volume": 5.0, "mol": 0.5}])
    predicted_value = model.predict(configuration)
    ```

=== "AutoGluon (`autogluon/`)"

    Load the predictor in the environment used for training. For example, a
    saved model that uses PyTorch also needs PyTorch to make predictions.

    ```python
    import pandas as pd
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor.load("pkh_models_finalized/autogluon")
    configuration = pd.DataFrame([{"temperature": 280.0, "volume": 5.0, "mol": 0.5}])
    predicted_value = predictor.predict(configuration)
    ```

!!! warning

    `model.pkl` is a Python pickle file. Only load pickle files from sources you
    trust.

### Accessing Full Space Predictions

`combined_predictions.csv` contains one row per candidate entity. The
`is_predicted` column is `False` for measured values and `True` for model
predictions. These predictions remain in the output files. PKH does not add
them to the sample store as measurements.

```python
import pandas as pd

df = pd.read_csv("pkh_models_finalized/combined_predictions.csv")

# Filter evaluated points vs. surrogate predictions
measured_points = df[~df["is_predicted"]]
predicted_points = df[df["is_predicted"]]
```

### Inspecting Runs from the CLI

PKH collects measurements through a related `random_walk` operation. Use these
commands to inspect the relationship, measurements, and run statistics:

```bash
ado show related operation "$OPERATION_IDENTIFIER" --max-hops 1
```

Copy the `random_walk@…` identifier listed under `operation`. The first
command prints the measurements as CSV, including the full target values.
The second shows counts of successful and failed requests.

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
| `autoGluon parameters must not be provided for random_forest` | AutoGluon settings were supplied for the Random Forest backend. | Remove `autoGluon`, or select `finalPredictor: autogluon`. |
| `Install 'ado-active-learning[autogluon]' to use finalPredictor='autogluon'` | The optional backend is unavailable. | Install the extra on Python 3.11–3.13 (see [Quick start](#quick-start)). |
| The discovery space has zero or multiple experiments | PKH requires exactly one experiment per entity. | Create a space containing one experiment. That experiment can return several outputs. |
| A selected entity has no finite target label | Its experiment finished without a usable target value. | Inspect the measurement results for the related `random_walk` operation. |
| The finalized output path already exists | PKH reserves the path before collecting measurements. | Choose a distinct `outputDirectory`. Earlier files remain untouched. |
| Validation error on an unknown parameter | PKH rejects parameters it does not define, so typos are caught. | Check the name against [Parameters at a glance](#parameters-at-a-glance). |

---

## What's next

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable no-inline-html -->
<!-- prettier-ignore-start -->

<div class="grid cards" markdown>

- :octicons-rocket-24:{ .lg .middle } **Try the PKH Tutorial**

    ---

    Build a surrogate model for the ideal gas law, from a discovery space to a
    saved model.

    [PKH tutorial :octicons-arrow-right-24:](../examples/pkh.md)

- :octicons-workflow-24:{ .lg .middle } **Also try FLORA**

    ---

    See how FLORA targets regions where the trees of a Random Forest disagree.

    [FLORA operator :octicons-arrow-right-24:](flora.md)

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
