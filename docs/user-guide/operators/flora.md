<!-- markdownlint-disable code-block-style -->

# The FLORA Operator

!!! abstract "Characterization Operator"

    The `flora` operator selects entities from a finite discovery space within
    a fixed budget. It fits a surrogate model on the measured entities and
    uses that model to predict the target value for the remaining entities.

!!! tip

    For a step-by-step walkthrough on an example problem, see the
    [Surrogate Modeling with FLORA Tutorial](../examples/flora.md).

## Why Use FLORA?

Use FLORA when you want to concentrate a fixed measurement budget in regions
with few measurements. A Random Forest predicts the target by averaging the
predictions of many decision trees. FLORA gives priority to regions where the
trees disagree. In an acid–base titration, for example, pH can change sharply
as the added base neutralizes the acid. Disagreement can help locate such a
transition, but trees can also agree on an inaccurate prediction.

FLORA reuses existing target measurements. It saves a final model and a table
of measured and predicted values for the whole candidate pool. The pool is
every point in the finite grid of discrete values that defines the entity
space.

Each tree assigns an entity to a final node called a **leaf**. The candidate
entities, their encoded features, and their leaf assignments must fit in the
memory available to the operator.

FLORA requires exactly one experiment in the measurement space. That
experiment can return several outputs. `targetOutput` selects the numeric
output that guides sampling and model training.

### Comparison with Other Operators

| Operator | Type | Budget | Sampling / Strategy | Primary Deliverable |
| :--- | :--- | :--- | :--- | :--- |
| **`flora`** | `characterize` | Fixed entity budget | Prioritize leaves with high tree disagreement and few measurements | Surrogate model and predictions |
| **[`pkh`](pkh.md)** | `characterize` | Fixed entity budget | Match the pool in forest leaf space | Surrogate model and predictions |
| **[`trim`](trim.md)** | `characterize` | Adaptive up to a point limit | Stop when holdout scores stabilize or the budget is reached | Surrogate model and predictions |
| **[`random_walk`](random-walk.md)** | `explore` | Fixed count or all entities | Sequential or random order (Sobol or CLHS with a custom sampler) | Sampled entity observations (no surrogate model) |
| **[`ray_tune`](ray-tune.md)** | `explore` | Sample or time budget, with optional early stopping | Optimize a target metric | Best observed configuration and stop reason |

## Quick Start

This example uses a current checkout of `ado` and a discovery space with at
least 60 entities and a numeric `ph` output. The
[titration tutorial](../examples/flora.md#step-1-create-the-discovery-space)
shows how to create that space.

**1. Install the core and plugin** from the repository root:

```bash
pip install -e . -e plugins/operators/active_learning
```

!!! tip

    To use `finalPredictor: autogluon` on Python 3.11–3.13, install the optional
    extra with
    `pip install -e "plugins/operators/active_learning[autogluon]"`.
    The default, `random_forest`, needs no extra install.

**2. Save this configuration** as `operation_flora.yaml`.

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
    operatorName: flora
  parameters:
    targetOutput: ph     # property to model
    numberEntities: 60   # new entities, including the pilot
```

**3. Run FLORA** on the most recently created space in your active context.
`--use-latest space` fills in the space identifier in the configuration.

```bash
ado create operation -f operation_flora.yaml --use-latest space
```

When the run finishes, the model and predictions are in
`flora_models_finalized/`. See
[Working with FLORA output](#working-with-flora-output) for how to use them.

---

## How FLORA Works

FLORA uses the disagreement between the trees of a Random Forest and the
number of measurements in each leaf to score the remaining entities. It
selects the entity with the highest score. A **label** is the measured target
value for one entity.

### 1. The Pilot Sample (Bootstrap Phase)

FLORA first reads existing target measurements. Each entity with a target
measurement counts once toward `pilotSize`. If an entity has several
measurements of the target, FLORA uses their mean. If fewer than `pilotSize`
labelled entities are available, FLORA selects additional unmeasured entities
uniformly at random without replacement, using `seed`. These measurements
provide the initial training data for the Random Forest.

### 2. Disagreement Between Trees

FLORA fits a Random Forest of $T$ trees on the labelled entities. Let
$\hat{y}_t(\mathbf{x})$ be the prediction of tree $t$ for entity $\mathbf{x}$
and $\bar{y}(\mathbf{x})$ the mean prediction of the trees. The **disagreement**
of entity $\mathbf{x}$ is the variance of the tree predictions:

$$
d(\mathbf{x}) = \frac{1}{T} \sum_{t=1}^{T}
\left( \hat{y}_t(\mathbf{x}) - \bar{y}(\mathbf{x}) \right)^2
$$

FLORA computes $d(\mathbf{x})$ for every entity in the pool. A high value
means the individual trees give different predictions for that entity.

### 3. Leaf Gain

For each tree $t$ and leaf $\ell$, let

- $p_{t\ell}$ be the number of pool entities in the leaf.
- $c_{t\ell}$ be the number of measured entities in the leaf.
- $D_{t\ell} = \sum_{\mathbf{x} \in \ell} d(\mathbf{x})$ be the total
  disagreement of the entities in the leaf.
- $N$ be the number of entities in the pool.

FLORA treats the leaf's mean disagreement as a proxy for its prediction error.
The $p_{t\ell}$ pool entities share $c_{t\ell}$ labels, so the leaf's estimated
risk is $\frac{D_{t\ell}}{N\,c_{t\ell}}$. One more label in the leaf lowers
this risk by $\frac{D_{t\ell}}{N\,c_{t\ell}(c_{t\ell}+1)}$, and the newly
measured entity no longer needs a prediction, which removes a further
$\frac{D_{t\ell}}{N\,p_{t\ell}}$. The **gain** of the leaf is the sum of both
terms:

$$
g_{t\ell} = \frac{D_{t\ell}}{N}
\left( \frac{1}{c_{t\ell}(c_{t\ell}+1)} + \frac{1}{p_{t\ell}} \right)
$$

The gain is large for leaves with high disagreement and few labels. It falls as
labels accumulate. Every leaf of a fitted tree contains at least one measured
entity, because each leaf is built from labelled entities, so $c_{t\ell} \ge 1$.

At each step, FLORA scores every unmeasured entity by the mean gain of the
leaves it reaches, where $\ell_t(\mathbf{x})$ is its leaf in tree $t$:

$$
s(\mathbf{x}) = \frac{1}{T} \sum_{t=1}^{T} g_{t\,\ell_t(\mathbf{x})}
$$

FLORA selects $\arg\max_{\mathbf{x}} s(\mathbf{x})$ over the unmeasured
entities. When several entities have exactly the same maximum score, it draws
one of them uniformly with a seeded random generator.

For example, consider two leaves in one tree of the forest. The pool contains
ten entities, and four entities have target measurements:

| Leaf | Pool count $p$ | Measured count $c$ | Total disagreement $D$ | Gain |
| :--- | :--- | :--- | :--- | :--- |
| Left | 6 | 1 | 3.0 | `0.3 × (1/2 + 1/6) = 0.200` |
| Right | 4 | 3 | 0.8 | `0.08 × (1/12 + 1/4) = 0.027` |

This tree contributes a larger gain for candidates in its left leaf, where
one label supports six entities and disagreement is higher. FLORA averages
that contribution with each candidate's gain from the other trees.

### 4. Refits and Stopping

Between refits, FLORA keeps the fitted trees and updates the measured count of
each leaf after every new measurement. After fitting on $n$ measured entities,
FLORA refits the forest when the number of measured entities reaches

$$
n + \left\lceil \frac{n}{\ln(n + 1)} \right\rceil
$$

Starting with five entities, the first four fits use 5, 8, 12, and 17
measured entities. The interval between refits grows more slowly than the
sample, so FLORA refits often while the forest is learning quickly and less
often later. Each refit draws a new forest seed from `seed`. FLORA has no
`epochLength` parameter.

Sampling ends when the operation selects `numberEntities` new entities or
exhausts the unmeasured pool. Existing target measurements count toward the
pilot and training set, but they do not consume this budget.

FLORA then fits the final predictor on all available target measurements. It
saves the predictor, a table of measured and predicted values, and metadata.

---

## Configuration Parameters

### Example Configuration

This example sets most parameters explicitly. Values shown are typical, not
required.

<!-- prettier-ignore-start -->

```yaml
{%
  include-markdown "../../../examples/titration/example_yamls/operation_flora.yaml"
%}
```

<!-- prettier-ignore-end -->

### Parameters at a Glance

| Parameter | Type | Default | Required |
| :--- | :--- | :--- | :--- |
| [`targetOutput`](#targetoutput) | `str` | — | Yes |
| [`numberEntities`](#numberentities) | `int` | — | Yes |
| [`pilotSize`](#pilotsize) | `int` | `5` | No |
| [`seed`](#seed) | `int` | `42` | No |
| [`nEstimators`](#nestimators) | `int` | `100` | No |
| [`minSamplesLeaf`](#minsamplesleaf) | `int` | `2` | No |
| [`maxFeatures`](#maxfeatures) | `float \| int \| "sqrt" \| "log2" \| None` | `"sqrt"` | No |
| [`criterion`](#criterion) | `str` | `"squared_error"` | No |
| [`nJobs`](#njobs) | `int` | `-1` | No |
| [`finalPredictor`](#finalpredictor) | `"random_forest"`, `"autogluon"` | `"random_forest"` | No |
| [`outputDirectory`](#outputdirectory) | `str` | `"flora_models"` | No |
| [`autoGluon`](#autogluon) | object | `None` | No |

---

Counts (`numberEntities`, `pilotSize`, `nEstimators`, and `minSamplesLeaf`)
must be positive integers. FLORA rejects parameters it does not define,
including `epochLength`.

### Core Parameters

#### `targetOutput`

The numeric target property to model, such as `ph`. Use its bare name or
fully qualified observed-property identifier. A single experiment can return
several outputs, but FLORA models only the selected target. Exports use its
bare name. Choose distinct names for the target and input properties. FLORA
reserves `entityIdentifier` and `is_predicted` for the results table.

Every measurement of the target must be a finite scalar. FLORA averages repeated
measurements of the same entity and fails on a non-finite value.

#### `numberEntities`

The maximum number of new entities selected for measurement, including pilot
additions. Existing target measurements do not consume this budget. Ten existing
measurements and a budget of twenty provide up to thirty training entities.

The budget cannot exceed the total pool size. Sampling ends early if fewer
unmeasured entities remain.

FLORA selects one entity at a time and uses `random_walk` to run the space's
experiment. Each selected entity must produce a finite measurement of
`targetOutput` before FLORA selects the next entity.

#### `pilotSize`

`pilotSize` sets the initial labelled-entity count. Existing labels count toward
the threshold. FLORA samples any shortfall uniformly without replacement. Keep
it below `numberEntities` when starting from an empty store to reserve budget
for active selection. It must be an integer.

#### `seed`

An integer from `0` to `2³² − 1`. It seeds the pilot draw, the forests, and the
tie-breaking, each with an independent random stream. Reproducibility assumes
the same candidate pool and prior measurements.

### Random Forest Parameters

These parameters configure the sampling forest and the `random_forest` final
predictor.

#### `nEstimators`

Each Random Forest contains this number of trees. More trees give a smoother
disagreement estimate.

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

`nJobs` sets a nonzero parallel job count. The default `-1` uses all
processors. PKH defaults to `1`.

### Final Predictor Parameters

#### `finalPredictor`

FLORA trains this regression model on all available labels. The `"autogluon"`
backend combines candidate model types, requires the optional extra on
Python 3.11–3.13, and needs at least four measured entities. The
`"random_forest"` backend uses the standard dependencies.

#### `outputDirectory`

FLORA reserves `{outputDirectory}_finalized/` before collecting measurements.
Relative paths start from the current directory. An existing path raises
`FileExistsError`, even if it is empty. Earlier files remain untouched.

Choose a distinct `outputDirectory` before rerunning FLORA. If a run fails,
FLORA removes its empty reserved directory and retains any partial output
files.

#### `autoGluon`

`autoGluon` applies only to `finalPredictor: autogluon`. FLORA uses these
defaults when you omit it:

- `presets`: `"medium_quality"` (default), `"good_quality"`, `"high_quality"`, or
  `"best_quality"`.
- `timeLimit`: Positive training time limit in seconds, fractions allowed
  (default `60`).
- `numCpus`: Positive number of CPUs for AutoGluon (default `1`).
- `hyperparameters`: Model settings passed to `TabularPredictor.fit` (default
  `None`).

## Working with FLORA Output

After sampling, FLORA saves all artifacts under `{outputDirectory}_finalized/`
(by default, `flora_models_finalized/`):

| File | Contents |
| :--- | :--- |
| `model.pkl` | Fitted preprocessing pipeline and Random Forest (when `finalPredictor: random_forest`) |
| `autogluon/` | Trained AutoGluon predictor (when `finalPredictor: autogluon`) |
| `combined_predictions.csv` | Measured and predicted values for every entity in the space |
| `metadata.json` | Model backend, target, feature names, categorical columns, measured count, and creation time |

### Loading the Trained Model

Both backends accept raw feature values in a DataFrame with the training
property names. This example predicts the pH of a 0.1 mol/L acid after adding
25 mL of a 0.1 mol/L base.

```python
import pandas as pd

titration = pd.DataFrame(
    {
        "acid_concentration": [0.1],
        "base_concentration": [0.1],
        "titrant_volume": [25],
    }
)
```

=== "Random Forest (`model.pkl`)"

    The saved pipeline includes its fitted categorical preprocessing.

    ```python
    import pickle

    with open("flora_models_finalized/model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    predicted_ph = model.predict(titration)
    ```

=== "AutoGluon (`autogluon/`)"

    Load the predictor in the environment used for training. For example, a
    saved model that uses PyTorch also needs PyTorch to make predictions.

    ```python
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor.load("flora_models_finalized/autogluon")
    predicted_ph = predictor.predict(titration)
    ```

!!! warning

    `model.pkl` is a Python pickle file. Only load pickle files from sources you
    trust.

### Accessing Full Space Predictions

`combined_predictions.csv` contains one row per candidate entity. The
`is_predicted` column is `False` for measured values and `True` for model
predictions. These predictions remain in the output files. FLORA does not add
them to the sample store as measurements.

```python
import pandas as pd

df = pd.read_csv("flora_models_finalized/combined_predictions.csv")

# Filter evaluated points vs. surrogate predictions
measured_points = df[~df["is_predicted"]]
predicted_points = df[df["is_predicted"]]
```

### Inspecting Runs from the CLI

FLORA collects measurements through a related `random_walk` operation. Set this
variable to the `operation-flora@…` identifier printed by
`ado create operation`, and select the FLORA identifier explicitly. The related
measurement operation can be the most recently created operation.

```bash
OPERATION_IDENTIFIER="<your FLORA operation identifier>"
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
| `Active learning requires exactly 1 experiment` | The discovery space has zero or multiple experiments. | Create a space containing one experiment. That experiment can return several outputs. |
| `Target '...' requires finite scalar measurements` | A measurement of the target is `NaN`, infinite, or not a single number. | Inspect the measurement results for the related `random_walk` operation. |
| `Missing finite target measurement for '...'` | A selected entity finished without a usable target value. | Inspect the measurement results for the related `random_walk` operation. |
| `FLORA requires at least four target measurements for AutoGluon` | The budget and existing labels give fewer than four measured entities. | Raise `numberEntities`, or use `finalPredictor: random_forest`. |
| `autoGluon parameters must not be provided for random_forest` | AutoGluon settings were supplied for the Random Forest backend. | Remove `autoGluon`, or select `finalPredictor: autogluon`. |
| `Install 'ado-active-learning[autogluon]' to use finalPredictor='autogluon'` | The optional backend is unavailable. | Install the extra on Python 3.11–3.13 (see [Quick start](#quick-start)). |
| The finalized output path already exists | FLORA reserves the path before collecting measurements. | Choose a distinct `outputDirectory`. Earlier files remain untouched. |
| `Unknown category in feature '...'` when calling `model.predict` | The input contains a category value absent from the discovery space. | Use values from the space's domain. |
| Validation error on an unknown parameter | FLORA rejects parameters it does not define, so typos are caught. | Check the name against [Parameters at a glance](#parameters-at-a-glance). |

---

## What's next

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable no-inline-html -->
<!-- prettier-ignore-start -->

<div class="grid cards" markdown>

- :octicons-rocket-24:{ .lg .middle } **Try FLORA**

    ---

    Build a surrogate model for pH from an acid–base titration space.

    [FLORA tutorial :octicons-arrow-right-24:](../examples/flora.md)

- :octicons-workflow-24:{ .lg .middle } **Also try PKH**

    ---

    See how PKH selects a sample that matches the whole space in the leaf
    space of a Random Forest.

    [PKH operator :octicons-arrow-right-24:](pkh.md)

- :octicons-workflow-24:{ .lg .middle } **Also try TRIM**

    ---

    See how TRIM uses holdout scores to decide when to stop sampling.

    [TRIM operator :octicons-arrow-right-24:](trim.md)

- :octicons-search-24:{ .lg .middle } **Learn about Random Walk**

    ---

    FLORA collects its measurements through a `random_walk` operation. See how
    that operator runs experiments.

    [Random Walk operator :octicons-arrow-right-24:](random-walk.md)

</div>

<!-- prettier-ignore-end -->

<!-- markdownlint-enable no-inline-html -->
<!-- markdownlint-enable line-length -->
