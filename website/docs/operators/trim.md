# TRIM Operator: Configuration Guide

> [!TIP]
>
> For a quick introduction to TRIM, see this
> [example](https://ibm.github.io/ado/examples/trim/). This guide is for users
> who want to adapt TRIM's behavior for their specific use case.
>
> If you want to skip to the configuration options go to the
> [complete parameter reference](#complete-parameter-reference).

## Overview

The **TRIM** (Transfer Refined Iterative Modeling) operator is a
_characterize_ operator that intelligently samples a `discoveryspace` to
build a predictive model for a target property. Unlike simple random
sampling, TRIM uses active learning to focus measurements on the most
informative points, minimizing the number of experiments needed to
achieve good model accuracy.

### What does the TRIM operator do?

TRIM builds a machine learning model (using AutoGluon) that predicts a
target property across your entire `discoveryspace`. It does this by:

1. **Assessing existing data** in your space
2. **Gathering initial samples** if needed (no-priors characterization)
3. **Iteratively sampling** ensuring uniform coverage of the space,
   prioritizing uniformity on the most informative features
4. **Training intermediate models** and evaluating the
   expected improvement upon sampling additional points
5. **Producing a final model** trained on all collected data when the
   stopping criterion is met

### When should you use the TRIM operator?

Use TRIM when you want to:

- **Build a predictive model** for a target property across a large
  parameter space
- **Minimize measurement costs** by sampling only the most informative
  points
- **Automatically stop** when additional measurements provide
  diminishing returns
- **Characterize high-dimensional spaces** efficiently

TRIM is particularly valuable when:

- Measurements are expensive or time-consuming
- Your parameter space is large (hundreds to thousands of possible
  configurations)
- You want a model that can predict unmeasured points
- You need to balance exploration and exploitation

## How TRIM Works: The Three-Phase Workflow

Understanding TRIM's internal workflow helps you configure it effectively.

### Phase 1: Initial State Assessment

When you launch a TRIM operation, it first inspects your `discoveryspace`:

1. Counts how many entities already have measured values for your
   `targetOutput`
2. Compares this count against `samplingBudget.minPoints`
3. Decides whether to proceed directly to iterative modeling or gather
   more initial data via no-priors characterization

### Phase 2: No-Priors Characterization (Conditional)

This phase runs **only if** existing measured points <
`samplingBudget.minPoints`.

**Goal:** Build a small initial dataset with uniform coverage of the
parameter space.
**Why it matters:** Starting with good initial coverage ensures the
iterative modeling phase has a solid foundation.

**How it works:**

- Uses a space-filling sampling strategy (Latin Hypercube, Sobol,
  etc.)
- Samples until `samplingBudget.minPoints` is reached

**Why it matters:** Starting with good initial coverage ensures the
iterative modeling phase has a solid foundation.

### Phase 3: Iterative Modeling

Once sufficient initial data exists, TRIM begins its main loop:

#### Step 1: Feature Importance Analysis

- Trains a temporary AutoGluon model on all available data
- Determines which input features (constitutive properties)
  are most important for predicting the target
- Keeps only the top `independentVariablesToKeep` features

#### Step 2: Smart Point Ordering

- Sorts all unmeasured points in the space
- Uses feature importance to prioritize a uniform projection of points
  on the most important feature dimensions

#### Step 3: Iterative Sampling and Training

For each iteration:

1. **Sample**
2. **Update** the training+validation and holdout datasets with new
   results
3. **Train** a new intermediate AutoGluon model
4. **Evaluate** model performance on the holdout set
5. **Check** stopping criterion

> [!NOTE] The holdout set
>
> The holdout set is made up of the last `holdoutSize` points that
> have been sampled.

#### Step 4: Stopping

After every point:

- Compares model performance between the last two non-overlapping windows
  of `iterationSize` points. Here's what this means:
  - TRIM trains a new model after sampling each new point
  - Each model is evaluated on its holdout set, producing a performance score
  - A "window" contains `iterationSize` consecutive performance scores
  - The "recent window" contains scores from the last `iterationSize` models
  - The "previous window" contains scores from the `iterationSize` models
    before that
  - Models trained later have more training data (higher training size)
- Calculates mean and standard deviation ratios,
  e.g. the mean performance metric
  of the last window divided by the mean performance metric of the previous one.
- Stops if both ratios fall below their thresholds (improvement has plateaued)
- Otherwise, continues sampling

#### Step 5: Final Model Generation

When the stopping criterion is met (or `samplingBudget.maxPoints` is
reached):

- Trains one final, high-quality AutoGluon model
- Uses **all** data collected across all phases
- Saves to `finalModelAutoGluonArgs.tabularPredictorArgs.path` if specified,
  or to `outputDirectory` with `_finalized` suffix if not specified.
- This is your production-ready predictive model

---

## Complete Parameter Reference

All parameters are configured under the `parameters` key in your
operation YAML.

### Core Configuration

#### `targetOutput`

**Type:** `str` (required)

**Purpose:** The measured property you want to predict. This is your
"y" variable.

**Tuning Guidance:** Must exactly match an output property identifier
from your experiment. All of TRIM's logic revolves around this target.

**Example:**

```yaml
parameters:
  targetOutput: pressure
```

#### `outputDirectory`

**Type:** `str | None`

**Default:** `None`

**Purpose:** Directory where AutoGluon models are saved. The final
model is saved in a subfolder with `_finalized` suffix.

**Tuning Guidance:** Always set this explicitly. If not set, models
may be saved to a temporary location and lost.

**Example:**

```yaml
parameters:
  outputDirectory: trim_models
```

---

### Phase 1: Initial Data Gathering

#### `samplingBudget`

Controls the overall sampling constraints.

##### `samplingBudget.minPoints`

**Type:** `int`

**Default:** `18`

**Purpose:** Minimum number of measured points required before
iterative modeling begins. If fewer points exist, triggers no-priors
characterization.

**Tuning Guidance:**

- **Higher values** (e.g., 30-50): More robust initial dataset, better for
  high-dimensional spaces
- **Lower values** (e.g., 10-15): Faster start, but may lead to poor
  initial models
- **Rule of thumb:** Set to at least 2× the number of input features

**Example:**

```yaml
parameters:
  samplingBudget:
    minPoints: 25
```

##### `samplingBudget.maxPoints`

**Type:** `int`

**Default:** `40`

**Purpose:** Hard cap on total new points to measure. Acts as a
cost-control backstop.

**Tuning Guidance:**

- Set based on your measurement budget
- Operation stops when this limit is reached, regardless of model
  performance
- Should be significantly larger than `minPoints`
  to allow iterative improvement. Aim for at least `minPoints`×5.

**Example:**

```yaml
parameters:
  samplingBudget:
    minPoints: 20
    maxPoints: 100
```

#### `noPriorsParameters`

Configures the initial characterization phase (if triggered).

##### `noPriorsParameters.samples`

##### `noPriorsParameters.sampling_strategy`

**Type:** `str`

**Default:** `'clhs'`

**Purpose:** Algorithm for selecting initial points.

**Supported values:**

- `'clhs'` (Concatenated Latin Hypercube): Excellent default, ensures
  even spread
- `'sobol'`: Quasi-random, often provides best uniform coverage
- `'random'`: Simple random sampling, can leave gaps

**Tuning Guidance:**

- Avoid `'random'` unless you have specific reasons

**Example:**

```yaml
parameters:
  noPriorsParameters:
    sampling_strategy: "sobol"
```

---

### Phase 2: Iterative Modeling Configuration

#### `iterationSize`

**Type:** `int`

**Default:** `5`

**Purpose:** Number of points sampled per iteration. Also defines the
window size for stopping criterion evaluation.

**Tuning Guidance:**

- **Larger values** (e.g., 8-10): More stable stopping decisions, less
  responsive
- **Smaller values** (e.g., 3-4): More responsive, but may stop
  prematurely
- **Trade-off:** Stability vs. responsiveness,
  5 was a good value in almost all our tests.

#### `stoppingCriterion`

Controls when the iterative loop terminates.

##### `stoppingCriterion.enabled`

**Type:** `bool`

**Default:** `true`

**Purpose:** Enable/disable automatic stopping.

**Tuning Guidance:**

- Set to `false` to always run until `samplingBudget.maxPoints`
- Keep `true` for cost-efficient operation

##### `stoppingCriterion.meanThreshold`

**Type:** `float`

**Default:** `0.9`

**Purpose:** Threshold for mean performance ratio between consecutive
windows.

**How it works:** TRIM calculates:

```text
mean_ratio = mean(recent_window_scores) / mean(previous_window_scores)
```

If `1/meanThreshold < mean_ratio < meanThreshold`, the mean has
stabilized.

**Tuning Guidance:**

- **Lower values** (e.g., 0.8): Less patient, stops sooner
- **Higher values** (e.g., 0.95): More patient, continues longer
- Use higher values when accuracy is critical

##### `stoppingCriterion.stdThreshold`

**Type:** `float`

**Default:** `0.75`

**Purpose:** Threshold for standard deviation ratio between consecutive
windows.

**How it works:** TRIM calculates:

```text
std_ratio = std(recent_window_scores) / std(previous_window_scores)
```

If `std_ratio < stdThreshold`, the variance has stabilized.

**Tuning Guidance:**

- Controls sensitivity to score variability
- Lower values require more stable performance before stopping

**Stopping Logic:** Both `mean_ratio < meanThreshold` AND
`std_ratio < stdThreshold` must be true to trigger stopping.

**Example:**

```yaml
parameters:
  stoppingCriterion:
    enabled: true
    meanThreshold: 0.85
    stdThreshold: 0.7
```

---

### Phase 3: Model Configuration

For detailed information about AutoGluon configuration options, see:

- [AutoGluon Tabular Tutorials](https://auto.gluon.ai/dev/tutorials/tabular/index.html)
- [TabularPredictor API](https://auto.gluon.ai/dev/api/autogluon.tabular.TabularPredictor.html)
- [TabularPredictor.fit() API](https://auto.gluon.ai/cloud/stable/api/autogluon.cloud.TabularCloudPredictor.fit.html)

#### `autoGluonArgs`

Configures **intermediate** models trained during the iterative loop.

##### `tabularPredictorArgs`

Dictionary passed to `TabularPredictor()` constructor. Refer to
AutoGluon documentation for details.

##### `fitArgs`

Dictionary passed to `TabularPredictor.fit()`. Refer to AutoGluon
documentation for details.

**Tuning Guidance for Intermediate Models:**

- Keep `time_limit` low (30-60s) for fast iterations
- Use `'medium_quality'` presets

#### `finalModelAutoGluonArgs`

Configures the **final** production model.

**Default:** If not specified, uses same settings as `autoGluonArgs`

**Tuning Guidance for Final Model:**

- Use much higher `time_limit` (600-1800s)
- Consider using `'best_quality'` or `'high_quality'` presets.

**Example:**

```yaml
parameters:
  autoGluonArgs:
    fitArgs:
      time_limit: 45
      presets: "medium_quality"
  finalModelAutoGluonArgs:
    fitArgs:
      time_limit: 1200
      presets: "best_quality"
```

---

## Configuration Examples

### Example 1: Quick Exploration (Low Budget)

For rapid prototyping with limited measurement budget:

```yaml
{% include "../../../examples/trim/example_yamls/quick_exploration.yaml" %}
```

### Example 2: High-Quality Characterization

Balanced approach for production use:

<!-- markdownlint-disable MD013 -->
```yaml
{% include "../../../examples/trim/example_yamls/high_quality_characterization.yaml" %}
```
<!-- markdownlint-enable MD013 -->

---

## Understanding TRIM Output

### Viewing Sampled Entities

To see the entities sampled during a TRIM operation:

```bash
ado show entities operation $OPERATION_IDENTIFIER
```

This displays entities in the order they were sampled, showing the
progression through the no-priors and iterative phases.

### Accessing the Final Model

The final AutoGluon model is saved to:

```text
{outputDirectory}_finalized/
```

You can load and use it for predictions:

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor.load('trim_models_finalized')
predictions = predictor.predict(new_data)
```

### Operation Metadata

TRIM operations create metadata tracking:

- Number of entities submitted
- Number of experiments requested
- Which phase completed (no-priors vs. iterative)

Access via:

```bash
ado show details operation $OPERATION_IDENTIFIER
```

---

## Advanced Topics

### Relationship with RandomWalk

TRIM internally uses the `RandomWalk` operator with custom samplers:

- **No-priors phase:** Uses `NoPriorsSampleSelector`
- **Iterative phase:** Uses `TrimSampleSelector`

This means TRIM inherits RandomWalk's capabilities like measurement
replay.

### Feature Importance and Dimensionality Reduction

TRIM uses AutoGluon's feature importance to:

1. Identify which input parameters matter most
2. Order unmeasured points based on important features

This makes TRIM efficient even in high-dimensional spaces.

### Holdout Set Mechanism

TRIM maintains a rolling holdout set:

- Size: `holdoutSize` (equals `iterationSize`)
- Composition: Most recently measured points
- Purpose: Evaluate model generalization without overfitting
- Used for: Stopping criterion calculations

### Debugging and Troubleshooting

Enable debug logging to save intermediate files.
Set logging level when you launch your operation, for example:

<!-- markdownlint-disable line-length -->

```bash
LOGLEVEL=DEBUG ado -l DEBUG create operation -f \
  operation_pressure.yaml --use-latest space
```

<!-- markdownlint-enable line-length -->

Set your preference for the debug directory
in your experiment configuration:

```yaml
parameters:
  debugDirectory: debug_output
```

This saves:

- Source/target dataframes at each iteration
- Model performance metrics
- Feature importance rankings

---

## What's Next

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line no-inline-html -->
<div class="grid cards" markdown>

- :octicons-rocket-24:{ .lg .middle } **Try the TRIM Quickstart**

    ---

    Get started quickly with a hands-on tutorial using the ideal gas
    law example.

    [TRIM Quickstart :octicons-arrow-right-24:](https://ibm.github.io/ado/examples/trim/)

- :octicons-workflow-24:{ .lg .middle } **Explore Other Operators**

    ---

    Learn about other exploration operators like RandomWalk and
    ray_tune.

    [Explore Operators :octicons-arrow-right-24:](https://ibm.github.io/ado/operators/explore_operators/)

- :octicons-beaker-24:{ .lg .middle } **Create Custom Experiments**

    ---

    Define your own experiments to use with TRIM.

    [Custom Experiments Guide :octicons-arrow-right-24:](https://ibm.github.io/ado/actuators/creating-custom-experiments/)

- :octicons-search-24:{ .lg .middle } **Optimize with Ray Tune**

    ---

    Use ray_tune for optimization tasks instead of
    characterization.

    [Ray Tune Documentation :octicons-arrow-right-24:](https://ibm.github.io/ado/operators/optimisation-with-ray-tune/)

</div>
<!-- markdownlint-enable line-length -->