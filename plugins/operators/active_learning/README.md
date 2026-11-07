# ADO Active-Learning Operators

`ado-active-learning` is an operator plugin for the
[Accelerated Discovery Orchestrator (ADO)](https://github.com/IBM/ado),
providing two regression-based active-learning operators for finite Discovery
Spaces: **PKH** and **FLORA**.

Both are designed for scenarios where measuring an entity's target property is
slow or costly. Each operator trains a Random Forest regressor on the labels
collected so far and uses it to choose, one entity at a time, which unmeasured
entity to measure next.

## How it Works

Both operators start with a pilot of `pilotSize` labelled entities (default `5`).
Set `pilotStrategy` to `random` (default), `clhs` (concatenated Latin hypercube
sampling), or `sobol`. Each strategy selects distinct entities from the finite
pool using `seed`; CLHS and Sobol use the original input dimensions.
Existing target labels count toward `pilotSize`. Adaptive selection starts
once all pilot labels are available. A smaller pool or measurement budget can
end the operation during the pilot. New pilot measurements count toward
`numberEntities`; existing labels do not.

Both operators select one entity, wait for ADO to record its target value, and
update the labelled counts in the forest leaves. They incorporate the collected
target values when the forest is next refitted. Measurements are sequential;
`nJobs` controls parallel workers within the forest.

At completion, each operator builds a final predictor from the available target
measurements, including prior measurements and the final selected entity. Set
`finalPredictor` to `random_forest` (default) or `autogluon`. This model can
predict the target for unmeasured configurations; acquisition uses the operator's
Random Forest with either final predictor.

### PKH (Predictive Kernel Herding)

PKH treats forest leaves as a coarse summary of the entity pool's
distribution. After fitting, it builds two histograms per tree: how much of
the *full pool* falls in each leaf, and how much of the *labelled sample*
falls in each leaf. At each step, PKH selects the entity with the largest mean
leaf deficit across trees, where pool coverage exceeds labelled coverage.
This pulls the labelled sample's distribution toward the full pool's
distribution, the same
discrepancy-minimization idea behind kernel herding. The forest is refit every
`epochLength` selections (default `10`), and leaf counts are updated in between.
Equal maximum scores are resolved uniformly at random using a dedicated stream
initialized by `seed`, independently of pilot sampling and forest fitting.

### FLORA

FLORA uses the forest to track predictive disagreement. For every pool entity
it computes the variance across the individual trees' predictions — a proxy
for how uncertain the forest is there — and
averages that disagreement within each leaf. At each step, FLORA selects the
entity with the highest average leaf gain across trees. This score estimates
risk reduction using leaf disagreement, pool size and labelled counts. The
forest is refit on a log-tempered schedule, with longer intervals as the labelled
sample grows.

For FLORA, `seed` initializes separate random streams for forest refits and
tie-breaking. Each stream is reproducible, and extra tie-breaking leaves the
sequence of forest seeds unchanged. Pilot
sampling uses its own random generator.

Prediction accuracy depends on the experiment, measurement budget and final
predictor. The acquisition scores describe how each method chooses measurements;
they do not guarantee a particular prediction error or ranking between methods.

## Installation

Run these commands from the ADO repository root in the Python environment where
ADO is installed:

```bash
uv pip install -e plugins/operators/active_learning
```

Confirm that ADO discovered both operators:

```bash
ado get operators
```

The output should include `pkh` and `flora`.

## Examples

[`examples/operation_pkh.yaml`](examples/operation_pkh.yaml) and
[`examples/operation_flora.yaml`](examples/operation_flora.yaml) are complete
operation files that run against
[`examples/discoveryspace.yaml`](examples/discoveryspace.yaml), a 100-entity
space built on ADO's `calculate_density` custom experiment:

```bash
uv pip install -e examples/density_example
uv pip install -e plugins/operators/active_learning
ado create space \
  -f plugins/operators/active_learning/examples/discoveryspace.yaml \
  --new-sample-store
ado create operation \
  -f plugins/operators/active_learning/examples/operation_pkh.yaml \
  --use-latest space
```

For an independent FLORA run, create a fresh space with `--new-sample-store`,
then use `operation_flora.yaml`.

The ideal-gas walkthroughs cover installation, sampling and prediction:

- [Build a surrogate model with PKH](examples/ideal_gas/PKH.md)
- [Build a surrogate model with FLORA](examples/ideal_gas/FLORA.md)

## Configure the final predictor

The default `finalPredictor: random_forest` fits on every available target
measurement and uses the operation's forest settings.

For AutoGluon, install the optional extra in a Python 3.11–3.13 environment:

```bash
python -m pip install -e 'plugins/operators/active_learning[autogluon]'
```

Set these fields under `operation.parameters` for either operator:

```yaml
finalPredictor: autogluon
autoGluon:
  presets: medium_quality
  timeLimit: 60
  numCpus: 1
```

These are the default AutoGluon fit settings. `timeLimit` is an approximate
training budget in seconds; `numCpus` controls CPU use. Supported `presets`
are `medium_quality`, `good_quality`, `high_quality`, and `best_quality`.
An optional `autoGluon.hyperparameters` configuration selects the candidate
models; see the [AutoGluon fit options][autogluon-fit] for details.
The operation's `seed` also controls the final learner.

AutoGluon requires at least four usable target measurements. The default
`medium_quality` preset reserves part of the measured data for internal
validation. Bagging presets use cross-validation and may refit on all supplied
rows. The saved `metadata.json` records `n_training_samples` (supplied labels),
`n_fit_samples`, and `n_validation_samples`. Predictions for the remaining
pool use the saved final predictor.

## Saved models and predictions

Each operation writes `model.pkl` and `metadata.json` to
`{outputDirectory}_finalized`. The defaults are `pkh_models_finalized/` and
`flora_models_finalized/`. AutoGluon artifacts also contain an `autogluon/`
subdirectory. Keep the entire finalized directory together. Set a new
`outputDirectory` in the operation YAML before another run; an existing
finalized directory is rejected.

Export the unmeasured configurations from the space:

```bash
ado show measurements space --use-latest --include unmeasured -o csv \
  --output-file unmeasured.csv
```

Load either final predictor from a PKH run and predict its target on those
configurations:

```python
import pandas as pd

from active_learning.regression import load_predictor

model = load_predictor("pkh_models_finalized")
unmeasured = pd.read_csv("unmeasured.csv", index_col=0)
unmeasured[f"predicted_{model.target_output}"] = model.predict(unmeasured)
unmeasured.to_csv("predictions.csv", index=False)
```

Use `flora_models_finalized` to load a FLORA model. Provide the original feature
names and values; the saved predictor applies the feature encoding. Predictions
are estimates and are exported separately from ADO's measured values.

[autogluon-fit]: https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html
