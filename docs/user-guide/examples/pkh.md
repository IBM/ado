<!-- markdownlint-disable code-block-style -->

# Build a Surrogate Model with PKH

Measuring every configuration of an experiment is often too expensive.
Predictive Kernel Herding (PKH) selects a small, representative set of
configurations to measure. It then trains a **surrogate model** on those
measurements to estimate the rest.

This tutorial applies PKH to the ideal gas example from the
[TRIM tutorial](trim.md). The space has 1,215 possible gas conditions and the
budget is 20 measurements. You will create the discovery space, run PKH, and
inspect the measurements and predictions. You will also compare one prediction
with the ideal gas law, $P = nRT/V$, which gives a known answer to check the
model against.

## Prerequisites

Use Python 3.11–3.14 and a current `ado` source checkout.
Activate your Python environment, then run all commands from the repository
root. See [Getting Started](../getting-started.md#installing) for setup.

Install the core, PKH plugin, and example experiment from the same checkout:

```bash
pip install -e . \
  -e plugins/operators/active_learning \
  -e examples/trim/custom_experiments/
```

PKH requires a discovery space with exactly one experiment. That experiment can
return several outputs. `targetOutput` selects the numeric output to model.
A **context** selects the database where `ado` stores resources and
measurements. Check your active context and the installed plugins:

```bash
ado context
ado get operators
ado describe experiment calculate_pressure_ideal_gas
```

Check that `pkh` appears as a `characterize` operator.
The experiment description should list `pressure` as its output.
See the [context guide](../../cli-reference/index.md#ado-context) to select
another database.

## Step 1: Create the Discovery Space

A **discovery space** defines the candidate conditions and the experiment to
run. Each **entity** is one combination of temperature, volume, and amount of
gas. The **target** is the property to model: pressure. A **label** is its
measured value for one entity.

Create the space with an empty sample store for its measurements:

```bash
ado create space \
  -f examples/trim/example_yamls/space_pressure.yaml --new-sample-store
```

The command prints the new space identifier. Your identifiers will differ from
the ones in this example:

```text
Success! Created space with identifier: space-e3da9d-82ee36
```

The configuration defines this grid. Each range includes its lower bound and
excludes its upper bound:

| Property | Range | Step | Number of values |
| :--- | :--- | :--- | :--- |
| `temperature` | 270 to 300 K | 2 | 15 |
| `volume` | 1 to 10 m³ | 1 | 9 |
| `mol` | 0.1 to 1 mol | 0.1 | 9 |

The grid contains 15 × 9 × 9 = **1,215 entities**.
The `calculate_pressure_ideal_gas` experiment calculates pressure for a
selected entity. Creating the space does not run this experiment.

## Step 2: Run PKH

Use the operation configuration at
`plugins/operators/active_learning/examples/ideal_gas/operation_pkh.yaml`:

<!-- prettier-ignore-start -->

```yaml
{%
  include-markdown "../../../plugins/operators/active_learning/examples/ideal_gas/operation_pkh.yaml"
%}
```
<!-- prettier-ignore-end -->

The **pilot** is the initial sample used to train the first forest.
These four settings control the run:

| Parameter | Value | Meaning |
| :--- | :--- | :--- |
| `targetOutput` | `pressure` | The measured property that guides selection and model training |
| `numberEntities` | `20` | Maximum new entities to measure, including the pilot |
| `pilotSize` | `5` | Labels needed before herding starts |
| `epochLength` | `5` | New measurements between forest refits |

!!! warning "Rerunning this example"

    PKH creates `pkh_models_finalized/` before it collects measurements. It
    refuses to run if that path already exists, and it leaves earlier files
    untouched. Before a second run, set a new `outputDirectory` in the
    configuration, for example `pkh_models_run2`. Then use the matching
    `_finalized` directory in the commands below.

Start the operation:

```bash
ado create operation \
  -f plugins/operators/active_learning/examples/ideal_gas/operation_pkh.yaml \
  --use-latest space
```

`--use-latest space` uses the space you just created in the active context.
It replaces the example's `inputs.discoverySpace` identifier.
The command prints the operation resource and ends with the PKH operation
identifier:

<!-- markdownlint-disable MD013 -->

```text
Success! Created operation with identifier operation-pkh@0.1.0-d817a02e and it finished successfully.
```

<!-- markdownlint-enable MD013 -->

### What happens during the run

1. **Pilot:** PKH measures five entities chosen uniformly at random.
   Change `pilotSize` to adjust this.
2. **Learn:** PKH fits a Random Forest to these measurements.
   Each tree groups conditions into regions called leaves.
3. **Select:** PKH chooses one entity from leaves that need more representation.
   It waits for the target label, then updates the measured leaf counts.
4. **Refit and repeat:** After every five new measurements, PKH refits the
   forest. Selection continues until it measures 20 entities or exhausts the
   pool.
5. **Save:** PKH fits the final model and predicts pressure for the other
   1,195 entities.

Every selected entity must produce a finite target value. A missing or invalid
label fails the operation.

The selection step is **herding**: PKH tries to match the sample's leaf
proportions to those of the whole space.
The [algorithm guide](../operators/pkh.md#how-pkh-works) shows a small example
and the equations.

Existing target measurements count toward `pilotSize` and do not use the
new-entity budget. This tutorial starts with an empty store, so PKH collects 20
new measurements.

!!! tip "Train the final model with AutoGluon"

    This tutorial uses the default Random Forest for the final model. To train
    an AutoGluon predictor instead, install the extra on Python 3.11–3.13 with
    `pip install -e "plugins/operators/active_learning[autogluon]"`. Then set
    `finalPredictor: autogluon` in the configuration. PKH saves the model in
    `autogluon/` instead of `model.pkl`, and you load it with
    `TabularPredictor.load`. See the
    [`finalPredictor` parameter](../operators/pkh.md#finalpredictor).

## Step 3: Inspect the Results

### Check the measurements

```bash
ado show measurements space --use-latest --property pressure
```

The table shows the identifier and pressure of each of the 20 measured
entities. The row order can vary. This run starts with:

```text
INFO:   Using space space-e3da9d-82ee36.
┌───────┬──────────────────────────────────┬────────────────────┐
│ INDEX │ identifier                       │ pressure           │
├───────┼──────────────────────────────────┼────────────────────┤
│ 0     │ mol.0.3-temperature.292-volume.6 │ 121.39115422280001 │
│ 1     │ mol.0.1-temperature.272-volume.4 │ 56.5383458024      │
│ 2     │ mol.0.4-temperature.276-volume.2 │ 458.9583365136     │
```

These are experiment results that `ado` stores in the sample store.

### Find the output files

PKH writes these files to `pkh_models_finalized/`, relative to the directory
where you ran the operation:

| File | Contents |
| :--- | :--- |
| `model.pkl` | Fitted preprocessing pipeline and Random Forest |
| `combined_predictions.csv` | Measured and predicted pressure for all 1,215 entities |
| `metadata.json` | Backend, feature names, measured count, and final MMD² |

Predictions stay in these local files. PKH does not record them as experimental
measurements in the sample store.

### Read the predictions

```python
import pandas as pd

table = pd.read_csv("pkh_models_finalized/combined_predictions.csv")
print(f"Total entities:     {len(table)}")
print(f"Measured entities:  {(~table['is_predicted']).sum()}")
print(f"Predicted entities: {table['is_predicted'].sum()}")
columns = ["temperature", "volume", "mol", "pressure"]
print(table.loc[table["is_predicted"], columns].head(3))
```

```text
Total entities:     1215
Measured entities:  20
Predicted entities: 1195
   temperature  volume  mol    pressure
0          270       1  0.1  865.325770
1          270       1  0.2  865.250016
2          270       1  0.3  888.143460
```

The `is_predicted` column is `False` for measured values and `True` for model
estimates. The default `seed` makes the run repeatable on the same package
versions.

### Predict a new configuration

Load the model and supply a DataFrame with the same property names. This
configuration is one of the 1,195 entities that PKH did not measure:

```python
import pickle

import pandas as pd

with open("pkh_models_finalized/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

configuration = pd.DataFrame([{"temperature": 284.0, "volume": 4.0, "mol": 0.6}])
predicted_pressure = float(model.predict(configuration)[0])
ideal_pressure = 0.6 * 8.314462618 * 284.0 / 4.0
print(f"Model estimate: {predicted_pressure:.1f} Pa")
print(f"Ideal gas law:  {ideal_pressure:.1f} Pa")
```

```text
Model estimate: 315.9 Pa
Ideal gas law:  354.2 Pa
```

For this configuration, the model underestimates the pressure by about 11%.

!!! warning

    `model.pkl` is a Python pickle file. Only load pickle files you trust.

### Check the sampling diagnostic

PKH also reports **MMD²**. It measures how closely the measured entities match
the whole space in the forest's leaf representation. A lower value means a
closer match. MMD² does not measure prediction error.

The final value is stored in `metadata.json`:

```bash
grep mmdSquared pkh_models_finalized/metadata.json
```

```text
  "mmdSquared": 0.007886089518874154
```

PKH also stores this value as `metadata.pkhMmdSquared` on the PKH operation.
Replace the example identifier below with the one printed by
`ado create operation`:

```bash
OPERATION_IDENTIFIER="operation-pkh@0.1.0-d817a02e"
ado get operation "$OPERATION_IDENTIFIER" -o yaml
```

!!! tip

    Do not use `--use-latest` to find the PKH operation. PKH collects
    measurements through a related `random_walk` operation, which is created
    later and so is the most recent operation.

Set `LOGLEVEL=INFO` when you start the operation to log MMD² at each refit.
The value can increase after a refit, because each refit changes the forest.
See the [MMD reference](../operators/pkh.md#mmd-statistic) for details.

## Next Steps

- Try a different `numberEntities`, `pilotSize`, or `epochLength`.
  Compare the predictions with the ideal gas law for several conditions.
- Use the [configuration guide](../operators/pkh.md#configuration-parameters)
  to adapt PKH to another finite discovery space.
- Try [TRIM](trim.md), an experimental design framework that uses prediction
  stability to decide when to stop instead of a fixed measurement budget.
