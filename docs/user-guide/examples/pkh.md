<!-- markdownlint-disable code-block-style -->

# Build a Surrogate Model with PKH

In this tutorial, Predictive Kernel Herding (PKH) selects 20 of the 1,215 gas
conditions from the [TRIM tutorial](trim.md). PKH trains a **surrogate model**
on the measured pressures and predicts the pressure for the remaining conditions.
You will check one prediction against the ideal gas law, $P = nRT/V$.

## Prerequisites

Use Python 3.11–3.14 and a current `ado` source checkout.
Activate your Python environment. Run the commands from the repository root.
See [Getting Started](../getting-started.md#installing) for setup.

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
Success! Created space with identifier: space-ab6c24-a57faa
```

The configuration defines this grid. Each range includes its lower bound and
excludes its upper bound:

| Property | Range | Step | Number of values |
| :--- | :--- | :--- | :--- |
| `temperature` | 270 to 300 K | 2 | 15 |
| `volume` | 1 to 10 m³ | 1 | 9 |
| `mol` | 0.1 to 1 mol | 0.1 | 9 |

The grid contains 15 × 9 × 9 = **1,215 entities**. When PKH selects an
entity, the `calculate_pressure_ideal_gas` experiment calculates its pressure.

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

The **pilot** is the initial sample of entities that PKH measures at random to
train the first Random Forest before active learning begins.
These four settings control the run:

| Parameter | Value | Meaning |
| :--- | :--- | :--- |
| `targetOutput` | `pressure` | The measured property that guides selection and model training |
| `numberEntities` | `20` | Maximum new entities to measure, including the pilot |
| `pilotSize` | `5` | Labels needed before herding starts |
| `epochLength` | `5` | New measurements between forest refits |

!!! warning "Rerunning this example"

    PKH creates a new output directory before it collects measurements. For a
    second run, set `outputDirectory` to a new name, such as `pkh_models_run2`.
    Use `pkh_models_run2_finalized/` in the commands below for that run.

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
Success! Created operation with identifier operation-pkh@0.1.0-423031d3 and it finished successfully.
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

PKH needs a finite pressure measurement from each selected entity before it
can select the next one.

The selection step is **herding**: PKH tries to match the sample's leaf
proportions to those of the whole space.
The [algorithm guide](../operators/pkh.md#how-pkh-works) shows a small example
and the equations.

Existing target measurements count toward `pilotSize`. The budget counts new
measurements. This tutorial starts with an empty store, so PKH collects all 20.

!!! tip "Train the final model with AutoGluon"

    This tutorial uses the default Random Forest. For AutoGluon on
    Python 3.11–3.13, install the extra with
    `pip install -e "plugins/operators/active_learning[autogluon]"`.
    Set `finalPredictor: autogluon` in the configuration. PKH saves the model
    in `pkh_models_finalized/autogluon/`. The
    [AutoGluon example](../operators/pkh.md#__tabbed_1_2) shows how to reload
    it and predict pressure.

## Step 3: Inspect the Results

### Check the measurements

```bash
ado show measurements space --use-latest --property pressure
```

The table shows the identifier and pressure of each of the 20 measured
entities. The row order can vary. This run starts with:

```text
INFO:   Using space space-ab6c24-a57faa.
┌───────┬──────────────────────────────────┬────────────────────┐
│ INDEX │ identifier                       │ pressure           │
├───────┼──────────────────────────────────┼────────────────────┤
│ 0     │ mol.0.5-temperature.282-volume.5 │ 234.4678458276     │
│ 1     │ mol.0.3-temperature.278-volume.1 │ 693.4261823412     │
│ 2     │ mol.0.4-temperature.292-volume.2 │ 485.5646168912     │
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

!!! warning

    Load `model.pkl` only if you trust its source.

Load the model. Supply the same property names it used for training.
This configuration comes from the 1,195 rows where `is_predicted` is `True`:

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

In this run, the model estimates a pressure about 11% below the ideal gas
value. The model estimate and MMD² can vary slightly across package versions
and platforms.

### Check the sampling diagnostic

PKH also reports **MMD²**. It measures how closely the measured entities match
the whole space in the forest's leaf representation. A lower value means a
closer match. Use the ideal-gas comparison above to check prediction accuracy.

Read the final value from `metadata.json`:

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
OPERATION_IDENTIFIER="operation-pkh@0.1.0-423031d3"
ado get operation "$OPERATION_IDENTIFIER" -o yaml
```

!!! tip

    Use the printed PKH operation ID to check MMD². The related
    `random_walk` operation can be the most recent operation.

Set `LOGLEVEL=INFO` when you start the operation to log MMD² at each refit.
The value can increase after a refit, because each refit changes the forest.
See the [MMD reference](../operators/pkh.md#mmd-statistic) for details.

## Next Steps

- Try a different `numberEntities`, `pilotSize`, or `epochLength`.
  Compare the predictions with the ideal gas law for several conditions.
- Use the [configuration guide](../operators/pkh.md#configuration-parameters)
  to adapt PKH to another finite discovery space.
- Try [TRIM](trim.md) to stop sampling when holdout scores stabilize or the run
  reaches its budget.
