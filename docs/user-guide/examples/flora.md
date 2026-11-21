<!-- markdownlint-disable code-block-style -->

# Build a Surrogate Model with FLORA

Measuring every configuration of an experiment is often too expensive.
FLORA uses a Random Forest to find regions where trees disagree and few
measurements are available. It then trains a **surrogate model** on the
selected measurements to estimate the rest.

This tutorial applies FLORA to the [acid–base titration example](titration.md).
The space has 1,830 possible conditions and the budget is 60 measurements.
You will create the discovery space, run FLORA, and compare a model prediction
with the pH from the titration equations. Choose a Random Forest or AutoGluon
for the final model.

## Prerequisites

Use Python 3.11–3.14 and a current `ado` source checkout.
Activate your Python environment, then run all commands from the repository
root. See [Getting Started](../getting-started.md#installing) for setup.

Install the core, the active-learning plugin, and the example experiment from
the same checkout:

```bash
pip install -e . \
  -e plugins/operators/active_learning \
  -e examples/titration/custom_experiments/
```

For the AutoGluon path, use Python 3.11–3.13 and install the optional extra:

```bash
pip install -e "plugins/operators/active_learning[autogluon]"
```

FLORA requires a discovery space with exactly one experiment. That experiment
can return several outputs. `targetOutput` selects the numeric output to model.
A **context** selects the database where `ado` stores resources and
measurements. Check your active context and the installed plugins:

```bash
ado context
ado get operators
ado describe experiment titrate_strong_acid
```

Check that `flora` appears as a `characterize` operator.
Look for `titrate_strong_acid-ph` under `Outputs`. The operation configuration
uses its short name, `ph`.
See the [context guide](../../cli-reference/index.md#ado-context) to select
another database.

## Step 1: Create the Discovery Space

A **discovery space** defines the candidate conditions and the experiment to
run. Each **entity** is one combination of acid concentration, base
concentration, and titrant volume. The **target** is the property to model: the
pH. A **label** is its measured value for one entity.

Create the space with an empty sample store for its measurements:

```bash
ado create space \
  -f examples/titration/example_yamls/space_titration.yaml --new-sample-store
```

The command prints the new space identifier. Your identifier will differ from
the one in this example:

```text
Success! Created space with identifier: space-611e3b-f1690c
```

The [titration grid](titration.md#step-3-define-a-discovery-space) contains 30
acid–base concentration pairs and 61 titrant volumes for each pair. The pH
rises sharply near the **equivalence point**, where the added base neutralises
the acid. Use the [titration equations](titration.md#the-titration-equations)
to check FLORA's predictions against the pH before simulated meter noise.

## Step 2: Run FLORA

Use the operation configuration at
`examples/titration/example_yamls/operation_flora.yaml`:

<!-- prettier-ignore-start -->

```yaml
{%
  include-markdown "../../../examples/titration/example_yamls/operation_flora.yaml"
%}
```
<!-- prettier-ignore-end -->

The **pilot** is the initial sample used to train the first forest.
These three settings control the run:

| Parameter | Value | Meaning |
| :--- | :--- | :--- |
| `targetOutput` | `ph` | The measured property that guides selection and model training |
| `numberEntities` | `60` | Maximum new entities to measure, including the pilot |
| `pilotSize` | `5` | Labels needed before active selection starts |

The budget covers about 3% of the space. This configuration sets
`maxFeatures: 1.0` so each tree can consider all three inputs at each split.
FLORA schedules its forest refits automatically.

!!! warning "Rerunning this example"

    Choose a new `outputDirectory` for each run, such as `flora_models_run2`.
    Use the matching `_finalized` directory in the commands below. To
    reproduce the results shown here, also repeat Step 1 with
    `--new-sample-store`. If you keep the same space, FLORA reuses its earlier
    measurements and spends its budget on additional entities.

Choose the final model for this run. FLORA uses the same forest to select
measurements with either backend.

=== "Random Forest"

    Run the example configuration as supplied:

    ```bash
    ado create operation \
      -f examples/titration/example_yamls/operation_flora.yaml \
      --use-latest space
    ```

=== "AutoGluon"

    Use the same configuration and set `finalPredictor` on the command line:

    ```bash
    ado create operation \
      -f examples/titration/example_yamls/operation_flora.yaml \
      --use-latest space --set operation.parameters.finalPredictor=autogluon
    ```

`--use-latest space` uses the space you just created in the active context.
It replaces the example's `inputs.discoverySpace` identifier.
The command prints the operation resource and ends with the FLORA operation
identifier:

<!-- markdownlint-disable MD013 -->

```text
Success! Created operation with identifier operation-flora@0.2.0-06739736 and it finished successfully.
```

<!-- markdownlint-enable MD013 -->

### What happens during the run

1. **Pilot:** FLORA measures five entities chosen uniformly at random.
   Change `pilotSize` to adjust this.
2. **Learn:** FLORA fits a Random Forest to these measurements. Each tree
   groups conditions into regions called leaves.
3. **Score:** For every entity, FLORA computes how much the trees disagree
   about its pH. It combines this disagreement with the number of
   measurements in each leaf to estimate the gain from one more label.
4. **Select:** FLORA chooses the unmeasured entity with the highest mean gain
   across the trees. It waits for the target label, then updates the measured
   leaf counts.
5. **Refit and repeat:** FLORA refits the forest after 5, 8, 12, 17, 23, 31, 40,
   and 51 measurements. Selection continues until it measures 60 entities or
   exhausts the pool.
6. **Save:** FLORA fits the final model and predicts the pH of the other 1,770
   entities.

Every selected entity must produce a finite target value. A missing or invalid
label fails the operation.

The [algorithm guide](../operators/flora.md#how-flora-works) shows a small
example and the equations.

Existing target measurements count toward `pilotSize` and do not use the
new-entity budget. This tutorial starts with an empty store, so FLORA collects
60 new measurements.

## Step 3: Inspect the Results

### Check the measurements

```bash
ado show measurements space --use-latest --property ph -o csv
```

The output lists the identifier and pH of each of the 60 measured entities. The
row order can vary. This run starts with:

```text
,identifier,ph
0,acid_concentration.0.08-base_concentration.0.4-titrant_volume.11,12.73728753825219
1,acid_concentration.0.1-base_concentration.0.1-titrant_volume.31,12.083661810715315
```

These are experiment results that `ado` stores in the sample store.

### Find the output files

FLORA writes these files to `flora_models_finalized/`, relative to the
directory where you ran the operation:

| File | Contents |
| :--- | :--- |
| `model.pkl` | Fitted Random Forest and its preprocessing pipeline |
| `autogluon/` | Fitted AutoGluon predictor |
| `combined_predictions.csv` | Measured and predicted pH for all 1,830 entities |
| `metadata.json` | Backend, feature names, measured count, and creation time |

Predictions stay in these local files. FLORA does not record them as
experimental measurements in the sample store.

### Read the predictions

The model learns from 60 of the 1,830 conditions. Where few measurements
bracket the sharp rise in pH, the model can miss the location of the jump.

```python
import pandas as pd

table = pd.read_csv("flora_models_finalized/combined_predictions.csv")
print(f"Total entities:     {len(table)}")
print(f"Measured entities:  {(~table['is_predicted']).sum()}")
print(f"Predicted entities: {table['is_predicted'].sum()}")
columns = ["acid_concentration", "base_concentration", "titrant_volume", "ph"]
print(table.loc[table["is_predicted"], columns].head(3))
```

Both paths give the same row counts. The pH estimates below are from the
Random Forest run:

```text
Total entities:     1830
Measured entities:  60
Predicted entities: 1770
   acid_concentration  base_concentration  titrant_volume        ph
0                0.02                 0.1               0  5.105825
1                0.02                 0.1               1  5.105825
2                0.02                 0.1               2  5.105825
```

The `is_predicted` column is `False` for measured values and `True` for model
estimates. The default `seed` selects the same entities on every run with the
same package versions.

The forest predicts 5.1 for each of the first three volumes. At 0 mL, the
titration equations give a pH of 1.7, so the first estimate is 3.4 pH units
too high. FLORA measured only two entities for this acid–base pair.

### Predict a new configuration

Use a DataFrame with the same property names as the training data. This
configuration is one of the 1,770 entities that FLORA did not measure. It lies
0.5 mL before the equivalence point. At this volume the titration equations
give a pH of 2.6.

```python
import pandas as pd

configuration = pd.DataFrame(
    [{"acid_concentration": 0.1, "base_concentration": 0.2, "titrant_volume": 12.0}]
)
```

Load the model from your run and predict the pH:

=== "Random Forest"

    ```python
    import pickle

    with open("flora_models_finalized/model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    predicted_ph = model.predict(configuration)[0]
    print(f"Model estimate: {predicted_ph:.1f}")
    ```

    In this run, the Random Forest predicted 5.3, which is 2.7 pH units above
    the value from the titration equations.

=== "AutoGluon"

    Load the predictor in the Python environment you used to train it:

    ```python
    from autogluon.tabular import TabularPredictor

    model = TabularPredictor.load("flora_models_finalized/autogluon")
    predicted_ph = model.predict(configuration).iloc[0]
    print(f"Model estimate: {predicted_ph:.1f}")
    ```

    In this run, AutoGluon predicted 2.7. Your estimate can vary with the
    models AutoGluon fits within its time limit.

!!! warning

    Only load saved models from sources you trust.

### Inspect the operation

FLORA collects measurements through a related `random_walk` operation. Replace
the example identifier below with the one printed by `ado create operation`:

```bash
OPERATION_IDENTIFIER="operation-flora@0.2.0-06739736"
ado show related operation "$OPERATION_IDENTIFIER" --max-hops 1
```

!!! tip

    Do not use `--use-latest` to find the FLORA operation. FLORA collects
    measurements through a related `random_walk` operation, which is created
    later and so is the most recent operation.

## Next Steps

- Try a different `numberEntities` or `pilotSize`. Compare the predictions
  with the titration equations for several conditions.
- Use the [configuration guide](../operators/flora.md#configuration-parameters)
  to adapt FLORA to another finite discovery space.
- Try [TRIM](trim.md), an experimental design framework that uses prediction
  stability to decide when to stop instead of a fixed measurement budget.
