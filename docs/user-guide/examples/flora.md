<!-- markdownlint-disable code-block-style -->

# Build a Surrogate Model with FLORA

Measuring every configuration of an experiment is often too expensive.
FLORA uses a Random Forest to find regions where trees disagree and few
measurements are available. It then trains a **surrogate model** on the
selected measurements to estimate the rest.

This tutorial applies FLORA to the [acid–base titration example](titration.md).
The space has 1,830 possible conditions and the budget is 60 measurements.
You will create the discovery space, run FLORA, and inspect the measurements
and predictions. You will also compare predictions with the pH given by the
titration equations and check how the error changes with [PKH](pkh.md) and
random sampling.

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
The experiment description should list `ph` as its output.
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

The [titration grid](titration.md#create-the-discovery-space) contains 30
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

The budget covers about 3% of the space. FLORA schedules its forest refits
automatically.

!!! warning "Rerunning this example"

    FLORA creates `flora_models_finalized/` before it collects measurements. It
    refuses to run if that path already exists, and it leaves earlier files
    untouched. Before a second run, set a new `outputDirectory` in the
    configuration, for example `flora_models_run2`. Then use the matching
    `_finalized` directory in the commands below.

Start the operation:

```bash
ado create operation \
  -f examples/titration/example_yamls/operation_flora.yaml \
  --use-latest space
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

!!! tip "Train the final model with AutoGluon"

    This tutorial uses the default Random Forest for the final model. To train
    an AutoGluon predictor instead, install the extra on Python 3.11–3.13 with
    `pip install -e "plugins/operators/active_learning[autogluon]"`. Then set
    `finalPredictor: autogluon` in the configuration. FLORA saves the model in
    `autogluon/` instead of `model.pkl`, and you load it with
    `TabularPredictor.load`. See the
    [`finalPredictor` parameter](../operators/flora.md#finalpredictor).

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
| `model.pkl` | Fitted preprocessing pipeline and Random Forest |
| `combined_predictions.csv` | Measured and predicted pH for all 1,830 entities |
| `metadata.json` | Backend, feature names, measured count, and creation time |

Predictions stay in these local files. FLORA does not record them as
experimental measurements in the sample store.

### Read the predictions

```python
import pandas as pd

table = pd.read_csv("flora_models_finalized/combined_predictions.csv")
print(f"Total entities:     {len(table)}")
print(f"Measured entities:  {(~table['is_predicted']).sum()}")
print(f"Predicted entities: {table['is_predicted'].sum()}")
columns = ["acid_concentration", "base_concentration", "titrant_volume", "ph"]
print(table.loc[table["is_predicted"], columns].head(3))
```

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
same package versions. Predictions can differ in the last digits because the
forest trains on several processors.

The forest predicts 5.1 for each of the first three volumes. At 0 mL, the
titration equations give a pH of 1.7, so the first estimate is 3.4 pH units
too high. FLORA measured only two entities for this acid–base pair.

### Predict a new configuration

Load the model and supply a DataFrame with the same property names. This
configuration is one of the 1,770 entities that FLORA did not measure. It lies
0.5 mL before the equivalence point.

```python
import pickle

import pandas as pd

with open("flora_models_finalized/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

configuration = pd.DataFrame(
    [{"acid_concentration": 0.1, "base_concentration": 0.2, "titrant_volume": 12.0}]
)
predicted_ph = float(model.predict(configuration)[0])
print(f"Model estimate: {predicted_ph:.1f}")
```

```text
Model estimate: 5.3
```

The titration equations give a pH of 2.6 for this configuration, so the model
overestimates the pH by 2.7 units.

!!! warning

    `model.pkl` is a Python pickle file. Only load pickle files you trust.

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

### Compare with PKH

Run PKH on the same problem to see where each operator spends its budget. Create
a second space with its own sample store. If you reuse the first space, PKH
starts from the 60 measurements FLORA already made:

```bash
ado create space \
  -f examples/titration/example_yamls/space_titration.yaml --new-sample-store
ado create operation \
  -f examples/titration/example_yamls/operation_pkh.yaml --use-latest space
```

Both operators start with five pilot measurements and a budget of 60. PKH
refits after every five new measurements. Score both models against the
noiseless titration equations on the entities each operator left unmeasured.
The root mean squared error (RMSE) is in pH units.

```python
import numpy as np
import pandas as pd

for operator in ("flora", "pkh"):
    table = pd.read_csv(f"{operator}_models_finalized/combined_predictions.csv")
    acid = table["acid_concentration"]
    base = table["base_concentration"]
    volume = table["titrant_volume"]
    excess_acid = (acid * 25 - base * volume) / (25 + volume)
    hydrogen = (excess_acid + np.sqrt(excess_acid**2 + 4e-14)) / 2
    true_ph = -np.log10(hydrogen)

    error = (table["ph"] - true_ph)[table["is_predicted"]]
    rmse = np.sqrt((error**2).mean())
    near_equivalence = (volume - acid * 25 / base).abs() < 8
    measured = ~table["is_predicted"]
    measured_near_equivalence = near_equivalence[measured].sum()
    print(
        f"{operator:5s} RMSE {rmse:.2f} pH, "
        f"{measured_near_equivalence} of {measured.sum()} measurements "
        "within 8 mL of equivalence"
    )
```

```text
flora RMSE 2.03 pH, 31 of 60 measurements within 8 mL of equivalence
pkh   RMSE 2.58 pH, 15 of 60 measurements within 8 mL of equivalence
```

Of the 1,830 entities, 436 lie within 8 mL of an equivalence point (24%).
FLORA measured 31 entities in this region (52% of its budget) and PKH measured
15 (25%). FLORA's RMSE was 21% lower in this run.

To check how the choice of seed affects this result, the comparison script
repeats the selection with seeds 0–29. It runs each operator's sampler on the
full pool. Each seed gives a different pilot sample and forest. The random
method selects 60 entities uniformly. The script fits each final forest with
the same training-row order and forest seed as the operator. The random
baseline uses FLORA's final forest settings. Run it from the repository root:

```bash
python examples/titration/compare_sampling.py --seeds 30 --output comparison.csv
```

The script prints package versions and summary statistics, and saves each
seed's RMSE and measurement counts in `comparison.csv`. Use `--seed 42
--seeds 1` to compare the seed used in the CLI example, or `--budget 120` to
increase the measurement budget. With NumPy 2.3.5 and scikit-learn 1.9.1, the
median and interquartile range of the pH RMSE on each method's unmeasured
entities are:

| Method | Median RMSE | Interquartile range |
| :--- | :--- | :--- |
| FLORA | 2.16 | 2.11–2.19 |
| PKH | 2.63 | 2.50–2.79 |
| Random | 2.97 | 2.81–3.26 |

FLORA had a lower RMSE than PKH for all 30 seeds. Its median error still
exceeded 2 pH units with this measurement budget. Compare the operators on
your own discovery space to see how each samples the conditions of interest.

## Next Steps

- Try a different `numberEntities` or `pilotSize`. Compare the predictions
  with the titration equations for several conditions.
- Use the [configuration guide](../operators/flora.md#configuration-parameters)
  to adapt FLORA to another finite discovery space.
- Try [TRIM](trim.md), an experimental design framework that uses prediction
  stability to decide when to stop instead of a fixed measurement budget.
