<!-- markdownlint-disable code-block-style -->

# Build a Surrogate Model with PKH

In this tutorial, Predictive Kernel Herding (PKH) selects 20 of the 1,215 gas
conditions from the [TRIM tutorial](trim.md). PKH trains a **surrogate model**
on the measured pressures and predicts the pressure for the remaining conditions.
You will check one prediction against the ideal gas law, $P = nRT/V$.

## Prerequisites

Use Python 3.11–3.13 and a current `ado` source checkout.
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
5. **Predict:** PKH fits the final model and predicts pressure for the other
   1,195 entities.

PKH needs a finite pressure measurement from each selected entity before it
can select the next one.

The selection step is **herding**: PKH tries to match the sample's leaf
proportions to those of the whole space.
The [algorithm guide](../operators/pkh.md#how-pkh-works) shows a small example
and the equations.

Existing target measurements count toward `pilotSize`. The budget counts new
measurements. This tutorial starts with an empty store, so PKH collects all 20.

!!! tip "Choose the final predictor"

    With the default `finalPredictor: random_forest`, AutoGluon fits a Random
    Forest. Set `finalPredictor: autogluon` to let AutoGluon compare models.

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

### Read the predictions

PKH saves the `predictions` table in a `DataContainer` linked to the operation.
Find it using the PKH operation identifier printed in Step 2.

```bash
OPERATION_IDENTIFIER="<your PKH operation identifier>"
ado show related operation "$OPERATION_IDENTIFIER" --max-hops 1
```

Copy the `datacontainer-…` identifier from the output and export the resource
to JSON for the calculation below.

```bash
ado get datacontainer "<datacontainer-id>" -o json > pkh-results.json
```

### Compare a prediction with the ideal gas law

The table contains 20 measured pressures and 1,195 predictions. Select a row
where `is_predicted` is true and calculate $P = nRT/V$ for that gas condition.

```python
from pathlib import Path

from ado.core.datacontainer.resource import DataContainerResource

resource = DataContainerResource.model_validate_json(
    Path("pkh-results.json").read_text()
)
table = resource.config.tabularData["predictions"].dataframe()
row = table.loc[table["is_predicted"]].iloc[0]

ideal_pressure = row["mol"] * 8.314462618 * row["temperature"] / row["volume"]
print(f"Model estimate: {row['pressure']:.1f} Pa")
print(f"Ideal gas law:  {ideal_pressure:.1f} Pa")
```

### Check the sampling diagnostic

PKH also reports **MMD²**. It measures how closely the measured entities match
the whole space in the forest's leaf representation. A lower value means a
closer match. Use the ideal-gas comparison above to check prediction accuracy.

Read `metadata.pkhMmdSquared` on the PKH operation.

```bash
ado get operation "$OPERATION_IDENTIFIER" -o yaml
```

PKH records only the final value, because each forest refit changes the kernel.
See the [MMD reference](../operators/pkh.md#mmd-statistic) for details.

## Next Steps

- Try a different `numberEntities`, `pilotSize`, or `epochLength`.
  Compare the predictions with the ideal gas law for several conditions.
- Use the [configuration guide](../operators/pkh.md#configuration-parameters)
  to adapt PKH to another finite discovery space.
- Try [TRIM](trim.md) to stop sampling when holdout scores stabilize or the run
  reaches its budget.
