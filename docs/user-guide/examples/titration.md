# Simulate an acid–base titration

How much base does it take to neutralise a sample of acid? In this example
you will measure pH on either side of the equivalence point.
You will then create a discovery space, sample 20 combinations of acid
concentration, base concentration and volume, and inspect their pH values.

## Install the experiment

From the root of an `ado` checkout, run these commands in your activated
Python environment.

```bash
pip install -e . -e examples/titration/custom_experiments/
ado describe experiment titrate_strong_acid
```

The experiment description lists three inputs and one output, `ph`.

## The titration equations

The experiment adds $V$ mL of strong base to a $V_s = 25$ mL sample of
strong acid at 25 °C. For acid and base concentrations $c_{\mathrm{acid}}$
and $c_{\mathrm{base}}$ in mol/L, the excess acid concentration after mixing
is

$$
c = \frac{c_{\mathrm{acid}} V_s - c_{\mathrm{base}} V}{V_s + V}.
$$

Using the ion product of water $K_w = 10^{-14}$, the experiment calculates
the hydrogen ion concentration and pH.

$$
[\mathrm{H}^+] = \frac{c + \sqrt{c^2 + 4K_w}}{2}, \qquad
\mathrm{pH} = -\log_{10}[\mathrm{H}^+].
$$

At the **equivalence point**, the added base neutralises the acid. Its volume
is $V_{\mathrm{eq}} = c_{\mathrm{acid}} V_s / c_{\mathrm{base}}$. The pH
rises sharply around this volume. The simulated meter adds Gaussian noise
with a standard deviation of 0.05 pH units. Each combination of inputs seeds
the meter, so repeating a measurement gives the same reading.

## Measure one pH value

For a 0.1 mol/L acid and a 0.2 mol/L base, the equivalence point is 12.5 mL.
Save the following configuration as `titration_point.yaml` to measure pH just
before equivalence.

```yaml
entity:
  acid_concentration: 0.1
  base_concentration: 0.2
  titrant_volume: 12.0
experiments:
  - actuatorIdentifier: custom_experiments
    experimentIdentifier: titrate_strong_acid
```

```bash
run_experiment titration_point.yaml
```

The `ph` reading is about 2.56. Change `titrant_volume` to `13.0` and rerun
the command to see the pH on the other side of the equivalence point.

## Create the discovery space

The supplied discovery space defines 1,830 combinations of concentrations
and volumes.

| Property | Values | Number of values |
| :--- | :--- | ---: |
| `acid_concentration` | 0.02 to 0.20 mol/L in steps of 0.02 | 10 |
| `base_concentration` | 0.1, 0.2 and 0.4 mol/L | 3 |
| `titrant_volume` | 0 to 60 mL in steps of 1 | 61 |

Create the space with an empty sample store.

```bash
ado create space \
  -f examples/titration/example_yamls/space_titration.yaml --new-sample-store
```

## Measure 20 entities

Generate an operation configuration from the built-in
[Random Walk](../operators/random-walk.md) template. Its sampler selects
grid points in random order. Run it on the space you just created with a
budget of 20 entities.

```bash
ado template operation --operator-name random_walk \
  --output-file titration_walk.yaml
ado create operation -f titration_walk.yaml --use-latest space \
  --set operation.parameters.numberEntities=20
```

Read the stored measurements from the space.

```bash
ado show measurements space --use-latest --property ph -o csv
```

Each of the 20 rows gives an entity identifier and its measured pH.
