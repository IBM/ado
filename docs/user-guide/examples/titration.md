<!-- markdownlint-disable code-block-style -->

# Simulate an acid–base titration

!!! question "New to `ado`?"

    This walkthrough uses the same concepts as
    [Your first ado experiment](tutorials/density-example.md): **custom
    experiments**, **discovery spaces**, and **operations**. If you want an
    overview first, read the [Concepts](../../concepts/core-concepts.md) page.

How much base does it take to neutralise a sample of acid? In a titration, you
add base to the acid in small steps and read the pH after each step. The pH
stays low, then rises sharply when the base has neutralised the acid. This
point is the **equivalence point**.

This walkthrough uses a simulated titration. The particularity of this example
is that the pH changes by about ten units within a few millilitres of the
equivalence point and changes little elsewhere. Such abrupt changes, very common
in real-world environments, represent challenging structures to learn for
predictive models.

We will:

1. Inspect the `titrate_strong_acid` custom experiment
2. Measure the pH at one point on each side of the equivalence point
3. Create a **discovery space** of 1,830 titration conditions
4. Run an **operation** that measures 20 of them
5. Read the measurements

!!! warning "Prerequisites"

    - `ado-core` installed (`pip install ado-core`)
    - The example package cloned from GitHub (the wheel is not published to
      PyPI)

    Clone the repository if you have not already done so:

    ```bash
    git clone https://github.com/ibm/ado.git
    cd ado
    ```

    Then install the example package into your environment:

    ```bash
    pip install -e examples/titration/custom_experiments/
    ```

    Run all commands in this walkthrough from the **repository root**
    (`ado/`).

!!! example "TL;DR"

    After you install the package, save the YAML from Step 2 as
    `titration_point.yaml`. Then run:

    <!-- markdownlint-disable MD013 -->
    ```bash
    run_experiment titration_point.yaml
    ado create space -f examples/titration/example_yamls/space_titration.yaml --new-sample-store
    ado template operation --operator-name random_walk --output-file titration_walk.yaml
    ado create operation -f titration_walk.yaml --use-latest space --set operation.parameters.numberEntities=20
    ado show measurements space --use-latest --property ph -o csv
    ```
    <!-- markdownlint-enable MD013 -->

## Step 1 — The custom experiment

A **custom experiment** is a Python function that `ado` can call. The
`titrate_strong_acid` experiment simulates the pH of a 25 mL sample of strong
acid at 25 °C while you add a strong base to it. It takes three inputs and
returns one output, `ph`.

| Input | Meaning | Unit |
| :--- | :--- | :--- |
| `acid_concentration` | Concentration of the acid sample | mol/L |
| `base_concentration` | Concentration of the base | mol/L |
| `titrant_volume` | Volume of base added to the sample | mL |

Check that `ado` can see the experiment:

```bash
ado describe experiment titrate_strong_acid
```

The output lists the three inputs and the `ph` output:

```terminaloutput
Identifier: custom_experiments.titrate_strong_acid
...
Required Inputs:
   Constitutive Properties:
     Identifier: acid_concentration
     ...
     Identifier: base_concentration
     ...
     Identifier: titrant_volume
     ...
Outputs:
   titrate_strong_acid-ph
```

### The titration equations

After you add $V$ mL of base to the $V_s = 25$ mL sample, the excess acid
concentration is

$$
c = \frac{c_{\mathrm{acid}} V_s - c_{\mathrm{base}} V}{V_s + V}.
$$

With the ion product of water $K_w = 10^{-14}$, the experiment calculates the
hydrogen ion concentration and the pH:

$$
[\mathrm{H}^+] = \frac{c + \sqrt{c^2 + 4K_w}}{2}, \qquad
\mathrm{pH} = -\log_{10}[\mathrm{H}^+].
$$

The equivalence point is at $V_{\mathrm{eq}} = c_{\mathrm{acid}} V_s /
c_{\mathrm{base}}$.

The simulated pH meter adds Gaussian noise with a standard deviation of 0.05
pH units. The inputs seed the noise, so the same inputs always give the same
reading.

## Step 2 — Test on a single point

Test the experiment on one point before you use it in a campaign. For a
0.1 mol/L acid and a 0.2 mol/L base, the equivalence point is at 12.5 mL.
Save this configuration as `titration_point.yaml`. It measures the pH at
12 mL, just before the equivalence point.

```yaml
entity:
  acid_concentration: 0.1
  base_concentration: 0.2
  titrant_volume: 12.0
experiments:
  - actuatorIdentifier: custom_experiments
    experimentIdentifier: titrate_strong_acid
```

Run it:

```bash
run_experiment titration_point.yaml
```

The result includes the `ph` reading. The output here is shortened:

<!-- markdownlint-disable MD013 -->

```text
Point: {'acid_concentration': 0.1, 'base_concentration': 0.2, 'titrant_volume': 12.0}
Executing: custom_experiments.titrate_strong_acid
Result:
[
    ...
    experiment_id    custom_experiments.titrate_strong_acid
    valid            True
    ph               2.55968
]
```

<!-- markdownlint-enable MD013 -->

Change `titrant_volume` to `13.0` and run the command again. The `ph` is now
11.389963. Half a millilitre after the equivalence point, the sample is basic.

The readings include meter noise. Without noise, the equations give 2.57 and
11.42.

!!! tip

    `run_experiment` is a quick test tool. `ado` does not store its results.
    See [Running experiments on single entities](../advanced/run-experiment.md)
    for details.

## Step 3 — Define a discovery space

A **discovery space** has two parts:

- an **entity space**, the set of points to measure
- a **measurement space**, the experiments to run on them

The file `examples/titration/example_yamls/space_titration.yaml` defines a
grid:

<!-- prettier-ignore-start -->

```yaml
{% include "../../../examples/titration/example_yamls/space_titration.yaml" %}
```

<!-- prettier-ignore-end -->

The grid has 10 × 3 × 61 = **1,830 entities**. Each acid–base pair has 61
volumes. The measurement space has one experiment, `titrate_strong_acid`.

| Property | Values | Number of values |
| :--- | :--- | ---: |
| `acid_concentration` | 0.02 to 0.20 mol/L in steps of 0.02 | 10 |
| `base_concentration` | 0.1, 0.2, and 0.4 mol/L | 3 |
| `titrant_volume` | 0 to 60 mL in steps of 1 | 61 |

Create the space with an empty sample store:

```bash
ado create space \
  -f examples/titration/example_yamls/space_titration.yaml --new-sample-store
```

The command prints the identifier of the new space. Your identifier is
different.

```text
Success! Created space with identifier: space-cf0845-d44df2
```

To check the space, run `ado describe space --use-latest`. The output starts
with `Number of entities: 1830`.

## Step 4 — Run an operation

An **operation** applies an **operator** to a discovery space. The operator
chooses which entities to measure. The `random_walk` operator chooses them in
random order. See the [Random Walk guide](../operators/random-walk.md) for its
options.

Create a configuration from the built-in template:

```bash
ado template operation --operator-name random_walk \
  --output-file titration_walk.yaml
```

Run the operation on the space that you created. Set the budget to 20
entities:

```bash
ado create operation -f titration_walk.yaml --use-latest space \
  --set operation.parameters.numberEntities=20
```

`--use-latest space` puts the identifier of your latest space into the
configuration. When the operation ends, `ado` prints:

<!-- markdownlint-disable MD013 -->

```text
Success! Created operation with identifier random_walk@2.1.0-6926b6 and it finished successfully.
```

<!-- markdownlint-enable MD013 -->

## Step 5 — View the results

Read the measurements that are stored for the space:

```bash
ado show measurements space --use-latest --property ph -o csv
```

The output has one row for each of the 20 measured entities. The entities and
readings in your run are different:

<!-- markdownlint-disable MD013 -->

```text
,identifier,ph
0,acid_concentration.0.02-base_concentration.0.2-titrant_volume.7,12.418498077167238
1,acid_concentration.0.06-base_concentration.0.2-titrant_volume.31,12.903444859373977
2,acid_concentration.0.14-base_concentration.0.1-titrant_volume.0,0.8135362233505194
...
```

<!-- markdownlint-enable MD013 -->

!!! note "Where are the readings in between?"

    Most readings are either below 2 or above 11. Without meter noise, the pH
    is between 3 and 11 for only 17 of the 1,830 entities (0.9%). The volume
    of each of these entities equals an equivalence volume. A random sample of
    20 entities contains none of them in about 83% of draws.

## Summary

| Step | What you did | `ado` concept |
| :--- | :--- | :--- |
| 1 | Inspected the titration experiment | Custom experiment |
| 2 | Measured the pH at 12 mL and 13 mL | `run_experiment` |
| 3 | Created a discovery space of 1,830 entities | Discovery space |
| 4 | Measured 20 entities with `random_walk` | Operation / operator |
| 5 | Read the measured pH values | `ado show measurements` |

## What's next

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable no-inline-html -->
<!-- prettier-ignore-start -->

<div class="grid cards" markdown>

- :octicons-graph-24:{ .lg .middle } **Build a surrogate model**

    ---

    Use the TRIM operator to sample a space with a stopping rule, and train a
    model that predicts the unmeasured entities.

    [Quickly building a predictive model :octicons-arrow-right-24:](trim.md)

- :octicons-rocket-24:{ .lg .middle } **Search with an optimizer**

    ---

    Use `ray_tune` to look for the best point in a space instead of sampling
    it at random.

    [Search a space with an optimizer :octicons-arrow-right-24:](best-configuration-search.md)

</div>

<!-- prettier-ignore-end -->

<!-- markdownlint-enable no-inline-html -->
<!-- markdownlint-enable line-length -->
