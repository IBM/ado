<!-- markdownlint-disable code-block-style -->

# Your first ado experiment

!!! abstract "New to `ado`?"

    This walkthrough introduces the key concepts — **custom experiments**,
    **discovery spaces**, and **operations** — as we go. If you'd prefer a
    full overview first, the [Concepts](../../concepts/index.md) page has you
    covered.

The simplest way to write an experiment in `ado` is a **custom experiment** —
a plain Python function decorated with `@custom_experiment`. No boilerplate,
no subclassing: if you can write a function, you can write an `ado` experiment.

This walkthrough takes you end-to-end through the full `ado` workflow using a
deliberately simple example — computing density from mass and volume — so you
can focus on the concepts rather than the domain.

We will:

1. Write a Python function and register it with `ado` as a **custom experiment**
2. Test it on a single point
3. Describe a **discovery space** over a grid of `mass` and `volume` values
4. Run an **operation** to sample the space and collect results

!!! warning "Prerequisites"

    - `ado-core` installed (`pip install ado-core`)
    - The example package cloned from GitHub (the wheel is not published to PyPI)

    Clone the repository if you have not already done so:

    ```bash
    git clone https://github.com/ibm/ado.git
    cd ado
    ```

    Then install the example package into your environment:

    ```bash
    pip install -e examples/density_example/
    ```

    All commands in this walkthrough are run from the **repository root** (`ado/`).

!!! example "TL;DR"

    Once the package is installed:

    <!-- markdownlint-disable MD013 -->
    ```bash
    run_experiment examples/density_example/point.yaml
    ado create space -f examples/density_example/space.yaml
    ado create operation -f examples/density_example/operation.yaml --use-latest space
    ado show measurements operation --use-latest
    ```
    <!-- markdownlint-enable MD013 -->

## Step 1 — The custom experiment

A **custom experiment** is an ordinary Python function decorated with
`@custom_experiment`. `ado` uses the function signature to infer the inputs
and registers the function name as the experiment identifier.

The density experiment lives in
`density/density.py`:

```python
{% include "../../../examples/density_example/density/density.py" %}
```

A few things to note:

- **Inputs** (`mass`, `volume`) are positional parameters and become the
  _required properties_ of the experiment. Because they are typed as `float`,
  `ado` infers a continuous domain for each.
- **Output** `density` is declared in `output_property_identifiers`. The
  function returns a dictionary with that key.

The package registers the function with `ado` via an
[entry point](../../developer-guide/creating-custom-experiments.md) in
`pyproject.toml`:

```toml
[project.entry-points."ado.custom_experiments"]
optimization_test_functions = "density.density"
```

After installing the package, confirm `ado` can see the experiment:

```bash
ado get experiments
```

You should see `calculate_density` listed under `custom_experiments`:

```text
Available experiments:
  actuator: custom_experiments
    - calculate_density
```

## Step 2 — Test on a single point

Before wiring the experiment into anything larger, it is useful to verify it
works on a single point. The file `examples/density_example/point.yaml`
defines one:

```yaml
{% include "../../../examples/density_example/point.yaml" %}
```

Run it with:

```bash
run_experiment examples/density_example/point.yaml
```

You should see output similar to:

```text
Results for entity mass=8 volume=4:
  density: 2.0
```

!!! tip

    `run_experiment` is a lightweight testing tool — results are not tracked or
    stored anywhere. It is purely for quick feedback during development.
    See [Running experiments on single entities](../actuators/run-experiment.md)
    for details.

## Step 3 — Define a discovery space

A **discovery space** brings together two things:

- an **entity space** — the set of points (entities) to measure
- a **measurement space** — the experiments to apply to them

The file `examples/density_example/space.yaml` defines a grid of ten masses
and ten volumes:

```yaml
{% include "../../../examples/density_example/space.yaml" %}
```

The entity space is 10 × 10 = **100 entities** (every combination of
`mass` and `volume`). The measurement space contains a single experiment:
`calculate_density`.

Create the discovery space:

```bash
ado create space -f examples/density_example/space.yaml
```

```text
Success! Created space with identifier: $DISCOVERY_SPACE_IDENTIFIER
```

Inspect it to confirm everything looks right:

```bash
ado describe space --use-latest
```

The output shows the entity space dimensions and the experiment interface:

```terminaloutput
Entity Space:

   Number of entities: 100

   Discrete properties:

      name   ┃ range ┃ interval ┃ values
     ━━━━━━━━╋━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      mass   ┃ None  ┃ None     ┃ [1.0, 2.5, 5.0, 10.0, 25.0, 50.0, ...]
      volume ┃ None  ┃ None     ┃ [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, ...]


Measurement Space:

   Experiments:

      base identifier                         ┃ required major version ┃ parameterization
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━
      custom_experiments.calculate_density    ┃ nan                    ┃ nan
```

## Step 4 — Run an operation

An **operation** drives an **operator** over the discovery space. The operator
decides which entities to visit and in what order. Here we use `random_walk`,
which samples entities at random.

The file `examples/density_example/operation.yaml` configures a random walk
that visits 10 of the 100 entities:

```yaml
{% include "../../../examples/density_example/operation.yaml" %}
```

!!! tip

    The `spaces:` field contains a real space identifier used during development
    of this example. The `--use-latest space` flag replaces it with the
    identifier of the discovery space you just created.

Create the operation:

```bash
ado create operation -f examples/density_example/operation.yaml --use-latest space
```

`ado` samples and measures entities, printing a line for each one as it
completes. When finished you will see a summary like:

```yaml
identifier: random_walk@2.0.0-a1b2c3
kind: operation
metadata:
  entities_submitted: 10
  experiments_requested: 10
status:
- event: finished
  exit_state: success
  recorded_at: '2026-01-01T12:00:00.000000Z'
```

## Step 5 — View the results

```bash
ado show measurements operation --use-latest
```

This prints every entity sampled in the operation alongside its measured
density:

<!-- markdownlint-disable MD013 -->

```text
┌───────────────┬──────────────┬──────────────────────────┬──────────┬────────┬────────────────────┬───────┐
│ request_index │ result_index │ identifier               │ mass     │ volume │ density            │ valid │
├───────────────┼──────────────┼──────────────────────────┼──────────┼────────┼────────────────────┼───────┤
│ 0             │ 0            │ mass.1.0-volume.0.5      │ 1.0      │ 0.5    │ 2.0                │ True  │
│ 1             │ 0            │ mass.50.0-volume.5.0     │ 50.0     │ 5.0    │ 10.0               │ True  │
│ ...           │ ...          │ ...                      │ ...      │ ...    │ ...                │ ...   │
└───────────────┴──────────────┴──────────────────────────┴──────────┴────────┴────────────────────┴───────┘
```

<!-- markdownlint-enable MD013 -->

To see measured results aggregated back onto the full space (not just the
entities visited in this operation):

```bash
ado show measurements space --use-latest
```

!!! example

    Run the operation a second time. Because `ado` **memoizes** results,
    measurements already in the sample store are reused automatically — no
    duplicate computation.

## Summary

| Step | What you did | `ado` concept |
| :--- | :----------- | :------------ |
| 1 | Decorated a Python function with `@custom_experiment` | Custom experiment |
| 2 | Verified it on one point with `run_experiment` | Point testing |
| 3 | Described the space of things to measure in `space.yaml` | Discovery space |
| 4 | Ran a random walk over 10 of the 100 entities | Operation / operator |
| 5 | Retrieved the measured results | `ado show measurements` |

## What's next

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable no-inline-html -->
<!-- prettier-ignore-start -->

<div class="grid cards" markdown>

- :octicons-graph-24:{ .lg .middle } **Explore a real dataset**

    ---

    Walk through a cloud workload parameter space using pre-existing benchmark
    data and `ado`'s replay mechanism.

    [Taking a random walk :octicons-arrow-right-24:](random-walk.md)

- :octicons-rocket-24:{ .lg .middle } **Search with an optimizer**

    ---

    Use RayTune to find the best point in a space — a natural next step once
    you can define custom experiments.

    [Search a space with an optimizer :octicons-arrow-right-24:](best-configuration-search.md)

</div>

<!-- prettier-ignore-end -->
<!-- markdownlint-enable no-inline-html -->
<!-- markdownlint-enable line-length -->
