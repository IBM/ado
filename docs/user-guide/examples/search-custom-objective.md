<!-- markdownlint-disable code-block-style -->

# Search based on a custom objective function

!!! abstract "Intermediate tutorial"

    This walkthrough assumes you are already familiar with the core `ado`
    workflow — discovery spaces, operations, and replay. If you are new to
    `ado`, work through
    [Beyond the basics: `ado` with real data](tutorials/random-walk.md)
    first. If you haven't already, also work through
    [Search a space with an optimizer](best-configuration-search.md) to get
    familiar with `ray_tune`.

Often, experiments do not directly produce the value you care about. For
example, an experiment might measure the run time of an application, while
**the meaningful metric is cost — which requires knowing the price per hour
of the hardware used**. Another common scenario is aggregating several
measurements into a single score.

In this example we install a custom experiment that calculates a cost from
the workload configurations used in the
[Beyond the basics: `ado` with real data](tutorials/random-walk.md) example.
When the space is explored with a random walk, both `wallClockRuntime` and
the derived `cost` are measured in one pass.

We will:

1. Install the `ray_tune` operator and the custom experiment package
2. Inspect the custom experiment's interface
3. Define a discovery space that uses both the `replay` experiment and the
   custom cost function
4. Run a random walk and inspect the measurements

!!! warning "Prerequisites"

    - `ado-core` installed (`pip install ado-core`)
    - The example package cloned from GitHub (the wheel is not published to
      PyPI)

    Clone the repository if you have not already done so:

    ```bash
    git clone https://github.com/ibm/ado.git
    cd ado
    ```

    A pre-populated sample store is also required. Follow the
    [instructions in the random walk example](tutorials/random-walk.md#step-1-load-the-benchmark-data)
    to load the `ml-multi-cloud` dataset, then export its identifier:

    ```bash
    export SAMPLE_STORE_IDENTIFIER=<your-samplestore-identifier>
    ```

    All commands in this walkthrough are run from the **repository root**
    (`ado/`).

!!! example "TL;DR"

    Once the packages are installed and `SAMPLE_STORE_IDENTIFIER` is set:

    <!-- markdownlint-disable MD013 -->
    ```bash
    ado create space -f examples/ml-multi-cloud/ml_multicloud_space_with_custom.yaml --set "sampleStoreIdentifier=$SAMPLE_STORE_IDENTIFIER"
    ado create operation -f examples/ml-multi-cloud/randomwalk_ml_multicloud_operation.yaml --use-latest space
    ado show measurements operation --use-latest
    ```
    <!-- markdownlint-enable MD013 -->

## Step 1 — Install the required packages

### The `ray_tune` operator

The `ray_tune` operator is distributed as a separate package:

```bash
pip install ado-ray-tune
```

Confirm it is registered:

```bash
ado get operators
```

You should see `ray_tune` listed:

```text
Available operators by type:
┌───────┬─────────────┬─────────┬─────────┐
│ INDEX │ OPERATOR    │ VERSION │ TYPE    │
├───────┼─────────────┼─────────┼─────────┤
│ 0     │ random_walk │ 2.0.0   │ explore │
│ 1     │ ray_tune    │ 2.0.3   │ explore │
│ 2     │ rifferla    │ 2.0.3   │ modify  │
└───────┴─────────────┴─────────┴─────────┘
```

### The `ml-multicloud-cost` custom experiment

The example ships a custom experiment that derives a cost from two inputs: the
number of nodes in the configuration and the run time measured by the
`benchmark_performance` experiment.

Install it from the example directory:

```bash
pip install examples/ml-multi-cloud/custom_experiment/
```

Confirm `ado` can see it:

```bash
ado get experiments --details
```

<!-- markdownlint-disable line-length -->

```text
┌───────┬────────────────────┬─────────────────────┬─────────┬─────────────┐
│ INDEX │ ACTUATOR ID        │ EXPERIMENT ID       │ VERSION │ DESCRIPTION │
├───────┼────────────────────┼─────────────────────┼─────────┼─────────────┤
│ 0     │ custom_experiments │ ml-multicloud-cost  │ 1.0.0   │             │
│ 1     │ mock               │ test-experiment     │ None    │             │
│ 2     │ mock               │ test-experiment-two │ None    │             │
└───────┴────────────────────┴─────────────────────┴─────────┴─────────────┘
```

<!-- markdownlint-enable line-length -->

Inspect the experiment interface:

```bash
ado describe experiment ml-multicloud-cost
```

<!-- markdownlint-disable line-length -->

```terminaloutput
Identifier: custom_experiments.ml-multicloud-cost@1.0.0
Version: 1.0.0

Required Inputs:

   Constitutive Properties:
    ─────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: nodes
     Domain:

        Type: DISCRETE_VARIABLE_TYPE
        Interval: 1
        Range: [0, 1000]

    ─────────────────────────────────────────────────────────────────────────────────────────────────
    ─────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: cpu_family
     Domain:

        Type: DISCRETE_VARIABLE_TYPE
        Values: [0, 1]

    ─────────────────────────────────────────────────────────────────────────────────────────────────
   Observed Properties:

      benchmark_performance-wallClockRuntime


Outputs:
 ───────────────────────────────────────────────────────────────────────────────────────────────────────
   ml-multicloud-cost@v1-total_cost
 ───────────────────────────────────────────────────────────────────────────────────────────────────────
```

<!-- markdownlint-enable line-length -->

The `ml-multicloud-cost` experiment takes `nodes` and `cpu_family` as
constitutive properties (entity-space dimensions) and `wallClockRuntime` as
an **observed property** — a measurement produced by another experiment. The
single output, `total_cost`, is the derived metric.

!!! tip

    The `benchmark_performance-wallClockRuntime` observed property identifier
    tells you both the source experiment (`benchmark_performance`) and the
    property name (`wallClockRuntime`). Both experiments must appear in the
    same discovery space.

## Step 2 — Define a discovery space

The file `examples/ml-multi-cloud/ml_multicloud_space_with_custom.yaml`
extends the base space from the random walk example with an additional
`ml-multicloud-cost` entry:

```yaml
experiments:
  - experimentIdentifier: "benchmark_performance"
    actuatorIdentifier: "replay"
  - experimentIdentifier: "ml-multicloud-cost"
    actuatorIdentifier: "custom_experiments"
    experimentVersion: 1.0.0
```

Create the space, substituting the sample store identifier you exported
earlier:

<!-- markdownlint-disable MD013 -->
```bash
ado create space -f examples/ml-multi-cloud/ml_multicloud_space_with_custom.yaml --set "sampleStoreIdentifier=$SAMPLE_STORE_IDENTIFIER"
```
<!-- markdownlint-enable MD013 -->

```text
Success! Created space with identifier: $DISCOVERY_SPACE_IDENTIFIER
```

!!! important

    If an experiment takes the output of another experiment as input, both
    experiments **must** appear in the same discovery space. Omitting
    `benchmark_performance` in the example above would cause `ado create space`
    to fail with:

    **SpaceInconsistencyError**: MeasurementSpace does not contain an experiment
    measuring an observed property required by another experiment in the space

Inspect the space to confirm it contains both experiments:

```bash
ado describe space --use-latest
```

## Step 3 — Run an operation

Run a random walk on the new space:

<!-- markdownlint-disable MD013 -->
```bash
ado create operation -f examples/ml-multi-cloud/randomwalk_ml_multicloud_operation.yaml --use-latest space
```
<!-- markdownlint-enable MD013 -->

This behaves identically to the
[random walk example](tutorials/random-walk.md#step-3-run-a-random-walk), except
that `ado` now evaluates both experiments for each visited entity. The
terminal output will show additional detail related to the dependent
experiment.

## Step 4 — Inspect the measurements

Retrieve the full table of measurements:

```bash
ado show measurements operation --use-latest
```

<!-- markdownlint-disable line-length -->

```text
┌───────────────┬──────────────┬─────────────────────────────────────────────┬─────────────────────────────────────────────┬────────────┬───────┬──────────┬───────────┬────────────────────┬──────────────┬────────────────────┬───────┐
│ request_index │ result_index │ identifier                                  │ experiment_id                               │ cpu_family │ nodes │ provider │ vcpu_size │ wallClockRuntime   │ status       │ total_cost         │ valid │
├───────────────┼──────────────┼─────────────────────────────────────────────┼─────────────────────────────────────────────┼────────────┼───────┼──────────┼───────────┼────────────────────┼──────────────┼────────────────────┼───────┤
│ 1             │ 0            │ B_f0.0-c0.0-n3                              │ replay.benchmark_performance                │ 0.0        │ 3     │ B        │ 0.0       │ 153.51639366149902 │ ok           │ not_measured       │ True  │
│ 1             │ 0            │ B_f0.0-c0.0-n3                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ 3     │ B        │ 0.0       │ not_measured       │ not_measured │ 1.2793032805124918 │ True  │
└───────────────┴──────────────┴─────────────────────────────────────────────┴─────────────────────────────────────────────┴────────────┴───────┴──────────┴───────────┴────────────────────┴──────────────┴────────────────────┴───────┘
```

<!-- markdownlint-enable line-length -->

Each entity produces two rows — one per experiment. The `replay` row carries
`wallClockRuntime`; the `custom_experiments` row carries `total_cost`. Note
that each row only contains values for the properties measured by its
respective experiment (`not_measured` elsewhere).

## Summary

| Step | What you did | `ado` concept |
| :--- | :----------- | :------------ |
| 1 | Installed `ray_tune` and the custom experiment package | Operator / custom experiment |
| 2 | Defined a space with both `replay` and the cost function | Discovery space / dependent experiment |
| 3 | Ran a random walk that evaluated both experiments per entity | Operation / `random_walk` operator |
| 4 | Retrieved measurements including the derived `total_cost` | `ado show measurements` |

## Going further

Try extending this example:

- **Use a different operator** — swap `random_walk` for `ray_tune` to search
  for the configuration with the lowest cost; see
  [Search a space with an optimizer](best-configuration-search.md)
- **Compose multiple derived metrics** — add a second custom experiment that
  consumes `total_cost` as an observed property to compute, for example, a
  cost-per-throughput ratio
- **Change the cost formula** — edit `objective_function.py` in
  `examples/ml-multi-cloud/custom_experiment/` and bump the experiment
  version to keep stored measurements consistent
- **Inspect space-level measurements** — run `ado show measurements space
  --use-latest` to see all measurements accumulated across operations, not
  just the latest one

## What's next

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable no-inline-html -->
<!-- markdownlint-disable MD046 -->
<!-- prettier-ignore-start -->

<div class="grid cards" markdown>

- :octicons-rocket-24:{ .lg .middle } **Search using an optimizer**

    ---

    Try the [Search a space with an optimizer](best-configuration-search.md)
    example to see how you can drive `ray_tune` over a space to find the
    best-performing configuration.

    [Search a space with an optimizer :octicons-arrow-right-24:](best-configuration-search.md)

- :octicons-workflow-24:{ .lg .middle } **Discovering important entity space dimensions**

    ---

    Use `ado` to identify which entity space dimensions most influence a
    target metric.

    [Identify the important dimensions of a space :octicons-arrow-right-24:](lhu.md)

</div>

<!-- prettier-ignore-end -->
<!-- markdownlint-enable MD046 -->
<!-- markdownlint-enable no-inline-html -->
<!-- markdownlint-enable line-length -->
