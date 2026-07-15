<!-- markdownlint-disable code-block-style -->

# Beyond the basics: `ado` with real data

!!! question "New to `ado`?"

    This walkthrough introduces the key concepts — **sample stores**,
    **discovery spaces**, and **operations** — as we go. If you'd prefer a
    full overview first, the [Concepts](../../../concepts/core-concepts.md)
    page has you covered.

This walkthrough takes you end-to-end through a real-data `ado` workflow using
an imported cloud benchmark dataset, so you can focus on how `ado` works with
pre-existing measurements rather than synthetic examples.

Once imported, `ado`'s **replay** mechanism transparently serves the benchmark
results as the operation runs — no re-execution needed.

We will:

1. Load pre-existing benchmark data into a **sample store**
2. Define a **discovery space** over the cloud hardware configurations
3. Run a **random walk** operation to sample the space

!!! warning "Prerequisites"

    - `ado-core` installed (`pip install ado-core`)
    - The repository cloned from GitHub:

      ```bash
      git clone https://github.com/ibm/ado.git
      cd ado
      ```

    All commands are run from the **repository root** (`ado/`).

!!! example "TL;DR"

    <!-- markdownlint-disable MD013 -->
    ```bash
    ado create store -f examples/ml-multi-cloud/ml_multicloud_sample_store_from_root.yaml
    ado create space -f examples/ml-multi-cloud/ml_multicloud_space.yaml --use-latest samplestore
    ado create operation -f examples/ml-multi-cloud/randomwalk_ml_multicloud_operation.yaml --use-latest space
    ado show measurements operation --use-latest
    ```
    <!-- markdownlint-enable MD013 -->

## Step 1 — Load the benchmark data

This example uses pre-existing benchmark data from `ml_export.csv` — results of
running a workload on different cloud hardware configurations across three
providers.

In `ado`, configurations are called **entities** and are stored, together with
measurement results, in a [**sample store**](/ado/resources/sample-stores).

The file `examples/ml-multi-cloud/ml_multicloud_sample_store_from_root.yaml`
tells `ado` how to import the CSV:

<!-- prettier-ignore-start -->

```yaml
{%
    include "../../../../examples/ml-multi-cloud/ml_multicloud_sample_store_from_root.yaml"
%}
```

<!-- prettier-ignore-end -->

Create the sample store:

<!-- markdownlint-disable MD013 -->

```bash
ado create store -f examples/ml-multi-cloud/ml_multicloud_sample_store_from_root.yaml
```

<!-- markdownlint-enable MD013 -->

```text
Success! Created sample store with identifier $SAMPLE_STORE_IDENTIFIER
```

!!! tip

    The sample store only needs to be created once. It can be reused across
    multiple discovery spaces and examples that need the `ml_export.csv` data.

## Step 2 — Define a discovery space

A **discovery space** brings together:

- an **entity space** — the set of points to measure
- a **measurement space** — the experiments to apply to them

The file `examples/ml-multi-cloud/ml_multicloud_space.yaml` describes the space
of cloud configurations and maps them to the `replay.benchmark_performance`
experiment:

<!-- prettier-ignore-start -->

```yaml
{% include "../../../../examples/ml-multi-cloud/ml_multicloud_space.yaml" %}
```

<!-- prettier-ignore-end -->

Create the discovery space, linking it to the sample store you just created:

<!-- markdownlint-disable MD013 -->

```bash
ado create space -f examples/ml-multi-cloud/ml_multicloud_space.yaml --use-latest samplestore
```

<!-- markdownlint-enable MD013 -->

```text
Success! Created space with identifier: $DISCOVERY_SPACE_IDENTIFIER
```

Inspect it to confirm the entity space dimensions and experiment interface:

```bash
ado describe space --use-latest
```

```terminaloutput
Entity Space:

   Number of entities: 48

   Categorical properties:

      name     ┃ values
     ━━━━━━━━━━╋━━━━━━━━━━━━━━━━━
      provider ┃ ['A', 'B', 'C']

   Discrete properties:

      name       ┃ range ┃ interval ┃ values
     ━━━━━━━━━━━━╋━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━━━━
      cpu_family ┃ None  ┃ None     ┃ [0, 1]
      vcpu_size  ┃ None  ┃ None     ┃ [0, 1]
      nodes      ┃ None  ┃ None     ┃ [2, 3, 4, 5]


Measurement Space:

   Experiments:

      base identifier              ┃ required major version ┃ parameterization
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━
      replay.benchmark_performance ┃ None                   ┃ None
```

!!! example "Have a look"

    The entity space has 48 entities — the Cartesian product of `provider`
    (3), `cpu_family` (2), `vcpu_size` (2), and `nodes` (4). Compare this to
    the number of rows in `ml_export.csv`.

## Step 3 — Run a random walk

An **operation** drives an **operator** over the discovery space. Here we use
`random_walk`, which samples entities at random.

The file `examples/ml-multi-cloud/randomwalk_ml_multicloud_operation.yaml`
configures the walk:

<!-- markdownlint-disable MD013 -->
<!-- prettier-ignore-start -->

```yaml
{%
  include "../../../../examples/ml-multi-cloud/randomwalk_ml_multicloud_operation.yaml"
%}
```

<!-- prettier-ignore-end -->

<!-- markdownlint-enable MD013 -->

Run the operation:

<!-- markdownlint-disable MD013 -->

```bash
ado create operation -f examples/ml-multi-cloud/randomwalk_ml_multicloud_operation.yaml --use-latest space
```

<!-- markdownlint-enable MD013 -->

!!! tip

    The `spaces:` field of the operation contains a placeholder identifier.
    The `--use-latest space` flag replaces it with the discovery
    space you just created.

`ado` prints a line for each entity as it is measured. When finished, you will
see a summary like:

```yaml
identifier: random_walk@2.0.0-31d4c6
kind: operation
metadata:
  entities_submitted: 48
  experiments_requested: 74
status:
  - event: finished
    exit_state: success
    recorded_at: "2026-07-09T09:26:52.072109Z"
```

!!! tip "Memoization"

    `ado` transparently **reuses** any measurement already in the sample store
    rather than re-running it. The operator does not need to know whether a
    result will come from live execution or replay — `ado` handles it
    automatically.

## Step 4 — View the results

```bash
ado show measurements operation --use-latest
```

<!-- markdownlint-disable MD013 -->

```text
┌───────────────┬──────────────┬────────────────────────┬──────────────────────────────┬────────────┬───────┬──────────┬───────────┬────────────────────┬──────────────┬───────┐
│ request_index │ result_index │ identifier             │ experiment_id                │ cpu_family │ nodes │ provider │ vcpu_size │ wallClockRuntime   │ status       │ valid │
├───────────────┼──────────────┼────────────────────────┼──────────────────────────────┼────────────┼───────┼──────────┼───────────┼────────────────────┼──────────────┼───────┤
│ 0             │ 0            │ A_f1.0-c0.0-n4         │ replay.benchmark_performance │ 1.0        │ 4     │ A        │ 0.0       │ 158.70639538764954 │ ok           │ True  │
│ 1             │ 0            │ A_f1.0-c0.0-n2         │ replay.benchmark_performance │ 1.0        │ 2     │ A        │ 0.0       │ 378.31657004356384 │ ok           │ True  │
│ 2             │ 0            │ B_f0.0-c0.0-n3         │ replay.benchmark_performance │ 0.0        │ 3     │ B        │ 0.0       │ 153.51639366149902 │ ok           │ True  │
│ ...           │ ...          │ ...                    │ ...                          │ ...        │ ...   │ ...      │ ...       │ ...                │ ...          │ ...   │
└───────────────┴──────────────┴────────────────────────┴──────────────────────────────┴────────────┴───────┴──────────┴───────────┴────────────────────┴──────────────┴───────┘
```

<!-- markdownlint-enable MD013 -->

!!! warning

    Some entities have `valid: False` — these are points in the discovery space
    for which no matching data existed in `ml_export.csv`. The `status`
    column shows `not_measured` for those rows.

To see results aggregated across the full space (not just this operation):

```bash
ado show measurements space --use-latest
```

!!! example

    Run the operation a second time. Because `ado` **memoizes** results,
    measurements already in the sample store are reused automatically — no
    duplicate computation.

## Summary

| Step | What you did                                            | `ado` concept           |
| :--- | :------------------------------------------------------ | :---------------------- |
| 1    | Imported benchmark data into a sample store             | Sample store / replay   |
| 2    | Described the cloud configuration space in `space.yaml` | Discovery space         |
| 3    | Ran a random walk to sample 48 entities                 | Operation / operator    |
| 4    | Retrieved measured results                              | `ado show measurements` |

## What's next

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable no-inline-html -->
<!-- prettier-ignore-start -->

<div class="grid cards" markdown>

- :octicons-rocket-24:{ .lg .middle } **Search using an optimizer**

    ---

    Use RayTune to find the best-performing cloud configuration — a natural
    next step once the space is defined.

    [Search a space with an optimizer :octicons-arrow-right-24:](../best-configuration-search.md)

- :octicons-workflow-24:{ .lg .middle } **Discover important dimensions**

    ---

    Identify which hardware parameters most influence runtime using `ado`'s
    importance analysis.

    [Identify the important dimensions of a space :octicons-arrow-right-24:](../lhu.md)

</div>

<!-- prettier-ignore-end -->

<!-- markdownlint-enable no-inline-html -->
<!-- markdownlint-enable line-length -->
