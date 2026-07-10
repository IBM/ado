# Taking a random walk

> [!NOTE] The scenario
>
> When deploying a workload, you need to configure parameters such as the number
> of CPUs or the type of GPU. **In this example, `ado` is used to explore how
> performance varies across the workload parameter space for a cloud
> application.**
>
> Exploring a workload parameter space with `ado` involves:
>
> 1. Defining the values of the workload parameters to test and how to measure
>    them using a `discoveryspace`
> 2. Exploring the `discoveryspace` by creating an `operation` that samples
>    points and measures them
> 3. Getting the results of the `operation`

<!-- markdownlint-disable-next-line MD028 -->

> [!IMPORTANT] Prerequisites
>
> - Install the following Python package locally:
>
> ```bash
> pip install ado-core
> ```

<!-- markdownlint-disable line-length -->

> [!TIP] TL;DR
>
> To create the `discoveryspace` and explore it with a random walk execute:
>
> ```bash
> : # Create the space to explore (also creates the samplestore)
> ado create space -f ml_multicloud_space.yaml --with store=ml_multicloud_sample_store.yaml
> : # Explore!
> ado create operation -f randomwalk_ml_multicloud_operation.yaml --use-latest space
> ```

<!-- markdownlint-enable line-length -->

## Using pre-existing data with `ado`

For this example we will use some **pre-existing data**. This makes the example
simpler and quicker to execute but can also be useful in other situations. The
data is in the file `ml_export.csv` and consists of results of running a
benchmark on different cloud hardware configurations from different providers.

In `ado` such configurations are called `entities`, and are stored, along with
the results of measurements executed on them, in a
[`samplestore`](/ado/resources/sample-stores). Let's start by copying the data
in `ml_export.csv` into a new `samplestore`.

To do this execute,

```commandline
ado create store -f ml_multicloud_sample_store.yaml
```

and it will report that a `samplestore` has been created:

```commandline
Success! Created sample store with identifier $SAMPLE_STORE_IDENTIFIER
```

You can see all available sample stores using `ado get samplestores`.

<!-- markdownlint-disable code-block-style -->

!!! info end

    You only need to create this `samplestore` once.
    It can be reused in multiple `discoveryspaces`
    or examples that require the `ml_export.csv` data.

<!-- markdownlint-enable code-block-style -->

## Creating a `discoveryspace` for the `ml-multi-cloud` data

A `discoveryspace` describes a set of points and how to measure them. Here we
will create a `discoveryspace` to describe the space explored in
`ml_export.csv`.

Execute:

```commandline
ado create space -f ml_multicloud_space.yaml --use-latest samplestore
```

This will confirm the creation of the `discoveryspace` with:

```commandline
Success! Created space with identifier: $DISCOVERY_SPACE_IDENTIFIER
```

You can now describe the `discoveryspace` with:

```commandline
ado describe space --use-latest
```

This will output:

```terminaloutput
Identifier: 'space-ef59e6-2a6318'

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
      replay.benchmark_performance ┃ nan                    ┃ nan

    ───────────────────────────────────── benchmark_performance ─────────────────────────────────────
     Expected Interface

     Inputs:

        parameter  ┃ type     ┃ value ┃ parameterized
       ━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━╋━━━━━━━━━━━━━━━
        cpu_family ┃ required ┃ nan   ┃ na
        nodes      ┃ required ┃ nan   ┃ na
        provider   ┃ required ┃ nan   ┃ na
        vcpu_size  ┃ required ┃ nan   ┃ na

     Outputs:

        target property
       ━━━━━━━━━━━━━━━━━━
        wallClockRuntime
        status

    ─────────────────────────────────────────────────────────────────────────────────────────────────


Sample Store identifier: 2a6318
```

> [!NOTE]
>
> The set of points is defined by the properties in the `Entity Space` - here
> '_cpu_family_', '_provider_', '_vcpu_size_' and '_nodes_' - and the values
> those properties can take.

<!-- markdownlint-disable-next-line no-blanks-blockquote -->

> [!TIP]
>
> Consider why the size of the entityspace is 48. Compare this to the number of
> rows in `ml_export.csv`.

## Exploring the `discoveryspace`

Next we will run an operation that will "explore" the `discoveryspace` we just
created. Since we already have the data, `ado` will transparently identify and
reuse it. An example operation file is given in
`randomwalk_ml_multicloud_operation.yaml`. The contents are:

<!-- prettier-ignore-start -->
<!-- markdownlint-disable MD013 -->

```yaml
{% include "../../../examples/ml-multi-cloud/randomwalk_ml_multicloud_operation.yaml" %}
```

<!-- markdownlint-enable MD013 -->
<!-- prettier-ignore-end -->

To run the operation execute:

```commandline
ado create operation -f randomwalk_ml_multicloud_operation.yaml --use-latest space
```

This will output a lot of information as it samples all the entities. Typically,
you will see the following lines for each entity (point in the entity space)
sampled and measured:

<!-- markdownlint-disable line-length -->

```commandline
(RandomWalk pid=48600) Continuous batching: SUBMIT EXPERIMENT. Submitted experiment
replay.benchmark_performance for A_f1.0-c0.0-n2. Request identifier: replayed-measurement-fa465c
(RandomWalk pid=48600)
(RandomWalk pid=48600) Continuous batching: SUMMARY. Entities sampled and submitted: 2. Experiments
completed: 1 Waiting on 1 active requests. There are 0 dependent experiments
(RandomWalk pid=48600) Continuous Batching: EXPERIMENT COMPLETION. Received finished notification for
experiment in measurement request in group 1:
replayed-measurement-fa465c-experiment-benchmark_performance-entities-A_f1.0-c0.0-n2
(multi-cloud-ml)-time-2026-07-09 10:26:50.745505+01:00
```

<!-- markdownlint-enable line-length -->

The first line, "SUBMIT EXPERIMENT", indicates the entity - `A_f1.0-c0.0-n2` -
and experiment - `replay.benchmark_performance` submitted. The next line gives a
summary of what has happened so far: this is the second entity sampled and
submitted; one experiment has completed; and the sampler is waiting on one
active experiment before submitting a new one. Finally, the "EXPERIMENT
COMPLETION" line indicates the experiment has finished.

The operation will end with information like:

```yaml
=========== Operation Details ============

Space ID: space-ef59e6-2a6318
Sample Store ID:  2a6318
Operation:
 config:
  actuatorConfigurationIdentifiers: []
  metadata:
    description: Perform a random walk on all points in a space
    name: randomwalk-all
  operation:
    module:
      operationType: explore
      operatorName: random_walk
      operatorVersion: 2.0.0
    parameters:
      batchSize: 1
      filter:
        filterMode: noFilter
      maxRetries: 0
      numberEntities: 48
      samplerConfig:
        grouping: []
        mode: random
        samplerType: generator
      singleMeasurement: true
  spaces:
  - space-ef59e6-2a6318
created: '2026-07-09T09:26:50.609258Z'
identifier: random_walk@2.0.0-31d4c6
kind: operation
metadata:
  entities_submitted: 48
  experiments_requested: 74
operationType: explore
operatorIdentifier: random_walk@2.0.0
provenance:
  ado:
    distributionName: ado-core
    distributionVersion: 2.0.0
  operators:
    random_walk@2.0.0:
      distributionName: ado-core
      distributionVersion: 2.0.0
status:
- event: created
  recorded_at: '2026-07-09T09:26:50.609263Z'
- event: added
  recorded_at: '2026-07-09T09:26:50.609953Z'
- event: started
  recorded_at: '2026-07-09T09:26:50.612872Z'
- event: updated
  recorded_at: '2026-07-09T09:26:50.612883Z'
- event: finished
  exit_state: success
  recorded_at: '2026-07-09T09:26:52.072109Z'
- event: updated
  recorded_at: '2026-07-09T09:26:52.075540Z'
version: v1
```

The operation identifier is stored in the `identifier` field: in the output
above, it is `random_walk@2.0.0-31d4c6`.

> [!NOTE]
>
> The operation "reuses" existing measurements: this is an `ado` feature called
> **memoization**.
>
> `ado` transparently executes experiments or memoizes data as appropriate - so
> the operator does not need to know if a measurement needs to be performed at
> the time it requests it, or if previous data can be reused.

<!-- markdownlint-disable-next-line no-blanks-blockquote -->

> [!TIP]
>
> Operations are **domain agnostic**. If you look in
> `randomwalk_ml_multicloud_operation.yaml` you will see there is no reference
> to characteristics of the discoveryspace we created. Indeed, this operation
> file could work on any discoveryspace.
>
> This shows that operators, like randomwalk, don't have to know domain specific
> details. All information about what to explore and how to measure is captured
> in the `discoveryspace`.

## Looking at the `operation` output

The command

```commandline
ado show measurements operation --use-latest
```

displays the results of the operation i.e. the entities sampled and the
measurement results. You will see something like the following (the sampling is
random so the order can be different):

<!-- markdownlint-disable line-length -->

```text
┌───────────────┬──────────────┬─────────────────────────────────────────────┬──────────────────────────────┬────────────┬────────────────────────────────┬───────┬──────────┬───────────┬────────────────────┬──────────────┬──────────────────────────────────────────────────────────────────────────────────────────────┬─────────────────────────────────┬──────────────┬───────┐
│ request_index │ result_index │ identifier                                  │ experiment_id                │ cpu_family │ generatorid                    │ nodes │ provider │ vcpu_size │ wallClockRuntime   │ status       │ reason                                                                                       │ request_id                      │ entity_index │ valid │
├───────────────┼──────────────┼─────────────────────────────────────────────┼──────────────────────────────┼────────────┼────────────────────────────────┼───────┼──────────┼───────────┼────────────────────┼──────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────┼──────────────┼───────┤
│ 0             │ 0            │ A_f1.0-c0.0-n4                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 4     │ A        │ 0.0       │ 158.70639538764954 │ ok           │ not_measured                                                                                 │ replayed-measurement-d27306     │ 0            │ True  │
│ 1             │ 0            │ A_f1.0-c0.0-n2                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 2     │ A        │ 0.0       │ 378.31657004356384 │ ok           │ not_measured                                                                                 │ replayed-measurement-fa465c     │ 0            │ True  │
│ 2             │ 0            │ B_f0.0-c0.0-n3                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 3     │ B        │ 0.0       │ 153.51639366149902 │ ok           │ not_measured                                                                                 │ replayed-measurement-9a5539     │ 0            │ True  │
│ 10            │ 0            │ cpu_family.1-nodes.3-provider.B-vcpu_size.1 │ replay.benchmark_performance │ 1.0        │ explicit_grid_sample_generator │ 3     │ B        │ 1.0       │ not_measured       │ not_measured │ Externally defined experiments cannot be applied to entities: replay.benchmark_performance.  │ random_walk@2.0.0-31d4c6-f576d4 │ 0            │ False │
└───────────────┴──────────────┴─────────────────────────────────────────────┴──────────────────────────────┴────────────┴────────────────────────────────┴───────┴──────────┴───────────┴────────────────────┴──────────────┴──────────────────────────────────────────────────────────────────────────────────────────────┴─────────────────────────────────┴──────────────┴───────┘
```

<!-- markdownlint-enable line-length -->

> [!TIP] Some things to note and consider:
>
> - The table is in the order the points were measured.
> - Some points have multiple measurements — compare the entityspace size (48)
>   to the number of rows in `ml_export.csv`.
> - Some points were not measured (`valid: False`). These are points in the
>   discoveryspace for which no matching data was present in `ml_export.csv` to
>   replay.
> - The `reason` column shows `not_measured` even for successful results
>   (`status: ok`). This means the measurement was _replayed_ from existing data
>   rather than executed live; it is not an error.

## Exploring Further

Here are a variety of commands you can try after executing the example above:

### Viewing entities

There are multiple ways to view the entities related to a `discoveryspace`. Try:

```commandline
ado show measurements space --use-latest
ado show measurements space --use-latest --aggregate mean
ado show measurements space --use-latest --include unmeasured
ado show measurements space --use-latest --property-format target
```

Also, the following command will give you summary statistics of what has been
measured:

```commandline
ado show stats discoveryspace --use-latest
```

> [!NOTE]
>
> If you want to run these commands against the most recent space in the current
> context, use the `--use-latest` flag as above.

### Resource provenance

The `related` sub-command shows resource provenance:

```commandline
ado show related operation --use-latest
```

### Operation timeseries

The following commands give more details of the operation timeseries:

```commandline
ado show trace operation --use-latest --unroll-entities
ado show trace operation --use-latest
```

### Resource templates

Another helpful command is `template` which will output a default example of a
resource YAML along with an (optional) description of its fields. Try:

<!-- markdownlint-disable line-length -->

```commandline
ado template operation --include-schema --operator-name random_walk --output-file random_walk_template.yaml
```

<!-- markdownlint-enable line-length -->

### Rerun

An interesting thing to try is to run the operation again and compare the output
of `ado show measurements operation` for the two operations, and
`ado show measurements space`.

## Takeaways

- **create-explore-view pattern**: A common pattern in `ado` is to create a
  `discoveryspace` to describe a set of points to measure, create `operations`
  on it to explore or analyse it, and then view the results.
- **entity space and measurement space**: A `discoveryspace` consists of an
  `entityspace` - the set of points to measure - and a `measurementspace` - the
  set of experiments to apply to them.
- **operations are domain agnostic**: `ado` enables operations to run on
  multiple different domains without modification.
- **memoization**: By default `ado` will identify if a measurement has already
  been completed on an entity and reuse it.
- **provenance**: `ado` stores the relationship between the resources it
  creates.
- **results viewing**: `ado show measurements` outputs the data in a
  `discoveryspace` or measured in an `operation`.
- **measurement timeseries**: The sequence (timeseries) of measurements,
  successful or not, of each `operation` is preserved.
- **`discoveryspace` views**: By default `ado show measurements space` only
  shows successfully measured entities, but you can see what has not been
  measured if you want.

## What's next

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable no-inline-html -->
<!-- markdownlint-disable MD046 -->
<!-- prettier-ignore-start -->

<div class="grid cards" markdown>

- :octicons-rocket-24:{ .lg .middle } **Search using an optimizer**

    ---

    Try the [Search a space with an optimizer](best-configuration-search.md) example to see how you can use RayTune, and define custom experiments, via `ado`.

    [Search a space with an optimizer :octicons-arrow-right-24:](best-configuration-search.md)

- :octicons-workflow-24:{ .lg .middle } **Discovering important entity space dimensions**

      ---

      Try the [Identify the important dimensions of a space](lhu.md) example to see how you can use `ado` to discover which entity space dimensions most influence a target metric.

      [Identify the important dimensions of a space :octicons-arrow-right-24:](lhu.md)

</div>

<!-- prettier-ignore-end -->
<!-- markdownlint-enable MD046 -->
<!-- markdownlint-enable no-inline-html -->
<!-- markdownlint-enable line-length -->
