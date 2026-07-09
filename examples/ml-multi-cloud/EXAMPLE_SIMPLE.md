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
> - Get the example files
>
> ```commandline
> git clone https://github.com/IBM/ado.git
> cd ado/examples/ml-multi-cloud
> ```
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

```yaml
{% include "./randomwalk_ml_multicloud_operation.yaml" %}
```

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
│ 2             │ 0            │ B_f0.0-c0.0-n3                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 3     │ B        │ 0.0       │ 184.44801592826843 │ ok           │ not_measured                                                                                 │ replayed-measurement-a9c3bd     │ 0            │ True  │
│ 2             │ 0            │ B_f0.0-c0.0-n3                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 3     │ B        │ 0.0       │ 176.28814435005188 │ ok           │ not_measured                                                                                 │ replayed-measurement-677a17     │ 0            │ True  │
│ 3             │ 0            │ C_f1.0-c1.0-n2                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 2     │ C        │ 1.0       │ 363.2856709957123  │ ok           │ not_measured                                                                                 │ replayed-measurement-635508     │ 0            │ True  │
│ 4             │ 0            │ A_f1.0-c1.0-n3                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 3     │ A        │ 1.0       │ 151.58562421798706 │ ok           │ not_measured                                                                                 │ replayed-measurement-1dbbb4     │ 0            │ True  │
│ 4             │ 0            │ A_f1.0-c1.0-n3                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 3     │ A        │ 1.0       │ 155.02856159210205 │ ok           │ not_measured                                                                                 │ replayed-measurement-de0a9f     │ 0            │ True  │
│ 5             │ 0            │ C_f1.0-c1.0-n5                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 5     │ C        │ 1.0       │ 92.17141437530518  │ ok           │ not_measured                                                                                 │ replayed-measurement-e1b41d     │ 0            │ True  │
│ 5             │ 0            │ C_f1.0-c1.0-n5                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 5     │ C        │ 1.0       │ 100.97977471351624 │ ok           │ not_measured                                                                                 │ replayed-measurement-e1cdab     │ 0            │ True  │
│ 6             │ 0            │ A_f1.0-c1.0-n2                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 2     │ A        │ 1.0       │ 291.90445613861084 │ ok           │ not_measured                                                                                 │ replayed-measurement-73f29c     │ 0            │ True  │
│ 7             │ 0            │ A_f1.0-c0.0-n5                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 5     │ A        │ 0.0       │ 135.91092538833618 │ ok           │ not_measured                                                                                 │ replayed-measurement-c48361     │ 0            │ True  │
│ 7             │ 0            │ A_f1.0-c0.0-n5                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 5     │ A        │ 0.0       │ 117.94136571884157 │ ok           │ not_measured                                                                                 │ replayed-measurement-40a69e     │ 0            │ True  │
│ 8             │ 0            │ C_f0.0-c1.0-n4                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 4     │ C        │ 1.0       │ 121.42492485046388 │ ok           │ not_measured                                                                                 │ replayed-measurement-b59067     │ 0            │ True  │
│ 9             │ 0            │ C_f0.0-c1.0-n3                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 3     │ C        │ 1.0       │ 168.9163637161255  │ ok           │ not_measured                                                                                 │ replayed-measurement-f30779     │ 0            │ True  │
│ 9             │ 0            │ C_f0.0-c1.0-n3                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 3     │ C        │ 1.0       │ 174.0335624217987  │ ok           │ not_measured                                                                                 │ replayed-measurement-733703     │ 0            │ True  │
│ 10            │ 0            │ cpu_family.1-nodes.3-provider.B-vcpu_size.1 │ replay.benchmark_performance │ 1.0        │ explicit_grid_sample_generator │ 3     │ B        │ 1.0       │ not_measured       │ not_measured │ Externally defined experiments cannot be applied to entities: replay.benchmark_performance.  │ random_walk@2.0.0-31d4c6-f576d4 │ 0            │ False │
│ 11            │ 0            │ C_f0.0-c1.0-n2                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 2     │ C        │ 1.0       │ 309.8423240184784  │ ok           │ not_measured                                                                                 │ replayed-measurement-ea3637     │ 0            │ True  │
│ 12            │ 0            │ A_f1.0-c0.0-n3                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 3     │ A        │ 0.0       │ 206.74496150016785 │ ok           │ not_measured                                                                                 │ replayed-measurement-f2c3d8     │ 0            │ True  │
│ 12            │ 0            │ A_f1.0-c0.0-n3                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 3     │ A        │ 0.0       │ 236.1715066432953  │ ok           │ not_measured                                                                                 │ replayed-measurement-56beef     │ 0            │ True  │
│ 13            │ 0            │ B_f1.0-c0.0-n3                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 3     │ B        │ 0.0       │ 220.19828414916992 │ ok           │ not_measured                                                                                 │ replayed-measurement-d936ad     │ 0            │ True  │
│ 13            │ 0            │ B_f1.0-c0.0-n3                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 3     │ B        │ 0.0       │ 273.7120273113251  │ ok           │ not_measured                                                                                 │ replayed-measurement-c2e099     │ 0            │ True  │
│ 14            │ 0            │ cpu_family.0-nodes.3-provider.B-vcpu_size.1 │ replay.benchmark_performance │ 0.0        │ explicit_grid_sample_generator │ 3     │ B        │ 1.0       │ not_measured       │ not_measured │ Externally defined experiments cannot be applied to entities: replay.benchmark_performance.  │ random_walk@2.0.0-31d4c6-b37048 │ 0            │ False │
│ 15            │ 0            │ cpu_family.0-nodes.5-provider.B-vcpu_size.1 │ replay.benchmark_performance │ 0.0        │ explicit_grid_sample_generator │ 5     │ B        │ 1.0       │ not_measured       │ not_measured │ Externally defined experiments cannot be applied to entities: replay.benchmark_performance.  │ random_walk@2.0.0-31d4c6-25924a │ 0            │ False │
│ 16            │ 0            │ C_f1.0-c0.0-n5                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 5     │ C        │ 0.0       │ 136.3071050643921  │ ok           │ not_measured                                                                                 │ replayed-measurement-708e5a     │ 0            │ True  │
│ 16            │ 0            │ C_f1.0-c0.0-n5                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 5     │ C        │ 0.0       │ 135.47050046920776 │ ok           │ not_measured                                                                                 │ replayed-measurement-41dfd1     │ 0            │ True  │
│ 17            │ 0            │ C_f0.0-c0.0-n2                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 2     │ C        │ 0.0       │ 415.8292849063873  │ ok           │ not_measured                                                                                 │ replayed-measurement-e4b88a     │ 0            │ True  │
│ 18            │ 0            │ C_f1.0-c0.0-n2                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 2     │ C        │ 0.0       │ 463.396538734436   │ ok           │ not_measured                                                                                 │ replayed-measurement-8cde17     │ 0            │ True  │
│ 19            │ 0            │ A_f0.0-c1.0-n4                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 4     │ A        │ 1.0       │ 106.67012143135072 │ ok           │ not_measured                                                                                 │ replayed-measurement-6e6eba     │ 0            │ True  │
│ 20            │ 0            │ cpu_family.0-nodes.4-provider.B-vcpu_size.1 │ replay.benchmark_performance │ 0.0        │ explicit_grid_sample_generator │ 4     │ B        │ 1.0       │ not_measured       │ not_measured │ Externally defined experiments cannot be applied to entities: replay.benchmark_performance.  │ random_walk@2.0.0-31d4c6-305e2c │ 0            │ False │
│ 21            │ 0            │ A_f0.0-c1.0-n2                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 2     │ A        │ 1.0       │ 272.99782156944275 │ ok           │ not_measured                                                                                 │ replayed-measurement-f29362     │ 0            │ True  │
│ 22            │ 0            │ C_f0.0-c1.0-n5                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 5     │ C        │ 1.0       │ 85.67946743965149  │ ok           │ not_measured                                                                                 │ replayed-measurement-d9a69a     │ 0            │ True  │
│ 22            │ 0            │ C_f0.0-c1.0-n5                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 5     │ C        │ 1.0       │ 95.86326050758362  │ ok           │ not_measured                                                                                 │ replayed-measurement-5dd4c0     │ 0            │ True  │
│ 23            │ 0            │ A_f0.0-c1.0-n5                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 5     │ A        │ 1.0       │ 84.45346999168396  │ ok           │ not_measured                                                                                 │ replayed-measurement-40ad51     │ 0            │ True  │
│ 23            │ 0            │ A_f0.0-c1.0-n5                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 5     │ A        │ 1.0       │ 86.23016095161438  │ ok           │ not_measured                                                                                 │ replayed-measurement-40d15b     │ 0            │ True  │
│ 24            │ 0            │ A_f0.0-c0.0-n4                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 4     │ A        │ 0.0       │ 145.12948369979858 │ ok           │ not_measured                                                                                 │ replayed-measurement-345fad     │ 0            │ True  │
│ 25            │ 0            │ cpu_family.1-nodes.5-provider.B-vcpu_size.1 │ replay.benchmark_performance │ 1.0        │ explicit_grid_sample_generator │ 5     │ B        │ 1.0       │ not_measured       │ not_measured │ Externally defined experiments cannot be applied to entities: replay.benchmark_performance.  │ random_walk@2.0.0-31d4c6-c544ab │ 0            │ False │
│ 26            │ 0            │ B_f0.0-c0.0-n4                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 4     │ B        │ 0.0       │ 113.87676978111269 │ ok           │ not_measured                                                                                 │ replayed-measurement-7edf20     │ 0            │ True  │
│ 26            │ 0            │ B_f0.0-c0.0-n4                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 4     │ B        │ 0.0       │ 132.5415120124817  │ ok           │ not_measured                                                                                 │ replayed-measurement-a5b1ad     │ 0            │ True  │
│ 27            │ 0            │ B_f0.0-c1.0-n2                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 2     │ B        │ 1.0       │ 184.935049533844   │ ok           │ not_measured                                                                                 │ replayed-measurement-da472b     │ 0            │ True  │
│ 27            │ 0            │ B_f0.0-c1.0-n2                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 2     │ B        │ 1.0       │ 166.74843192100525 │ ok           │ not_measured                                                                                 │ replayed-measurement-c4bf32     │ 0            │ True  │
│ 28            │ 0            │ A_f0.0-c0.0-n2                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 2     │ A        │ 0.0       │ 335.2085180282593  │ ok           │ not_measured                                                                                 │ replayed-measurement-678aae     │ 0            │ True  │
│ 29            │ 0            │ A_f0.0-c1.0-n3                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 3     │ A        │ 1.0       │ 168.36590766906738 │ ok           │ not_measured                                                                                 │ replayed-measurement-cd1f84     │ 0            │ True  │
│ 29            │ 0            │ A_f0.0-c1.0-n3                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 3     │ A        │ 1.0       │ 170.15659737586975 │ ok           │ not_measured                                                                                 │ replayed-measurement-dd0f9a     │ 0            │ True  │
│ 30            │ 0            │ B_f0.0-c0.0-n5                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 5     │ B        │ 0.0       │ 103.90595746040344 │ ok           │ not_measured                                                                                 │ replayed-measurement-73cea7     │ 0            │ True  │
│ 30            │ 0            │ B_f0.0-c0.0-n5                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 5     │ B        │ 0.0       │ 112.7056987285614  │ ok           │ not_measured                                                                                 │ replayed-measurement-97139f     │ 0            │ True  │
│ 30            │ 0            │ B_f0.0-c0.0-n5                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 5     │ B        │ 0.0       │ 113.88505148887634 │ ok           │ not_measured                                                                                 │ replayed-measurement-f76ee3     │ 0            │ True  │
│ 31            │ 0            │ B_f1.0-c1.0-n2                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 2     │ B        │ 1.0       │ 298.8193049430847  │ ok           │ not_measured                                                                                 │ replayed-measurement-7aee8c     │ 0            │ True  │
│ 32            │ 0            │ C_f1.0-c0.0-n3                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 3     │ C        │ 0.0       │ 598.8834657669067  │ Timed out.   │ not_measured                                                                                 │ replayed-measurement-ff4df6     │ 0            │ True  │
│ 32            │ 0            │ C_f1.0-c0.0-n3                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 3     │ C        │ 0.0       │ 244.33887457847595 │ ok           │ not_measured                                                                                 │ replayed-measurement-02e84b     │ 0            │ True  │
│ 33            │ 0            │ B_f1.0-c0.0-n2                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 2     │ B        │ 0.0       │ 346.0709958076477  │ ok           │ not_measured                                                                                 │ replayed-measurement-dc5f7f     │ 0            │ True  │
│ 34            │ 0            │ C_f1.0-c0.0-n4                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 4     │ C        │ 0.0       │ 177.72359776496887 │ ok           │ not_measured                                                                                 │ replayed-measurement-88c126     │ 0            │ True  │
│ 35            │ 0            │ C_f0.0-c0.0-n4                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 4     │ C        │ 0.0       │ 188.09087824821472 │ ok           │ not_measured                                                                                 │ replayed-measurement-778830     │ 0            │ True  │
│ 36            │ 0            │ A_f0.0-c0.0-n5                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 5     │ A        │ 0.0       │ 106.0709307193756  │ ok           │ not_measured                                                                                 │ replayed-measurement-98cdbd     │ 0            │ True  │
│ 36            │ 0            │ A_f0.0-c0.0-n5                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 5     │ A        │ 0.0       │ 130.30512285232544 │ ok           │ not_measured                                                                                 │ replayed-measurement-938232     │ 0            │ True  │
│ 37            │ 0            │ cpu_family.1-nodes.4-provider.B-vcpu_size.1 │ replay.benchmark_performance │ 1.0        │ explicit_grid_sample_generator │ 4     │ B        │ 1.0       │ not_measured       │ not_measured │ Externally defined experiments cannot be applied to entities: replay.benchmark_performance.  │ random_walk@2.0.0-31d4c6-3be514 │ 0            │ False │
│ 38            │ 0            │ A_f1.0-c1.0-n5                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 5     │ A        │ 1.0       │ 105.63729166984558 │ ok           │ not_measured                                                                                 │ replayed-measurement-f7f930     │ 0            │ True  │
│ 38            │ 0            │ A_f1.0-c1.0-n5                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 5     │ A        │ 1.0       │ 96.8471610546112   │ ok           │ not_measured                                                                                 │ replayed-measurement-ce7af9     │ 0            │ True  │
│ 39            │ 0            │ B_f1.0-c0.0-n4                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 4     │ B        │ 0.0       │ 202.48239731788635 │ ok           │ not_measured                                                                                 │ replayed-measurement-ce9983     │ 0            │ True  │
│ 39            │ 0            │ B_f1.0-c0.0-n4                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 4     │ B        │ 0.0       │ 193.55997109413147 │ ok           │ not_measured                                                                                 │ replayed-measurement-744787     │ 0            │ True  │
│ 40            │ 0            │ A_f1.0-c1.0-n4                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 4     │ A        │ 1.0       │ 116.31417059898376 │ ok           │ not_measured                                                                                 │ replayed-measurement-a1baf4     │ 0            │ True  │
│ 41            │ 0            │ C_f0.0-c0.0-n3                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 3     │ C        │ 0.0       │ 240.07358503341675 │ ok           │ not_measured                                                                                 │ replayed-measurement-75c0d5     │ 0            │ True  │
│ 41            │ 0            │ C_f0.0-c0.0-n3                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 3     │ C        │ 0.0       │ 269.0906641483307  │ ok           │ not_measured                                                                                 │ replayed-measurement-30997e     │ 0            │ True  │
│ 42            │ 0            │ B_f0.0-c0.0-n2                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 2     │ B        │ 0.0       │ 225.1791422367096  │ ok           │ not_measured                                                                                 │ replayed-measurement-338bed     │ 0            │ True  │
│ 42            │ 0            │ B_f0.0-c0.0-n2                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 2     │ B        │ 0.0       │ 228.14362454414368 │ ok           │ not_measured                                                                                 │ replayed-measurement-afbdd0     │ 0            │ True  │
│ 43            │ 0            │ C_f1.0-c1.0-n3                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 3     │ C        │ 1.0       │ 154.9813470840454  │ ok           │ not_measured                                                                                 │ replayed-measurement-2f17c9     │ 0            │ True  │
│ 43            │ 0            │ C_f1.0-c1.0-n3                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 3     │ C        │ 1.0       │ 168.34859228134155 │ ok           │ not_measured                                                                                 │ replayed-measurement-d3d572     │ 0            │ True  │
│ 44            │ 0            │ C_f0.0-c0.0-n5                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 5     │ C        │ 0.0       │ 138.0605161190033  │ ok           │ not_measured                                                                                 │ replayed-measurement-b94055     │ 0            │ True  │
│ 44            │ 0            │ C_f0.0-c0.0-n5                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 5     │ C        │ 0.0       │ 150.9471504688263  │ ok           │ not_measured                                                                                 │ replayed-measurement-60d9c7     │ 0            │ True  │
│ 45            │ 0            │ A_f0.0-c0.0-n3                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 3     │ A        │ 0.0       │ 221.5101969242096  │ ok           │ not_measured                                                                                 │ replayed-measurement-c12d86     │ 0            │ True  │
│ 45            │ 0            │ A_f0.0-c0.0-n3                              │ replay.benchmark_performance │ 0.0        │ multi-cloud-ml                 │ 3     │ A        │ 0.0       │ 216.394127368927   │ ok           │ not_measured                                                                                 │ replayed-measurement-62e712     │ 0            │ True  │
│ 46            │ 0            │ C_f1.0-c1.0-n4                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 4     │ C        │ 1.0       │ 114.01436853408812 │ ok           │ not_measured                                                                                 │ replayed-measurement-215e79     │ 0            │ True  │
│ 47            │ 0            │ B_f1.0-c0.0-n5                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 5     │ B        │ 0.0       │ 141.99024295806885 │ ok           │ not_measured                                                                                 │ replayed-measurement-012dd9     │ 0            │ True  │
│ 47            │ 0            │ B_f1.0-c0.0-n5                              │ replay.benchmark_performance │ 1.0        │ multi-cloud-ml                 │ 5     │ B        │ 0.0       │ 168.79178500175476 │ ok           │ not_measured                                                                                 │ replayed-measurement-09afd9     │ 0            │ True  │
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
