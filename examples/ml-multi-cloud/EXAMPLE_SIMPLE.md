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
> : # Create the space to explore
> ado create space -f ml_multicloud_space.yaml --with store=ml_multicloud_sample_store.yaml
> : # Explore!
> ado create op -f randomwalk_ml_multicloud_operation.yaml --use-latest space
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
Identifier: 'space-19b2de-6da1f4'

Entity Space:

   Number of entities: 48

   Categorical properties:

      name       values
     ────────────────────────────
      provider   ['A', 'B', 'C']

   Discrete properties:

      name         range   interval   values
     ──────────────────────────────────────────────
      cpu_family   None    None       [0, 1]
      vcpu_size    None    None       [0, 1]
      nodes        None    None       [2, 3, 4, 5]


Measurement Space:

   Experiments:

      experiment                                   supported
     ────────────────────────────────────────────────────────
      replay.benchmark_performance                 True
      custom_experiments.ml-multicloud-cost-v1.0   True

    ─────────────────── replay.benchmark_performance ────────────────────
     Inputs:

        parameter    type       value   parameterized
       ───────────────────────────────────────────────
        cpu_family   required   None    na
        nodes        required   None    na
        provider     required   None    na
        vcpu_size    required   None    na

     Outputs:

        target property
       ──────────────────
        wallClockRuntime
        status

    ─────────────────────────────────────────────────────────────────────

    ──────────── custom_experiments.ml-multicloud-cost-v1.0 ─────────────
     Inputs:

        parameter                    type       value   parameterized
       ───────────────────────────────────────────────────────────────
        nodes                        required   None    na
        cpu_family                   required   None    na
        benchmark_performance-wal…   required   None    na

     Outputs:

        target property
       ─────────────────
        total_cost

    ─────────────────────────────────────────────────────────────────────


Sample Store identifier: 6da1f4
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
(RandomWalk pid=14797) Continuous batching: SUBMIT EXPERIMENT. Submitting experiment replay.benchmark_performance for provider.B-cpu_family.1-vcpu_size.1-nodes.4
(RandomWalk pid=14797)
(RandomWalk pid=14797) Continuous batching: SUMMARY. Entities sampled and submitted: 2. Experiments completed: 1 Waiting on 1 active requests. There are 0 dependent experiments
(RandomWalk pid=14797) Continuous Batching: EXPERIMENT COMPLETION. Received finished notification for experiment in measurement request in group 1: request-randomwalk-0.9.6.dev91+884f713b.dirty-c5ed4b-579021-experiment-benchmark_performance-entities-provider.B-cpu_family.1-vcpu_size.1-nodes.4 (explicit_grid_sample_generator)-requester-randomwalk-0.9.6.dev91+884f713b.dirty-c5ed4b-time-2025-07-29 20:03:00.976809+01:00
```

<!-- markdownlint-enable line-length -->

The first line, "SUBMIT EXPERIMENT", indicates the entity -
`provider.B-cpu_family.1-vcpu_size.1-nodes.4` - and experiment -
`replay.benchmark_performance` submitted. The next line gives a summary of what
has happened so far: this is the second entity sampled and submitted; one
experiment has completed; and the sampler is waiting on one active experiment
before submitting a new one. Finally, the "EXPERIMENT COMPLETION" line indicates
the experiment has finished.

The operation will end with information like:

```yaml
config:
  operation:
    module:
      moduleClass: RandomWalk
      moduleName: orchestrator.modules.operators.randomwalk
      modulePath: .
      moduleType: operation
    parameters:
      batchSize: 1
      numberEntities: 48
      samplerConfig:
        mode: sequential
        samplerType: generator
  spaces:
    - space-65cf33-a8df39
created: "2025-06-20T13:03:46.763154Z"
identifier: randomwalk-0.9.4.dev30+564196d4.dirty-b8a233
kind: operation
metadata:
  entities_submitted: 48
  experiments_requested: 74
operationType: search
operatorIdentifier: randomwalk-0.9.4.dev30+564196d4.dirty
status:
  - event: created
    recorded_at: "2025-06-20T13:03:40.267005Z"
  - event: added
    recorded_at: "2025-06-20T13:03:46.764750Z"
  - event: started
    recorded_at: "2025-06-20T13:03:46.769169Z"
  - event: finished
    exit_state: success
    recorded_at: "2025-06-20T13:03:48.369516Z"
  - event: updated
    recorded_at: "2025-06-20T13:03:48.374765Z"
version: v1
```

The identifier operation is stored in the `identifier` field: in the output
above, it is `randomwalk-0.9.4.dev30+564196d4.dirty-b8a233`.

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
ado show entities operation --use-latest
```

displays the results of the operation i.e. the entities sampled and the
measurement results. You will see something like the following (the sampling is
random so the order can be different):

<!-- markdownlint-disable line-length -->

```text
┌───────────────┬──────────────┬───────────────┬───────────────┬────────────┬───────────────┬───────┬──────────┬───────────┬────────────────┬───────────────┬──────────────┬────────────────┬───────────────┬──────────────┬───────┐
│ request_index │ result_index │ identifier    │ experiment_id │ cpu_family │ generatorid   │ nodes │ provider │ vcpu_size │ reason         │ wallClockRun… │ status       │ total_cost     │ request_id    │ entity_index │ valid │
├───────────────┼──────────────┼───────────────┼───────────────┼────────────┼───────────────┼───────┼──────────┼───────────┼────────────────┼───────────────┼──────────────┼────────────────┼───────────────┼──────────────┼───────┤
│ 0             │ 0            │ provider.B-c… │ replay.bench… │ 0.0        │ explicit_gri… │ 3     │ B        │ 1.0       │ Externally     │ not_measured  │ not_measured │ not_measured   │ randomwalk-1… │ 0            │ False │
│               │              │               │               │            │               │       │          │           │ defined        │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ experiments    │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ cannot be      │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ applied to     │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ entities:      │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ replay.benchm… │               │              │                │               │              │       │
│ 1             │ 0            │ C_f1.0-c1.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 5     │ C        │ 1.0       │ not_measured   │ 92.171414375… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 1             │ 0            │ C_f1.0-c1.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 5     │ C        │ 1.0       │ not_measured   │ 100.97977471… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 1             │ 0            │ C_f1.0-c1.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 5     │ C        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 2.56031706598… │ 9d1c78        │ 0            │ True  │
│ 1             │ 0            │ C_f1.0-c1.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 5     │ C        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 2.80499374204… │ 2b5c63        │ 0            │ True  │
│ 2             │ 0            │ provider.B-c… │ replay.bench… │ 0.0        │ explicit_gri… │ 5     │ B        │ 1.0       │ Externally     │ not_measured  │ not_measured │ not_measured   │ randomwalk-1… │ 0            │ False │
│               │              │               │               │            │               │       │          │           │ defined        │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ experiments    │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ cannot be      │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ applied to     │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ entities:      │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ replay.benchm… │               │              │                │               │              │       │
│ 3             │ 0            │ C_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 5     │ C        │ 0.0       │ not_measured   │ 136.30710506… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 3             │ 0            │ C_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 5     │ C        │ 0.0       │ not_measured   │ 135.47050046… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 3             │ 0            │ C_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 5     │ C        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 3.78630847401… │ ce622d        │ 0            │ True  │
│ 3             │ 0            │ C_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 5     │ C        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 3.76306945747… │ df682d        │ 0            │ True  │
│ 4             │ 0            │ B_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 5     │ B        │ 0.0       │ not_measured   │ 103.90595746… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 4             │ 0            │ B_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 5     │ B        │ 0.0       │ not_measured   │ 112.70569872… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 4             │ 0            │ B_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 5     │ B        │ 0.0       │ not_measured   │ 113.88505148… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 4             │ 0            │ B_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 5     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.44313829806… │ 510954        │ 0            │ True  │
│ 4             │ 0            │ B_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 5     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.56535692678… │ adc911        │ 0            │ True  │
│ 4             │ 0            │ B_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 5     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.58173682623… │ 7dc57c        │ 0            │ True  │
│ 5             │ 0            │ A_f1.0-c1.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 5     │ A        │ 1.0       │ not_measured   │ 105.63729166… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 5             │ 0            │ A_f1.0-c1.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 5     │ A        │ 1.0       │ not_measured   │ 96.847161054… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 5             │ 0            │ A_f1.0-c1.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 5     │ A        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 2.93436921305… │ f422f1        │ 0            │ True  │
│ 5             │ 0            │ A_f1.0-c1.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 5     │ A        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 2.69019891818… │ ad4f2e        │ 0            │ True  │
│ 6             │ 0            │ B_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 2     │ B        │ 0.0       │ not_measured   │ 346.07099580… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 6             │ 0            │ B_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 2     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 3.84523328675… │ 884535        │ 0            │ True  │
│ 7             │ 0            │ provider.B-c… │ replay.bench… │ 1.0        │ explicit_gri… │ 4     │ B        │ 1.0       │ Externally     │ not_measured  │ not_measured │ not_measured   │ randomwalk-1… │ 0            │ False │
│               │              │               │               │            │               │       │          │           │ defined        │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ experiments    │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ cannot be      │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ applied to     │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ entities:      │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ replay.benchm… │               │              │                │               │              │       │
│ 8             │ 0            │ C_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 2     │ C        │ 1.0       │ not_measured   │ 309.84232401… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 8             │ 0            │ C_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 2     │ C        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 1.72134624454… │ 0add34        │ 0            │ True  │
│ 9             │ 0            │ C_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 5     │ C        │ 0.0       │ not_measured   │ 138.06051611… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 9             │ 0            │ C_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 5     │ C        │ 0.0       │ not_measured   │ 150.94715046… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 9             │ 0            │ C_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 5     │ C        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.91750716831… │ b22cce        │ 0            │ True  │
│ 9             │ 0            │ C_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 5     │ C        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 2.09648820095… │ 6af57d        │ 0            │ True  │
│ 10            │ 0            │ B_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 4     │ B        │ 0.0       │ not_measured   │ 202.48239731… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 10            │ 0            │ B_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 4     │ B        │ 0.0       │ not_measured   │ 193.55997109… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 10            │ 0            │ B_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 4     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 4.49960882928… │ 641651        │ 0            │ True  │
│ 10            │ 0            │ B_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 4     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 4.30133269098… │ 93c307        │ 0            │ True  │
│ 11            │ 0            │ C_f1.0-c1.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 2     │ C        │ 1.0       │ not_measured   │ 363.28567099… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 11            │ 0            │ C_f1.0-c1.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 2     │ C        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 4.03650745550… │ b2acce        │ 0            │ True  │
│ 12            │ 0            │ C_f1.0-c1.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 4     │ C        │ 1.0       │ not_measured   │ 114.01436853… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 12            │ 0            │ C_f1.0-c1.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 4     │ C        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 2.53365263409… │ 6a3f5f        │ 0            │ True  │
│ 13            │ 0            │ A_f1.0-c1.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 3     │ A        │ 1.0       │ not_measured   │ 151.58562421… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 13            │ 0            │ A_f1.0-c1.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 3     │ A        │ 1.0       │ not_measured   │ 155.02856159… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 13            │ 0            │ A_f1.0-c1.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 3     │ A        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 2.52642707029… │ 4454cf        │ 0            │ True  │
│ 13            │ 0            │ A_f1.0-c1.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 3     │ A        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 2.58380935986… │ dd3bb4        │ 0            │ True  │
│ 14            │ 0            │ A_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 2     │ A        │ 0.0       │ not_measured   │ 335.20851802… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 14            │ 0            │ A_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 2     │ A        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.86226954460… │ e2fc0a        │ 0            │ True  │
│ 15            │ 0            │ provider.B-c… │ replay.bench… │ 1.0        │ explicit_gri… │ 3     │ B        │ 1.0       │ Externally     │ not_measured  │ not_measured │ not_measured   │ randomwalk-1… │ 0            │ False │
│               │              │               │               │            │               │       │          │           │ defined        │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ experiments    │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ cannot be      │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ applied to     │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ entities:      │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ replay.benchm… │               │              │                │               │              │       │
│ 16            │ 0            │ A_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 3     │ A        │ 0.0       │ not_measured   │ 206.74496150… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 16            │ 0            │ A_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 3     │ A        │ 0.0       │ not_measured   │ 236.17150664… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 16            │ 0            │ A_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 3     │ A        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 3.44574935833… │ 44cb62        │ 0            │ True  │
│ 16            │ 0            │ A_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 3     │ A        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 3.93619177738… │ 490165        │ 0            │ True  │
│ 17            │ 0            │ A_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 3     │ A        │ 0.0       │ not_measured   │ 221.51019692… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 17            │ 0            │ A_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 3     │ A        │ 0.0       │ not_measured   │ 216.39412736… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 17            │ 0            │ A_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 3     │ A        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.84591830770… │ bd3d9c        │ 0            │ True  │
│ 17            │ 0            │ A_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 3     │ A        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.80328439474… │ 12ea47        │ 0            │ True  │
│ 18            │ 0            │ A_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 5     │ A        │ 0.0       │ not_measured   │ 135.91092538… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 18            │ 0            │ A_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 5     │ A        │ 0.0       │ not_measured   │ 117.94136571… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 18            │ 0            │ A_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 5     │ A        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 3.77530348300… │ 8ad2ad        │ 0            │ True  │
│ 18            │ 0            │ A_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 5     │ A        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 3.27614904774… │ 1a0042        │ 0            │ True  │
│ 19            │ 0            │ A_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 5     │ A        │ 1.0       │ not_measured   │ 84.453469991… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 19            │ 0            │ A_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 5     │ A        │ 1.0       │ not_measured   │ 86.230160951… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 19            │ 0            │ A_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 5     │ A        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 1.17296486099… │ 96c0ed        │ 0            │ True  │
│ 19            │ 0            │ A_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 5     │ A        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 1.19764112432… │ 6cd3dc        │ 0            │ True  │
│ 20            │ 0            │ C_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 5     │ C        │ 1.0       │ not_measured   │ 85.679467439… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 20            │ 0            │ C_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 5     │ C        │ 1.0       │ not_measured   │ 95.863260507… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 20            │ 0            │ C_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 5     │ C        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 1.18999260332… │ b2ddab        │ 0            │ True  │
│ 20            │ 0            │ C_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 5     │ C        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 1.33143417371… │ 55db9b        │ 0            │ True  │
│ 21            │ 0            │ B_f1.0-c1.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 2     │ B        │ 1.0       │ not_measured   │ 298.81930494… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 21            │ 0            │ B_f1.0-c1.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 2     │ B        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 3.32021449936… │ ffbb95        │ 0            │ True  │
│ 22            │ 0            │ C_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 3     │ C        │ 0.0       │ not_measured   │ 598.88346576… │ Timed out.   │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 22            │ 0            │ C_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 3     │ C        │ 0.0       │ not_measured   │ 244.33887457… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 22            │ 0            │ C_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 3     │ C        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 9.98139109611… │ 91632b        │ 0            │ True  │
│ 22            │ 0            │ C_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 3     │ C        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 4.07231457630… │ 9e378c        │ 0            │ True  │
│ 23            │ 0            │ A_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 5     │ A        │ 0.0       │ not_measured   │ 106.07093071… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 23            │ 0            │ A_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 5     │ A        │ 0.0       │ not_measured   │ 130.30512285… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 23            │ 0            │ A_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 5     │ A        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.47320737110… │ 559d56        │ 0            │ True  │
│ 23            │ 0            │ A_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 5     │ A        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.80979337294… │ f3a447        │ 0            │ True  │
│ 24            │ 0            │ A_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 2     │ A        │ 0.0       │ not_measured   │ 378.31657004… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 24            │ 0            │ A_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 2     │ A        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 4.20351744492… │ 221473        │ 0            │ True  │
│ 25            │ 0            │ A_f1.0-c1.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 2     │ A        │ 1.0       │ not_measured   │ 291.90445613… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 25            │ 0            │ A_f1.0-c1.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 2     │ A        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 3.24338284598… │ 43333e        │ 0            │ True  │
│ 26            │ 0            │ A_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 4     │ A        │ 0.0       │ not_measured   │ 145.12948369… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 26            │ 0            │ A_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 4     │ A        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.61254981888… │ d984de        │ 0            │ True  │
│ 27            │ 0            │ C_f1.0-c1.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 3     │ C        │ 1.0       │ not_measured   │ 154.98134708… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 27            │ 0            │ C_f1.0-c1.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 3     │ C        │ 1.0       │ not_measured   │ 168.34859228… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 27            │ 0            │ C_f1.0-c1.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 3     │ C        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 2.58302245140… │ 63c714        │ 0            │ True  │
│ 27            │ 0            │ C_f1.0-c1.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 3     │ C        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 2.80580987135… │ 3390c8        │ 0            │ True  │
│ 28            │ 0            │ A_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 3     │ A        │ 1.0       │ not_measured   │ 168.36590766… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 28            │ 0            │ A_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 3     │ A        │ 1.0       │ not_measured   │ 170.15659737… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 28            │ 0            │ A_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 3     │ A        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 1.40304923057… │ 409b9e        │ 0            │ True  │
│ 28            │ 0            │ A_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 3     │ A        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 1.41797164479… │ 06db5c        │ 0            │ True  │
│ 29            │ 0            │ C_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 4     │ C        │ 1.0       │ not_measured   │ 121.42492485… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 29            │ 0            │ C_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 4     │ C        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 1.34916583167… │ 022af8        │ 0            │ True  │
│ 30            │ 0            │ B_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 3     │ B        │ 0.0       │ not_measured   │ 220.19828414… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 30            │ 0            │ B_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 3     │ B        │ 0.0       │ not_measured   │ 273.71202731… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 30            │ 0            │ B_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 3     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 3.66997140248… │ 3b4610        │ 0            │ True  │
│ 30            │ 0            │ B_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 3     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 4.56186712185… │ 6bf4dd        │ 0            │ True  │
│ 31            │ 0            │ B_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 2     │ B        │ 0.0       │ not_measured   │ 225.17914223… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 31            │ 0            │ B_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 2     │ B        │ 0.0       │ not_measured   │ 228.14362454… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 31            │ 0            │ B_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 2     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.25099523464… │ a302db        │ 0            │ True  │
│ 31            │ 0            │ B_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 2     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.26746458080… │ 62e595        │ 0            │ True  │
│ 32            │ 0            │ C_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 3     │ C        │ 1.0       │ not_measured   │ 168.91636371… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 32            │ 0            │ C_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 3     │ C        │ 1.0       │ not_measured   │ 174.03356242… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 32            │ 0            │ C_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 3     │ C        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 1.40763636430… │ 8bfb84        │ 0            │ True  │
│ 32            │ 0            │ C_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 3     │ C        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 1.45027968684… │ f339ac        │ 0            │ True  │
│ 33            │ 0            │ provider.B-c… │ replay.bench… │ 1.0        │ explicit_gri… │ 5     │ B        │ 1.0       │ Externally     │ not_measured  │ not_measured │ not_measured   │ randomwalk-1… │ 0            │ False │
│               │              │               │               │            │               │       │          │           │ defined        │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ experiments    │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ cannot be      │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ applied to     │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ entities:      │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ replay.benchm… │               │              │                │               │              │       │
│ 34            │ 0            │ B_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 4     │ B        │ 0.0       │ not_measured   │ 113.87676978… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 34            │ 0            │ B_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 4     │ B        │ 0.0       │ not_measured   │ 132.54151201… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 34            │ 0            │ B_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 4     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.26529744201… │ 91cc7f        │ 0            │ True  │
│ 34            │ 0            │ B_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 4     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.47268346680… │ 16affd        │ 0            │ True  │
│ 35            │ 0            │ B_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 2     │ B        │ 1.0       │ not_measured   │ 184.93504953… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 35            │ 0            │ B_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 2     │ B        │ 1.0       │ not_measured   │ 166.74843192… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 35            │ 0            │ B_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 2     │ B        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 1.02741694185… │ 128ae6        │ 0            │ True  │
│ 35            │ 0            │ B_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 2     │ B        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 0.92638017733… │ a75803        │ 0            │ True  │
│ 36            │ 0            │ C_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 3     │ C        │ 0.0       │ not_measured   │ 240.07358503… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 36            │ 0            │ C_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 3     │ C        │ 0.0       │ not_measured   │ 269.09066414… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 36            │ 0            │ C_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 3     │ C        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 2.00061320861… │ 24e5c5        │ 0            │ True  │
│ 36            │ 0            │ C_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 3     │ C        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 2.24242220123… │ fa537f        │ 0            │ True  │
│ 37            │ 0            │ A_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 2     │ A        │ 1.0       │ not_measured   │ 272.99782156… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 37            │ 0            │ A_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 2     │ A        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 1.51665456427… │ 1c5976        │ 0            │ True  │
│ 38            │ 0            │ A_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 4     │ A        │ 0.0       │ not_measured   │ 158.70639538… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 38            │ 0            │ A_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 4     │ A        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 3.52680878639… │ 9e4db8        │ 0            │ True  │
│ 39            │ 0            │ C_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 4     │ C        │ 0.0       │ not_measured   │ 177.72359776… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 39            │ 0            │ C_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 4     │ C        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 3.94941328366… │ 70dc0f        │ 0            │ True  │
│ 40            │ 0            │ A_f1.0-c1.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 4     │ A        │ 1.0       │ not_measured   │ 116.31417059… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 40            │ 0            │ A_f1.0-c1.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 4     │ A        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 2.58475934664… │ e809c3        │ 0            │ True  │
│ 41            │ 0            │ C_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 2     │ C        │ 0.0       │ not_measured   │ 415.82928490… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 41            │ 0            │ C_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 2     │ C        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 2.31016269392… │ 9995f8        │ 0            │ True  │
│ 42            │ 0            │ B_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 5     │ B        │ 0.0       │ not_measured   │ 141.99024295… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 42            │ 0            │ B_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 5     │ B        │ 0.0       │ not_measured   │ 168.79178500… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 42            │ 0            │ B_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 5     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 3.94417341550… │ 875849        │ 0            │ True  │
│ 42            │ 0            │ B_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 5     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 4.68866069449… │ a0cf51        │ 0            │ True  │
│ 43            │ 0            │ C_f1.0-c0.0-… │ replay.bench… │ 1.0        │ multi-cloud-… │ 2     │ C        │ 0.0       │ not_measured   │ 463.39653873… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 43            │ 0            │ C_f1.0-c0.0-… │ custom_exper… │ 1.0        │ multi-cloud-… │ 2     │ C        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 5.14885043038… │ 3b6b15        │ 0            │ True  │
│ 44            │ 0            │ C_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 4     │ C        │ 0.0       │ not_measured   │ 188.09087824… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 44            │ 0            │ C_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 4     │ C        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 2.08989864720… │ 0be80e        │ 0            │ True  │
│ 45            │ 0            │ A_f0.0-c1.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 4     │ A        │ 1.0       │ not_measured   │ 106.67012143… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 45            │ 0            │ A_f0.0-c1.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 4     │ A        │ 1.0       │ not_measured   │ not_measured  │ not_measured │ 1.18522357145… │ 4ae232        │ 0            │ True  │
│ 46            │ 0            │ B_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 3     │ B        │ 0.0       │ not_measured   │ 153.51639366… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 46            │ 0            │ B_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 3     │ B        │ 0.0       │ not_measured   │ 184.44801592… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 46            │ 0            │ B_f0.0-c0.0-… │ replay.bench… │ 0.0        │ multi-cloud-… │ 3     │ B        │ 0.0       │ not_measured   │ 176.28814435… │ ok           │ not_measured   │ replayed-mea… │ 0            │ True  │
│ 46            │ 0            │ B_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 3     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.27930328051… │ 4b56b9        │ 0            │ True  │
│ 46            │ 0            │ B_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 3     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.53706679940… │ d8119b        │ 0            │ True  │
│ 46            │ 0            │ B_f0.0-c0.0-… │ custom_exper… │ 0.0        │ multi-cloud-… │ 3     │ B        │ 0.0       │ not_measured   │ not_measured  │ not_measured │ 1.46906786958… │ 02acb3        │ 0            │ True  │
│ 47            │ 0            │ provider.B-c… │ replay.bench… │ 0.0        │ explicit_gri… │ 4     │ B        │ 1.0       │ Externally     │ not_measured  │ not_measured │ not_measured   │ randomwalk-1… │ 0            │ False │
│               │              │               │               │            │               │       │          │           │ defined        │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ experiments    │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ cannot be      │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ applied to     │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ entities:      │               │              │                │               │              │       │
│               │              │               │               │            │               │       │          │           │ replay.benchm… │               │              │                │               │              │       │
└───────────────┴──────────────┴───────────────┴───────────────┴────────────┴───────────────┴───────┴──────────┴───────────┴────────────────┴───────────────┴──────────────┴────────────────┴───────────────┴──────────────┴───────┘
```

<!-- markdownlint-enable line-length -->

> [!TIP] Some things to note and consider:
>
> - The table is in the order the points were measured
> - Some points have multiple measurements c.f. size of entityspace versus the
>   number of rows in `ml_export.csv`.
> - Some points were not measured - these are points in the discoveryspace for
>   which no data was present to replay.

## Exploring Further

Here are a variety of commands you can try after executing the example above:

### Viewing entities

There are multiple ways to view the entities related to a `discoveryspace`. Try:

```commandline
ado show entities space --use-latest
ado show entities space --use-latest --aggregate mean
ado show entities space --use-latest --include unmeasured
ado show entities space --use-latest --property-format target
```

Also,

```commandline
ado show details space --use-latest
```

will give you a summary of what has been measured.

> [!NOTE]
>
> If you want to run these commands with the latest space created use the
> `--use-latest` flag as above

### Resource provenance

The `related` sub-command shows resource provenance:

```commandline
ado show related operation --use-latest
```

### Operation timeseries

The following commands give more details of the operation timeseries:

```commandline
ado show results operation --use-latest
ado show requests operation --use-latest
```

### Resource templates

Another helpful command is `template` which will output a default example of a
resource YAML along with an (optional) description of its fields. Try:

```commandline
ado template operation --include-schema --operator-name random_walk
```

### Rerun

An interesting thing to try is to run the operation again and compare the output
of `show entities operation` for the two operations, and `show entities space`.

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
  been completed on an entity and reuse it
- **provenance**: `ado` stores the relationship between the resources it
  creates.
- **results viewing**: `ado show entities` outputs the data in a
  `discoveryspace` or measured in an `operation`.
- **measurement timeseries**: The sequence (timeseries) of measurements,
  successful or not, of each `operation` is preserved.
- **`discoveryspace` views**: By default `ado show entities space` only shows
  successfully measured entities, but you can see what has not been measured if
  you want.
