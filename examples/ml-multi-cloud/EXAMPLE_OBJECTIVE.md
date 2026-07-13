# Search based on a custom objective function

> [!NOTE]
>
> This example shows how to create and use a custom objective function as a
> dependent experiment — one that consumes the output of another experiment —
> with `ado`.

## The scenario

Often, experiments will not directly produce the value that you are interested
in. For example, an experiment might measure the run time of an application,
while **the meaningful metric is the associated cost, which requires knowing
information like the cost per hour of the GPUs used**. Another common scenario
involves aggregating data points from one or more experiments into a single
value.

In this example we will install **a custom objective function that calculates a
cost** for the application workload configurations used in the
[taking a random walk example](/ado/examples/random-walk/). When the workload
configuration space is explored using a random walk, both the `wallClockRuntime`
and the `cost`, as defined by the custom function, will be measured.

> [!CAUTION]
>
> The commands below assume you are in the directory `examples/ml-multi-cloud`
> in **the ado source repository**. See
> [the instructions for cloning the repository](/ado/getting-started/install/#__tabbed_1_3).

## Prerequisites

### Install the ray_tune ado operator

If you haven't already installed the ray_tune operator, run:

```commandline
pip install ado-ray-tune
```

Then verify the operator is registered:

```commandline
ado get operators
```

The output should show an entry for `ray_tune`:

```commandline
Available operators by type:
┌───────┬─────────────┬─────────┬─────────┐
│ INDEX │ OPERATOR    │ VERSION │ TYPE    │
├───────┼─────────────┼─────────┼─────────┤
│ 0     │ random_walk │ 2.0.0   │ explore │
│ 1     │ ray_tune    │ 2.0.3   │ explore │
│ 2     │ rifferla    │ 2.0.3   │ modify  │
└───────┴─────────────┴─────────┴─────────┘
```

## Installing the custom experiment

The custom experiment is defined in a Python package under
`custom_experiment/`. To install it run:

```commandline
pip install custom_experiment/
```

then

```commandline
ado get experiments --details
```

will output something similar to:

<!-- markdownlint-disable line-length -->

```commandline
┌───────┬────────────────────┬─────────────────────┬─────────┬─────────────┐
│ INDEX │ ACTUATOR ID        │ EXPERIMENT ID       │ VERSION │ DESCRIPTION │
├───────┼────────────────────┼─────────────────────┼─────────┼─────────────┤
│ 0     │ custom_experiments │ ml-multicloud-cost  │ 1.0.0   │             │
│ 1     │ mock               │ test-experiment     │ None    │             │
│ 2     │ mock               │ test-experiment-two │ None    │             │
└───────┴────────────────────┴─────────────────────┴─────────┴─────────────┘
```

<!-- markdownlint-enable line-length -->

You can see the custom experiment provided by the package,
**ml-multicloud-cost** on the first line. Executing
`ado describe experiment ml-multicloud-cost` outputs:

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

From this, you can see the `ml-multicloud-cost` requires an observed
property, i.e. a property measured by another experiment, as input. From the
observed property identifier, the experiment is called `benchmark_performance`
and the property is `wallClockRuntime`.

## Create a discoveryspace that uses the custom experiment

First create a `samplestore` with the `ml-multi-cloud` example data following
[these instructions](/ado/examples/random-walk/#using-pre-existing-data-with-ado).
If you have already completed the
[taking a random walk example](/ado/examples/random-walk/), reuse the
`samplestore` you created there. If you cannot recall the identifier, run:

```commandline
ado get samplestores
```

and set it as an environment variable before the next step:

```commandline
export SAMPLE_STORE_IDENTIFIER=<your-samplestore-identifier>
```

To use the custom experiment, you must add it in the `experiments` list of a
`discoveryspace`. The `actuatorIdentifier` will be `custom_experiments` and the
`experimentIdentifier` will be the name of your experiment. For this case the
relevant section looks like:

```yaml
experiments:
  - experimentIdentifier: "benchmark_performance"
    actuatorIdentifier: "replay"
  - experimentIdentifier: "ml-multicloud-cost"
    actuatorIdentifier: "custom_experiments"
    experimentVersion: 1.0.0
```

The complete `discoveryspace` for this example is given in
`ml_multicloud_space_with_custom.yaml` To create it execute:

```commandline
ado create space -f ml_multicloud_space_with_custom.yaml --set "sampleStoreIdentifier=$SAMPLE_STORE_IDENTIFIER"
```

> [!IMPORTANT]
>
> If an experiment takes the output of another experiment as input both
> experiments must be in the `discoveryspace`. In the above example if the entry
> `benchmark_performance` was omitted the `ado create space` command would fail
> with:
>
> **SpaceInconsistencyError**: MeasurementSpace does not contain an experiment
> measuring an observed property required by another experiment in the space

You can view a description of the space using the `ado describe` command:

```commandline
ado describe space --use-latest
```

This will output:

<!-- markdownlint-disable line-length -->

```terminaloutput
Identifier: 'space-e97d3e-7a4ba9'

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

      base identifier                       ┃ required major version ┃ parameterization
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━
      replay.benchmark_performance          ┃ None                   ┃ None
      custom_experiments.ml-multicloud-cost ┃ v1                     ┃ None

    ───────────────────────────────────── benchmark_performance ─────────────────────────────────────
     Expected Interface

     Inputs:

        parameter  ┃ type     ┃ value ┃ parameterized
       ━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━╋━━━━━━━━━━━━━━━
        cpu_family ┃ required ┃ None  ┃ na
        nodes      ┃ required ┃ None  ┃ na
        provider   ┃ required ┃ None  ┃ na
        vcpu_size  ┃ required ┃ None  ┃ na

     Outputs:

        target property
       ━━━━━━━━━━━━━━━━━━
        wallClockRuntime
        status

    ─────────────────────────────────────────────────────────────────────────────────────────────────

    ───────────────────────────────────── ml-multicloud-cost@v1 ─────────────────────────────────────
     Expected Interface (from v1.0.0)

     Inputs:

        parameter                              ┃ type     ┃ value ┃ parameterized
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━╋━━━━━━━━━━━━━━━
        nodes                                  ┃ required ┃ None  ┃ na
        cpu_family                             ┃ required ┃ None  ┃ na
        benchmark_performance-wallClockRuntime ┃ required ┃ None  ┃ na

     Outputs:

        target property
       ━━━━━━━━━━━━━━━━━
        total_cost

    ─────────────────────────────────────────────────────────────────────────────────────────────────


Sample Store identifier: 7a4ba9
```

<!-- markdownlint-enable line-length -->

## Exploring the `discoveryspace`

To run a `randomwalk` operation on the new space, execute:

```commandline
ado create operation -f randomwalk_ml_multicloud_operation.yaml --use-latest space
```

This produces an output similar to that described in the
[taking a random walk example](/ado/examples/random-walk/#exploring-the-discoveryspace)
and will exit printing the operation identifier. However, in this case there is
additional information related to the dependent experiment.

When it completes, you can get a table of the points visited with:

```commandline
ado show measurements operation --use-latest
```

You will see a table similar to the following - note the extra column for the
new cost function:

<!-- markdownlint-disable line-length -->

```commandline
┌───────────────┬──────────────┬─────────────────────────────────────────────┬─────────────────────────────────────────────┬────────────┬────────────────────────────────┬───────┬──────────┬───────────┬──────────────────────────────────────────────────────────────────────────────────────────────┬────────────────────┬──────────────┬────────────────────┬─────────────────────────────────┬──────────────┬───────┐
│ request_index │ result_index │ identifier                                  │ experiment_id                               │ cpu_family │ generatorid                    │ nodes │ provider │ vcpu_size │ reason                                                                                       │ wallClockRuntime   │ status       │ total_cost         │ request_id                      │ entity_index │ valid │
├───────────────┼──────────────┼─────────────────────────────────────────────┼─────────────────────────────────────────────┼────────────┼────────────────────────────────┼───────┼──────────┼───────────┼──────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────┼──────────────┼────────────────────┼─────────────────────────────────┼──────────────┼───────┤
│ 0             │ 0            │ cpu_family.1-nodes.4-provider.B-vcpu_size.1 │ replay.benchmark_performance                │ 1.0        │ explicit_grid_sample_generator │ 4     │ B        │ 1.0       │ Externally defined experiments cannot be applied to entities: replay.benchmark_performance.  │ not_measured       │ not_measured │ not_measured       │ random_walk@2.0.0-b9162b-23292a │ 0            │ False │
│ 1             │ 0            │ B_f0.0-c0.0-n3                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 3     │ B        │ 0.0       │ not_measured                                                                                 │ 153.51639366149902 │ ok           │ not_measured       │ replayed-measurement-388b80     │ 0            │ True  │
│ 1             │ 0            │ B_f0.0-c0.0-n3                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 3     │ B        │ 0.0       │ not_measured                                                                                 │ 184.44801592826843 │ ok           │ not_measured       │ replayed-measurement-f6fed2     │ 0            │ True  │
│ 1             │ 0            │ B_f0.0-c0.0-n3                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 3     │ B        │ 0.0       │ not_measured                                                                                 │ 176.28814435005188 │ ok           │ not_measured       │ replayed-measurement-72f30a     │ 0            │ True  │
│ 1             │ 0            │ B_f0.0-c0.0-n3                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 3     │ B        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.2793032805124918 │ 7e0c0d                          │ 0            │ True  │
│ 1             │ 0            │ B_f0.0-c0.0-n3                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 3     │ B        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.537066799402237  │ 38212e                          │ 0            │ True  │
│ 1             │ 0            │ B_f0.0-c0.0-n3                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 3     │ B        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.4690678695837658 │ 4c9339                          │ 0            │ True  │
│ 2             │ 0            │ cpu_family.1-nodes.5-provider.B-vcpu_size.1 │ replay.benchmark_performance                │ 1.0        │ explicit_grid_sample_generator │ 5     │ B        │ 1.0       │ Externally defined experiments cannot be applied to entities: replay.benchmark_performance.  │ not_measured       │ not_measured │ not_measured       │ random_walk@2.0.0-b9162b-b9d77f │ 0            │ False │
│ 3             │ 0            │ A_f0.0-c1.0-n2                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 2     │ A        │ 1.0       │ not_measured                                                                                 │ 272.99782156944275 │ ok           │ not_measured       │ replayed-measurement-e97853     │ 0            │ True  │
│ 3             │ 0            │ A_f0.0-c1.0-n2                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 2     │ A        │ 1.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.516654564274682  │ 04719f                          │ 0            │ True  │
│ 4             │ 0            │ cpu_family.0-nodes.3-provider.B-vcpu_size.1 │ replay.benchmark_performance                │ 0.0        │ explicit_grid_sample_generator │ 3     │ B        │ 1.0       │ Externally defined experiments cannot be applied to entities: replay.benchmark_performance.  │ not_measured       │ not_measured │ not_measured       │ random_walk@2.0.0-b9162b-5077c8 │ 0            │ False │
│ 5             │ 0            │ C_f0.0-c1.0-n3                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 3     │ C        │ 1.0       │ not_measured                                                                                 │ 168.9163637161255  │ ok           │ not_measured       │ replayed-measurement-3d1ea5     │ 0            │ True  │
│ 5             │ 0            │ C_f0.0-c1.0-n3                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 3     │ C        │ 1.0       │ not_measured                                                                                 │ 174.0335624217987  │ ok           │ not_measured       │ replayed-measurement-9cfa0d     │ 0            │ True  │
│ 5             │ 0            │ C_f0.0-c1.0-n3                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 3     │ C        │ 1.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.4076363643010457 │ bb07d8                          │ 0            │ True  │
│ 5             │ 0            │ C_f0.0-c1.0-n3                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 3     │ C        │ 1.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.4502796868483225 │ 897239                          │ 0            │ True  │
│ 6             │ 0            │ B_f0.0-c1.0-n2                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 2     │ B        │ 1.0       │ not_measured                                                                                 │ 184.935049533844   │ ok           │ not_measured       │ replayed-measurement-a2b6d5     │ 0            │ True  │
│ 6             │ 0            │ B_f0.0-c1.0-n2                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 2     │ B        │ 1.0       │ not_measured                                                                                 │ 166.74843192100525 │ ok           │ not_measured       │ replayed-measurement-0f11c2     │ 0            │ True  │
│ 6             │ 0            │ B_f0.0-c1.0-n2                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 2     │ B        │ 1.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.0274169418546888 │ 0051fa                          │ 0            │ True  │
│ 6             │ 0            │ B_f0.0-c1.0-n2                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 2     │ B        │ 1.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 0.9263801773389181 │ ac2cd2                          │ 0            │ True  │
│ 7             │ 0            │ C_f1.0-c1.0-n2                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 2     │ C        │ 1.0       │ not_measured                                                                                 │ 363.2856709957123  │ ok           │ not_measured       │ replayed-measurement-2b56dd     │ 0            │ True  │
│ 7             │ 0            │ C_f1.0-c1.0-n2                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 2     │ C        │ 1.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 4.036507455507914  │ 2c55b8                          │ 0            │ True  │
│ 8             │ 0            │ A_f1.0-c1.0-n3                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 3     │ A        │ 1.0       │ not_measured                                                                                 │ 151.58562421798706 │ ok           │ not_measured       │ replayed-measurement-e98ec9     │ 0            │ True  │
│ 8             │ 0            │ A_f1.0-c1.0-n3                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 3     │ A        │ 1.0       │ not_measured                                                                                 │ 155.02856159210205 │ ok           │ not_measured       │ replayed-measurement-8c6aa6     │ 0            │ True  │
│ 8             │ 0            │ A_f1.0-c1.0-n3                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 3     │ A        │ 1.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 2.526427070299784  │ 6cdde5                          │ 0            │ True  │
│ 8             │ 0            │ A_f1.0-c1.0-n3                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 3     │ A        │ 1.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 2.5838093598683676 │ c40f66                          │ 0            │ True  │
│ 9             │ 0            │ C_f0.0-c0.0-n3                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 3     │ C        │ 0.0       │ not_measured                                                                                 │ 240.07358503341675 │ ok           │ not_measured       │ replayed-measurement-34c6fd     │ 0            │ True  │
│ 9             │ 0            │ C_f0.0-c0.0-n3                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 3     │ C        │ 0.0       │ not_measured                                                                                 │ 269.0906641483307  │ ok           │ not_measured       │ replayed-measurement-4ba47d     │ 0            │ True  │
│ 9             │ 0            │ C_f0.0-c0.0-n3                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 3     │ C        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 2.000613208611806  │ 357f1b                          │ 0            │ True  │
│ 9             │ 0            │ C_f0.0-c0.0-n3                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 3     │ C        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 2.242422201236089  │ 918216                          │ 0            │ True  │
│ 10            │ 0            │ C_f0.0-c1.0-n5                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 5     │ C        │ 1.0       │ not_measured                                                                                 │ 85.67946743965149  │ ok           │ not_measured       │ replayed-measurement-2fea9b     │ 0            │ True  │
│ 10            │ 0            │ C_f0.0-c1.0-n5                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 5     │ C        │ 1.0       │ not_measured                                                                                 │ 95.86326050758362  │ ok           │ not_measured       │ replayed-measurement-86da3f     │ 0            │ True  │
│ 10            │ 0            │ C_f0.0-c1.0-n5                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 5     │ C        │ 1.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.1899926033284929 │ 42c3ce                          │ 0            │ True  │
│ 10            │ 0            │ C_f0.0-c1.0-n5                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 5     │ C        │ 1.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.331434173716439  │ b9a2b2                          │ 0            │ True  │
│ 11            │ 0            │ C_f0.0-c1.0-n2                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 2     │ C        │ 1.0       │ not_measured                                                                                 │ 309.8423240184784  │ ok           │ not_measured       │ replayed-measurement-29cbea     │ 0            │ True  │
│ 11            │ 0            │ C_f0.0-c1.0-n2                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 2     │ C        │ 1.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.7213462445471022 │ e24d38                          │ 0            │ True  │
│ 12            │ 0            │ B_f0.0-c0.0-n2                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 2     │ B        │ 0.0       │ not_measured                                                                                 │ 225.1791422367096  │ ok           │ not_measured       │ replayed-measurement-9f0213     │ 0            │ True  │
│ 12            │ 0            │ B_f0.0-c0.0-n2                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 2     │ B        │ 0.0       │ not_measured                                                                                 │ 228.14362454414368 │ ok           │ not_measured       │ replayed-measurement-529f8e     │ 0            │ True  │
│ 12            │ 0            │ B_f0.0-c0.0-n2                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 2     │ B        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.2509952346483866 │ d98259                          │ 0            │ True  │
│ 12            │ 0            │ B_f0.0-c0.0-n2                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 2     │ B        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.267464580800798  │ 8de27c                          │ 0            │ True  │
│ 13            │ 0            │ A_f0.0-c0.0-n3                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 3     │ A        │ 0.0       │ not_measured                                                                                 │ 221.5101969242096  │ ok           │ not_measured       │ replayed-measurement-eaf2d6     │ 0            │ True  │
│ 13            │ 0            │ A_f0.0-c0.0-n3                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 3     │ A        │ 0.0       │ not_measured                                                                                 │ 216.394127368927   │ ok           │ not_measured       │ replayed-measurement-ce26be     │ 0            │ True  │
│ 13            │ 0            │ A_f0.0-c0.0-n3                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 3     │ A        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.8459183077017467 │ c6994f                          │ 0            │ True  │
│ 13            │ 0            │ A_f0.0-c0.0-n3                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 3     │ A        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.8032843947410584 │ 5df6fe                          │ 0            │ True  │
│ 14            │ 0            │ C_f1.0-c0.0-n3                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 3     │ C        │ 0.0       │ not_measured                                                                                 │ 598.8834657669067  │ Timed out.   │ not_measured       │ replayed-measurement-d44bc1     │ 0            │ True  │
│ 14            │ 0            │ C_f1.0-c0.0-n3                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 3     │ C        │ 0.0       │ not_measured                                                                                 │ 244.33887457847595 │ ok           │ not_measured       │ replayed-measurement-64bf52     │ 0            │ True  │
│ 14            │ 0            │ C_f1.0-c0.0-n3                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 3     │ C        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 9.981391096115113  │ fe4dab                          │ 0            │ True  │
│ 14            │ 0            │ C_f1.0-c0.0-n3                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 3     │ C        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 4.0723145763079325 │ 8d3d2f                          │ 0            │ True  │
│ 15            │ 0            │ B_f0.0-c0.0-n5                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 5     │ B        │ 0.0       │ not_measured                                                                                 │ 103.90595746040344 │ ok           │ not_measured       │ replayed-measurement-9bd12b     │ 0            │ True  │
│ 15            │ 0            │ B_f0.0-c0.0-n5                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 5     │ B        │ 0.0       │ not_measured                                                                                 │ 112.7056987285614  │ ok           │ not_measured       │ replayed-measurement-35270f     │ 0            │ True  │
│ 15            │ 0            │ B_f0.0-c0.0-n5                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 5     │ B        │ 0.0       │ not_measured                                                                                 │ 113.88505148887634 │ ok           │ not_measured       │ replayed-measurement-d49497     │ 0            │ True  │
│ 15            │ 0            │ B_f0.0-c0.0-n5                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 5     │ B        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.443138298061159  │ 028d22                          │ 0            │ True  │
│ 15            │ 0            │ B_f0.0-c0.0-n5                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 5     │ B        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.5653569267855751 │ 23237f                          │ 0            │ True  │
│ 15            │ 0            │ B_f0.0-c0.0-n5                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 5     │ B        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.5817368262343936 │ 60610c                          │ 0            │ True  │
│ 16            │ 0            │ cpu_family.0-nodes.4-provider.B-vcpu_size.1 │ replay.benchmark_performance                │ 0.0        │ explicit_grid_sample_generator │ 4     │ B        │ 1.0       │ Externally defined experiments cannot be applied to entities: replay.benchmark_performance.  │ not_measured       │ not_measured │ not_measured       │ random_walk@2.0.0-b9162b-c8d9e7 │ 0            │ False │
│ 17            │ 0            │ A_f0.0-c1.0-n3                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 3     │ A        │ 1.0       │ not_measured                                                                                 │ 168.36590766906738 │ ok           │ not_measured       │ replayed-measurement-9613ce     │ 0            │ True  │
│ 17            │ 0            │ A_f0.0-c1.0-n3                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 3     │ A        │ 1.0       │ not_measured                                                                                 │ 170.15659737586975 │ ok           │ not_measured       │ replayed-measurement-24b1a2     │ 0            │ True  │
│ 17            │ 0            │ A_f0.0-c1.0-n3                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 3     │ A        │ 1.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.4030492305755615 │ 7ff444                          │ 0            │ True  │
│ 17            │ 0            │ A_f0.0-c1.0-n3                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 3     │ A        │ 1.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.4179716447989146 │ 8a5c8d                          │ 0            │ True  │
│ 18            │ 0            │ B_f1.0-c0.0-n4                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 4     │ B        │ 0.0       │ not_measured                                                                                 │ 202.48239731788635 │ ok           │ not_measured       │ replayed-measurement-f2930c     │ 0            │ True  │
│ 18            │ 0            │ B_f1.0-c0.0-n4                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 4     │ B        │ 0.0       │ not_measured                                                                                 │ 193.55997109413147 │ ok           │ not_measured       │ replayed-measurement-7a382f     │ 0            │ True  │
│ 18            │ 0            │ B_f1.0-c0.0-n4                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 4     │ B        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 4.499608829286363  │ 1bf59d                          │ 0            │ True  │
│ 18            │ 0            │ B_f1.0-c0.0-n4                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 4     │ B        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 4.301332690980699  │ 6450d1                          │ 0            │ True  │
│ 19            │ 0            │ C_f1.0-c0.0-n5                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 5     │ C        │ 0.0       │ not_measured                                                                                 │ 136.3071050643921  │ ok           │ not_measured       │ replayed-measurement-620e18     │ 0            │ True  │
│ 19            │ 0            │ C_f1.0-c0.0-n5                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 5     │ C        │ 0.0       │ not_measured                                                                                 │ 135.47050046920776 │ ok           │ not_measured       │ replayed-measurement-1eb108     │ 0            │ True  │
│ 19            │ 0            │ C_f1.0-c0.0-n5                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 5     │ C        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 3.786308474010892  │ 8cfe75                          │ 0            │ True  │
│ 19            │ 0            │ C_f1.0-c0.0-n5                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 5     │ C        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 3.763069457477993  │ 3eb2f7                          │ 0            │ True  │
│ 20            │ 0            │ C_f1.0-c1.0-n5                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 5     │ C        │ 1.0       │ not_measured                                                                                 │ 92.17141437530518  │ ok           │ not_measured       │ replayed-measurement-bcae10     │ 0            │ True  │
│ 20            │ 0            │ C_f1.0-c1.0-n5                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 5     │ C        │ 1.0       │ not_measured                                                                                 │ 100.97977471351624 │ ok           │ not_measured       │ replayed-measurement-b4b029     │ 0            │ True  │
│ 20            │ 0            │ C_f1.0-c1.0-n5                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 5     │ C        │ 1.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 2.560317065980699  │ 58c71d                          │ 0            │ True  │
│ 20            │ 0            │ C_f1.0-c1.0-n5                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 5     │ C        │ 1.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 2.8049937420421176 │ 79515e                          │ 0            │ True  │
│ 21            │ 0            │ B_f1.0-c0.0-n3                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 3     │ B        │ 0.0       │ not_measured                                                                                 │ 220.19828414916992 │ ok           │ not_measured       │ replayed-measurement-87de2c     │ 0            │ True  │
│ 21            │ 0            │ B_f1.0-c0.0-n3                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 3     │ B        │ 0.0       │ not_measured                                                                                 │ 273.7120273113251  │ ok           │ not_measured       │ replayed-measurement-9b477f     │ 0            │ True  │
│ 21            │ 0            │ B_f1.0-c0.0-n3                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 3     │ B        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 3.669971402486165  │ c22b3b                          │ 0            │ True  │
│ 21            │ 0            │ B_f1.0-c0.0-n3                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 3     │ B        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 4.561867121855418  │ de2f2a                          │ 0            │ True  │
│ 22            │ 0            │ B_f1.0-c1.0-n2                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 2     │ B        │ 1.0       │ not_measured                                                                                 │ 298.8193049430847  │ ok           │ not_measured       │ replayed-measurement-d280ee     │ 0            │ True  │
│ 22            │ 0            │ B_f1.0-c1.0-n2                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 2     │ B        │ 1.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 3.320214499367608  │ 78a5d5                          │ 0            │ True  │
│ 23            │ 0            │ A_f0.0-c1.0-n5                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 5     │ A        │ 1.0       │ not_measured                                                                                 │ 84.45346999168396  │ ok           │ not_measured       │ replayed-measurement-d89559     │ 0            │ True  │
│ 23            │ 0            │ A_f0.0-c1.0-n5                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 5     │ A        │ 1.0       │ not_measured                                                                                 │ 86.23016095161438  │ ok           │ not_measured       │ replayed-measurement-2544bc     │ 0            │ True  │
│ 23            │ 0            │ A_f0.0-c1.0-n5                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 5     │ A        │ 1.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.1729648609956105 │ dfd7de                          │ 0            │ True  │
│ 23            │ 0            │ A_f0.0-c1.0-n5                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 5     │ A        │ 1.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.1976411243279776 │ df4a08                          │ 0            │ True  │
│ 24            │ 0            │ A_f0.0-c0.0-n5                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 5     │ A        │ 0.0       │ not_measured                                                                                 │ 106.0709307193756  │ ok           │ not_measured       │ replayed-measurement-2d75c4     │ 0            │ True  │
│ 24            │ 0            │ A_f0.0-c0.0-n5                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 5     │ A        │ 0.0       │ not_measured                                                                                 │ 130.30512285232544 │ ok           │ not_measured       │ replayed-measurement-d67b4c     │ 0            │ True  │
│ 24            │ 0            │ A_f0.0-c0.0-n5                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 5     │ A        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.473207371102439  │ de56c9                          │ 0            │ True  │
│ 24            │ 0            │ A_f0.0-c0.0-n5                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 5     │ A        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.8097933729489646 │ 4c8d3f                          │ 0            │ True  │
│ 25            │ 0            │ A_f0.0-c1.0-n4                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 4     │ A        │ 1.0       │ not_measured                                                                                 │ 106.67012143135072 │ ok           │ not_measured       │ replayed-measurement-6db22c     │ 0            │ True  │
│ 25            │ 0            │ A_f0.0-c1.0-n4                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 4     │ A        │ 1.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.1852235714594526 │ d49866                          │ 0            │ True  │
│ 26            │ 0            │ C_f1.0-c0.0-n4                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 4     │ C        │ 0.0       │ not_measured                                                                                 │ 177.72359776496887 │ ok           │ not_measured       │ replayed-measurement-6aa6bd     │ 0            │ True  │
│ 26            │ 0            │ C_f1.0-c0.0-n4                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 4     │ C        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 3.949413283665975  │ 28e887                          │ 0            │ True  │
│ 27            │ 0            │ B_f1.0-c0.0-n2                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 2     │ B        │ 0.0       │ not_measured                                                                                 │ 346.0709958076477  │ ok           │ not_measured       │ replayed-measurement-de4d09     │ 0            │ True  │
│ 27            │ 0            │ B_f1.0-c0.0-n2                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 2     │ B        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 3.8452332867516414 │ 2d4d5b                          │ 0            │ True  │
│ 28            │ 0            │ A_f1.0-c0.0-n4                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 4     │ A        │ 0.0       │ not_measured                                                                                 │ 158.70639538764954 │ ok           │ not_measured       │ replayed-measurement-e4a7c4     │ 0            │ True  │
│ 28            │ 0            │ A_f1.0-c0.0-n4                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 4     │ A        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 3.526808786392212  │ 001f30                          │ 0            │ True  │
│ 29            │ 0            │ A_f0.0-c0.0-n2                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 2     │ A        │ 0.0       │ not_measured                                                                                 │ 335.2085180282593  │ ok           │ not_measured       │ replayed-measurement-875579     │ 0            │ True  │
│ 29            │ 0            │ A_f0.0-c0.0-n2                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 2     │ A        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.8622695446014403 │ 88da28                          │ 0            │ True  │
│ 30            │ 0            │ C_f0.0-c0.0-n2                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 2     │ C        │ 0.0       │ not_measured                                                                                 │ 415.8292849063873  │ ok           │ not_measured       │ replayed-measurement-a7dbc4     │ 0            │ True  │
│ 30            │ 0            │ C_f0.0-c0.0-n2                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 2     │ C        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 2.310162693924374  │ 7e8018                          │ 0            │ True  │
│ 31            │ 0            │ C_f1.0-c0.0-n2                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 2     │ C        │ 0.0       │ not_measured                                                                                 │ 463.396538734436   │ ok           │ not_measured       │ replayed-measurement-565ced     │ 0            │ True  │
│ 31            │ 0            │ C_f1.0-c0.0-n2                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 2     │ C        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 5.148850430382622  │ 840719                          │ 0            │ True  │
│ 32            │ 0            │ cpu_family.1-nodes.3-provider.B-vcpu_size.1 │ replay.benchmark_performance                │ 1.0        │ explicit_grid_sample_generator │ 3     │ B        │ 1.0       │ Externally defined experiments cannot be applied to entities: replay.benchmark_performance.  │ not_measured       │ not_measured │ not_measured       │ random_walk@2.0.0-b9162b-80c504 │ 0            │ False │
│ 33            │ 0            │ C_f1.0-c1.0-n4                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 4     │ C        │ 1.0       │ not_measured                                                                                 │ 114.01436853408812 │ ok           │ not_measured       │ replayed-measurement-f00e3f     │ 0            │ True  │
│ 33            │ 0            │ C_f1.0-c1.0-n4                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 4     │ C        │ 1.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 2.5336526340908474 │ d66a7d                          │ 0            │ True  │
│ 34            │ 0            │ cpu_family.0-nodes.5-provider.B-vcpu_size.1 │ replay.benchmark_performance                │ 0.0        │ explicit_grid_sample_generator │ 5     │ B        │ 1.0       │ Externally defined experiments cannot be applied to entities: replay.benchmark_performance.  │ not_measured       │ not_measured │ not_measured       │ random_walk@2.0.0-b9162b-763789 │ 0            │ False │
│ 35            │ 0            │ A_f1.0-c0.0-n2                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 2     │ A        │ 0.0       │ not_measured                                                                                 │ 378.31657004356384 │ ok           │ not_measured       │ replayed-measurement-19c4d6     │ 0            │ True  │
│ 35            │ 0            │ A_f1.0-c0.0-n2                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 2     │ A        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 4.203517444928488  │ 0cdb9f                          │ 0            │ True  │
│ 36            │ 0            │ C_f1.0-c1.0-n3                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 3     │ C        │ 1.0       │ not_measured                                                                                 │ 154.9813470840454  │ ok           │ not_measured       │ replayed-measurement-d1ae0c     │ 0            │ True  │
│ 36            │ 0            │ C_f1.0-c1.0-n3                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 3     │ C        │ 1.0       │ not_measured                                                                                 │ 168.34859228134155 │ ok           │ not_measured       │ replayed-measurement-85fcf1     │ 0            │ True  │
│ 36            │ 0            │ C_f1.0-c1.0-n3                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 3     │ C        │ 1.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 2.583022451400757  │ 10026b                          │ 0            │ True  │
│ 36            │ 0            │ C_f1.0-c1.0-n3                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 3     │ C        │ 1.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 2.8058098713556925 │ 8907b6                          │ 0            │ True  │
│ 37            │ 0            │ A_f1.0-c0.0-n5                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 5     │ A        │ 0.0       │ not_measured                                                                                 │ 135.91092538833618 │ ok           │ not_measured       │ replayed-measurement-509209     │ 0            │ True  │
│ 37            │ 0            │ A_f1.0-c0.0-n5                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 5     │ A        │ 0.0       │ not_measured                                                                                 │ 117.94136571884157 │ ok           │ not_measured       │ replayed-measurement-0f0ec2     │ 0            │ True  │
│ 37            │ 0            │ A_f1.0-c0.0-n5                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 5     │ A        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 3.775303483009338  │ f96e48                          │ 0            │ True  │
│ 37            │ 0            │ A_f1.0-c0.0-n5                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 5     │ A        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 3.276149047745599  │ 3f5b67                          │ 0            │ True  │
│ 38            │ 0            │ B_f1.0-c0.0-n5                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 5     │ B        │ 0.0       │ not_measured                                                                                 │ 141.99024295806885 │ ok           │ not_measured       │ replayed-measurement-b48ec5     │ 0            │ True  │
│ 38            │ 0            │ B_f1.0-c0.0-n5                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 5     │ B        │ 0.0       │ not_measured                                                                                 │ 168.79178500175476 │ ok           │ not_measured       │ replayed-measurement-b30f1f     │ 0            │ True  │
│ 38            │ 0            │ B_f1.0-c0.0-n5                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 5     │ B        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 3.944173415501912  │ 3c8e7b                          │ 0            │ True  │
│ 38            │ 0            │ B_f1.0-c0.0-n5                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 5     │ B        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 4.688660694493188  │ 672673                          │ 0            │ True  │
│ 39            │ 0            │ A_f1.0-c0.0-n3                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 3     │ A        │ 0.0       │ not_measured                                                                                 │ 206.74496150016785 │ ok           │ not_measured       │ replayed-measurement-1a7b8c     │ 0            │ True  │
│ 39            │ 0            │ A_f1.0-c0.0-n3                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 3     │ A        │ 0.0       │ not_measured                                                                                 │ 236.1715066432953  │ ok           │ not_measured       │ replayed-measurement-04e9be     │ 0            │ True  │
│ 39            │ 0            │ A_f1.0-c0.0-n3                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 3     │ A        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 3.4457493583361307 │ 4e6aee                          │ 0            │ True  │
│ 39            │ 0            │ A_f1.0-c0.0-n3                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 3     │ A        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 3.936191777388255  │ d59c4b                          │ 0            │ True  │
│ 40            │ 0            │ B_f0.0-c0.0-n4                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 4     │ B        │ 0.0       │ not_measured                                                                                 │ 113.87676978111269 │ ok           │ not_measured       │ replayed-measurement-fc84ac     │ 0            │ True  │
│ 40            │ 0            │ B_f0.0-c0.0-n4                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 4     │ B        │ 0.0       │ not_measured                                                                                 │ 132.5415120124817  │ ok           │ not_measured       │ replayed-measurement-da45c6     │ 0            │ True  │
│ 40            │ 0            │ B_f0.0-c0.0-n4                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 4     │ B        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.265297442012363  │ 5a843c                          │ 0            │ True  │
│ 40            │ 0            │ B_f0.0-c0.0-n4                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 4     │ B        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.4726834668053521 │ 08e14c                          │ 0            │ True  │
│ 41            │ 0            │ A_f1.0-c1.0-n4                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 4     │ A        │ 1.0       │ not_measured                                                                                 │ 116.31417059898376 │ ok           │ not_measured       │ replayed-measurement-e584f8     │ 0            │ True  │
│ 41            │ 0            │ A_f1.0-c1.0-n4                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 4     │ A        │ 1.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 2.5847593466440837 │ a7aaed                          │ 0            │ True  │
│ 42            │ 0            │ C_f0.0-c0.0-n4                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 4     │ C        │ 0.0       │ not_measured                                                                                 │ 188.09087824821472 │ ok           │ not_measured       │ replayed-measurement-a547a8     │ 0            │ True  │
│ 42            │ 0            │ C_f0.0-c0.0-n4                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 4     │ C        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 2.0898986472023857 │ 32173b                          │ 0            │ True  │
│ 43            │ 0            │ C_f0.0-c0.0-n5                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 5     │ C        │ 0.0       │ not_measured                                                                                 │ 138.0605161190033  │ ok           │ not_measured       │ replayed-measurement-3b2090     │ 0            │ True  │
│ 43            │ 0            │ C_f0.0-c0.0-n5                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 5     │ C        │ 0.0       │ not_measured                                                                                 │ 150.9471504688263  │ ok           │ not_measured       │ replayed-measurement-79a9ea     │ 0            │ True  │
│ 43            │ 0            │ C_f0.0-c0.0-n5                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 5     │ C        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.9175071683194902 │ 84cb77                          │ 0            │ True  │
│ 43            │ 0            │ C_f0.0-c0.0-n5                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 5     │ C        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 2.0964882009559207 │ e100e0                          │ 0            │ True  │
│ 44            │ 0            │ A_f1.0-c1.0-n5                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 5     │ A        │ 1.0       │ not_measured                                                                                 │ 105.63729166984558 │ ok           │ not_measured       │ replayed-measurement-dd25e8     │ 0            │ True  │
│ 44            │ 0            │ A_f1.0-c1.0-n5                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 5     │ A        │ 1.0       │ not_measured                                                                                 │ 96.8471610546112   │ ok           │ not_measured       │ replayed-measurement-fa0ac8     │ 0            │ True  │
│ 44            │ 0            │ A_f1.0-c1.0-n5                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 5     │ A        │ 1.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 2.934369213051266  │ ceceee                          │ 0            │ True  │
│ 44            │ 0            │ A_f1.0-c1.0-n5                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 5     │ A        │ 1.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 2.6901989181836448 │ 0e55b3                          │ 0            │ True  │
│ 45            │ 0            │ A_f1.0-c1.0-n2                              │ replay.benchmark_performance                │ 1.0        │ multi-cloud-ml                 │ 2     │ A        │ 1.0       │ not_measured                                                                                 │ 291.90445613861084 │ ok           │ not_measured       │ replayed-measurement-5174e4     │ 0            │ True  │
│ 45            │ 0            │ A_f1.0-c1.0-n2                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 1.0        │ multi-cloud-ml                 │ 2     │ A        │ 1.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 3.2433828459845646 │ 20a6d4                          │ 0            │ True  │
│ 46            │ 0            │ A_f0.0-c0.0-n4                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 4     │ A        │ 0.0       │ not_measured                                                                                 │ 145.12948369979858 │ ok           │ not_measured       │ replayed-measurement-75802b     │ 0            │ True  │
│ 46            │ 0            │ A_f0.0-c0.0-n4                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 4     │ A        │ 0.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.612549818886651  │ a70990                          │ 0            │ True  │
│ 47            │ 0            │ C_f0.0-c1.0-n4                              │ replay.benchmark_performance                │ 0.0        │ multi-cloud-ml                 │ 4     │ C        │ 1.0       │ not_measured                                                                                 │ 121.42492485046388 │ ok           │ not_measured       │ replayed-measurement-229377     │ 0            │ True  │
│ 47            │ 0            │ C_f0.0-c1.0-n4                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ multi-cloud-ml                 │ 4     │ C        │ 1.0       │ not_measured                                                                                 │ not_measured       │ not_measured │ 1.349165831671821  │ 3f8809                          │ 0            │ True  │
└───────────────┴──────────────┴─────────────────────────────────────────────┴─────────────────────────────────────────────┴────────────┴────────────────────────────────┴───────┴──────────┴───────────┴──────────────────────────────────────────────────────────────────────────────────────────────┴────────────────────┴──────────────┴────────────────────┴─────────────────────────────────┴──────────────┴───────┘
```

<!-- markdownlint-enable line-length -->

## Explore Further

- _Perform an optimization instead of a random walk_: See the
  [search a space with an optimizer example](/ado/examples/best-configuration-search).
- _Modify the objective function_: Try modifying the cost function and creating
  a new space - be careful to change the name of the experiment!
- _Create a custom experiment_: Explore
  [the documentation for writing your own custom experiment](/ado/actuators/creating-custom-experiments/)
- _Break the discoveryspace_: See what happens if you try to create the
  `discoveryspace` without the experiment that provides input to the cost
  function.
- _Examine the requests_: Run `ado show trace operation` to see what is
  replayed (`benchmark_performance`) and what is calculated
  (`ml-multicloud-cost`)

## Key Takeaways

- **Dependent experiments**: `ado` allows you to define experiments which
  consume the output of other experiments.
  - There is no limit to the depth of the chain of dependent experiments.
  - Dependent experiments are executed when the required inputs are available.
- **Custom experiments**: You can add your own Python functions as experiments
  using `ado`'s custom experiments feature.
- **Uniform usage pattern**: How you use `ado` to define spaces or perform
  operations does not change if you use custom or dependent experiments.
