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
[taking a random walk example](random-walk.md). When the workload
configuration space is explored using a random walk, both the `wallClockRuntime`
and the `cost`, as defined by the custom function, will be measured.

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
[these instructions](random-walk.md#using-pre-existing-data-with-ado).
If you have already completed the
[taking a random walk example](random-walk.md), reuse the
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

## Exploring the `discoveryspace`

To run a `randomwalk` operation on the new space, execute:

```commandline
ado create operation -f randomwalk_ml_multicloud_operation.yaml --use-latest space
```

This produces an output similar to that described in the
[taking a random walk example](random-walk.md#exploring-the-discoveryspace)
and will exit printing the operation identifier. However, in this case there is
additional information related to the dependent experiment.

When it completes, you can get a table of the points visited with:

```commandline
ado show measurements operation --use-latest
```

You will see a table similar to the following - note the extra column for the
new cost function (rows truncated for readability):

<!-- markdownlint-disable line-length -->

```commandline
┌───────────────┬──────────────┬─────────────────────────────────────────────┬─────────────────────────────────────────────┬────────────┬───────┬──────────┬───────────┬────────────────────┬──────────────┬────────────────────┬───────┐
│ request_index │ result_index │ identifier                                  │ experiment_id                               │ cpu_family │ nodes │ provider │ vcpu_size │ wallClockRuntime   │ status       │ total_cost         │ valid │
├───────────────┼──────────────┼─────────────────────────────────────────────┼─────────────────────────────────────────────┼────────────┼───────┼──────────┼───────────┼────────────────────┼──────────────┼────────────────────┼───────┤
│ 1             │ 0            │ B_f0.0-c0.0-n3                              │ replay.benchmark_performance                │ 0.0        │ 3     │ B        │ 0.0       │ 153.51639366149902 │ ok           │ not_measured       │ True  │
│ 1             │ 0            │ B_f0.0-c0.0-n3                              │ custom_experiments.ml-multicloud-cost@1.0.0 │ 0.0        │ 3     │ B        │ 0.0       │ not_measured       │ not_measured │ 1.2793032805124918 │ True  │
└───────────────┴──────────────┴─────────────────────────────────────────────┴─────────────────────────────────────────────┴────────────┴───────┴──────────┴───────────┴────────────────────┴──────────────┴────────────────────┴───────┘
```

<!-- markdownlint-enable line-length -->

## What's next

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable no-inline-html -->
<!-- markdownlint-disable MD046 -->
<!-- prettier-ignore-start -->

<div class="grid cards" markdown>

- :octicons-rocket-24:{ .lg .middle } **Search using an optimizer**

    ---

    Try the [Search a space with an optimizer](best-configuration-search.md) example to see how you can use RayTune in combination with custom experiments, via `ado`.

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
