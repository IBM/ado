# Identify the important dimensions of a space

> [!NOTE]
>
> This example shows:
>
> 1. Using the Latin Hyper-Cube Sampler from the `ray_tune` operator to explore
>    the space
>
> 2. Using a stopper to halt a `ray_tune` exploration when certain conditions
>    are met
>
> 3. Specifically, stopping the exploration when the dimensions/properties that
>    have the greatest influence on the target metric are known

## The scenario

When working with a high-dimensional configuration space, it's natural to ask
**which dimensions have the greatest influence on a specific experimental
outcome**. For instance, if a workload has 20 tunable parameters, you might want
to identify which ones most significantly affect a particular throughput metric.
Understanding this can help narrow future explorations to only the most
impactful parameters, potentially **reducing the time and resources spent by
ignoring those that are less relevant**.

The workloads in the `ml_multi_cloud` data set are defined by four parameters,
`provider`, `cpu_family`, `vcpu_size` and `nodes`, and hence the `entityspace`
in the related examples have 4 dimensions. Here we will try to find **which
dimensions have the most influence over the `wallClockRuntime` property.**

## Pre-requisites

### Install the ray_tune ado operator

If you haven't already installed the ray_tune operator, run:

```commandline
pip install ado-ray-tune
```

Then, executing

```commandline
ado get operators
```

should show an entry for `ray_tune` like below

```commandline
Available operators by type:
      OPERATOR     TYPE
0  random_walk  explore
1     ray_tune  explore
```

### Creating the `discoveryspace`

> [!CAUTION]
>
> The commands below assume you are in the directory `examples/ml-multi-cloud`
> in **the ado source repository**. See
> [the instructions for cloning the repository](/ado/getting-started/install/#__tabbed_1_3).

This example uses the same `discoveryspace` created in the
[taking a random walk example](/ado/examples/random-walk/). You can use
`ado get spaces` to find the identifier.

## Discovering what workload parameters most impact cost

We will use the
[Latin-Hyper-Cube sampler](/ado/operators/optimisation-with-ray-tune/#latin-hypercube-sampler)
to sample points. This is a sampling method which tries maintaining properties
similar to true random sampling, while ensuring the samples are more evenly
spread across the space. An example operation file is given in
[`lhc-sampler.yaml`](https://github.com/IBM/ado/blob/main/examples/ml-multi-cloud/lhc_sampler.yaml).

The operation also configures the exploration to monitor the relationship
between the four parameters and the cost metric and stop when it has determined
which are most important using the
[InformationGain stopper](/ado/operators/optimisation-with-ray-tune/#informationgainstopper).

To execute this operation run (replacing `$DISCOVERY_SPACE_IDENTIFIER` with the
identifier of the space created in the
[taking a random walk example](/ado/examples/random-walk/)):

```commandline
ado create operation -f lhc_sampler.yaml --set "spaces[0]=$DISCOVERY_SPACE_IDENTIFIER"
```

You will see a lot of RayTune-related output as it samples different entities
using the Latin-Hyper-Cube sampler. The number of samples to obtain is set to 32
in `lhc_sampler.yaml`, however, the operation will stop before reaching that due
to the InformationGain stopper. If you look back through the log of the
operation, within the logs for the last sample you will see lines like:

```commandline
(tune pid=7959) Stopping criteria reached after 13 samples.
(tune pid=7959) Total search space size is 48, search coverage is 0.2708333333333333.
(tune pid=7959) Entropy of target variable clusters: 1.2711814802605799 nats.
(tune pid=7959) Result:
(tune pid=7959)     dimension  rank        mi  uncertainty%
(tune pid=7959) 3       nodes     1  0.524715      0.412778
(tune pid=7959) 2   vcpu_size     2  0.396410      0.311844
(tune pid=7959) 0    provider     3  0.246428      0.193858
(tune pid=7959) 1  cpu_family     4  0.142884      0.112403
(tune pid=7959)
(tune pid=7959) Pareto selection:['provider', 'vcpu_size', 'nodes']
```

In this table the dimensions are ranked in order of importance, as determined by
their mutual information with the target variable, wallClockRuntime. The
`uncertainty%` is the ratio of the dimension's mutual information with the
entropy of the target variable (or clusters of the target variable to be more
exact) i.e. how much of the entropy or variance of the target variable is
explained by the dimension.

At the end of the output we can see the stopper has identified a "Pareto
selection" of three dimensions: ['provider', 'vcpu_size', 'nodes']. This is the
smallest number of dimensions, whose total mutual information exceeds a
threshold, which is `0.8` by default.

This is chosen as follows:

- For each possible dimension set size the stopper determines which set explains
  the most mutual information.
  - For example, for a set of size 2 dimensions, it evaluates the 6 possible
    pairs: [nodes,vcpu_size], [nodes, provider], [nodes, cpu_family] ,
    [vcpu_size, provider], [vcpu_size, nodes], [provider, nodes].
- This gives one set for each possible dimension set size: 1,2,3 and 4 in this
  case - the Pareto optimal sets
- Then the smallest of these sets which exceeds the threshold value is selected.

> [!NOTE]
>
> Since Latin Hypercube sampling is random, the Pareto set can change slightly
> from run to run as different entities are used. In this example over multiple
> runs you should see the Pareto set being 2 or 3 and always including `nodes`.
