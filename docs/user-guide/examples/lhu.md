<!-- markdownlint-disable code-block-style -->

# Identify the important dimensions of a space

!!! abstract "Intermediate tutorial"

    This walkthrough assumes you are already familiar with the core `ado`
    workflow — discovery spaces, operations, and sample stores. If you are new
    to `ado`, work through
    [Your first ado experiment](tutorials/density-example.md) first.

When working with a high-dimensional configuration space it is natural to ask
**which dimensions have the greatest influence on a specific experimental
outcome**. For instance, if a workload has 20 tunable parameters you might want
to identify which ones most significantly affect throughput. Understanding this
can help narrow future explorations to only the most impactful parameters,
**reducing the time and resources spent by ignoring those that are less
relevant**.

The workloads in the `ml_multi_cloud` dataset are defined by four parameters —
`provider`, `cpu_family`, `vcpu_size`, and `nodes` — giving the entity space
four dimensions. This walkthrough finds **which of those dimensions most
influence `wallClockRuntime`**.

We will:

1. Install the `ray_tune` operator
2. Reuse the discovery space from the [random walk example](tutorials/random-walk.md)
3. Run a Latin Hypercube sampling operation with an `InformationGain` stopper
4. Interpret the ranked dimension output

!!! warning "Prerequisites"

    - `ado-core` installed (`pip install ado-core`)
    - The repository cloned from GitHub:

      ```bash
      git clone https://github.com/ibm/ado.git
      cd ado
      ```

    - The discovery space and sample store from the
      [random walk example](tutorials/random-walk.md) already created.

    All commands in this walkthrough are run from the **repository root**
    (`ado/`).

!!! example "TL;DR"

    Once the packages are installed and the discovery space exists:

    <!-- markdownlint-disable MD013 -->
    ```bash
    ado create operation -f examples/ml-multi-cloud/lhc_sampler.yaml --set "spaces[0]=$DISCOVERY_SPACE_IDENTIFIER"
    ```
    <!-- markdownlint-enable MD013 -->

## Step 1 — Install the `ray_tune` operator

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

## Step 2 — Reuse the discovery space

This example uses the same discovery space created in the
[random walk example](tutorials/random-walk.md). Retrieve its identifier:

```bash
ado get spaces
```

Note the identifier — you will pass it to the operation in the next step.

## Step 3 — Run a Latin Hypercube sampling operation

The file `examples/ml-multi-cloud/lhc_sampler.yaml` configures `ray_tune` to
sample using the
[Latin Hypercube Sampler](../operators/ray-tune.md#latin-hypercube-sampler) and
stop automatically when the most influential dimensions are known:

<!-- markdownlint-disable MD013 -->
```bash
ado create operation -f examples/ml-multi-cloud/lhc_sampler.yaml --set "spaces[0]=$DISCOVERY_SPACE_IDENTIFIER"
```
<!-- markdownlint-enable MD013 -->

The operation is configured to:

- Sample up to **32 points** using the Latin Hypercube Sampler — a method that
  maintains properties similar to true random sampling while keeping samples
  more evenly spread across the space.
- Monitor the mutual information between each dimension and the target metric
  (`wallClockRuntime`).
- Stop early via the
  [InformationGain stopper](../operators/ray-tune.md#informationgainstopper)
  once the most influential dimensions are identified.

RayTune logs progress to the terminal as it runs. Because the `InformationGain`
stopper triggers before all 32 samples are collected, the operation finishes
early.

## Step 4 — Interpret the results

Within the logs for the last sample you will see output like:

```text
(tune pid=72306) Stopping criteria reached after 14 samples.
(tune pid=72306) Total search space size is 48, search coverage is 0.2916666666666667.
(tune pid=72306) Entropy of target variable clusters: 0.6517565611726529 nats.
(tune pid=72306) Result:
(tune pid=72306)     dimension  rank        mi  uncertainty%
(tune pid=72306) 3       nodes     1  0.491089      0.753486
(tune pid=72306) 1  cpu_family     2  0.352622      0.541033
(tune pid=72306) 0    provider     3  0.034638      0.053146
(tune pid=72306) 2   vcpu_size     4  0.000929      0.001425
(tune pid=72306)
(tune pid=72306) Pareto selection:['cpu_family', 'nodes']
```

### Reading the ranked dimension table

The dimensions are ranked by **mutual information** (`mi`) with the target
variable `wallClockRuntime`. The columns are:

| Column | Meaning |
| :----- | :------ |
| `rank` | Importance rank (1 = most influential) |
| `mi` | Mutual information with the target metric (higher = more influential) |
| `uncertainty%` | The dimension's `mi` as a fraction of the target variable's entropy — how much of the variance in `wallClockRuntime` is explained by this dimension |

In this run `nodes` contributes ~75% of the explainable variance, making it by
far the most important dimension, while `vcpu_size` is nearly irrelevant.

### The Pareto selection

At the end of the output the stopper reports a **Pareto selection** — the
smallest set of dimensions whose combined mutual information exceeds a
threshold (0.8 by default).

The stopper computes this as follows:

- For each possible set size, it finds the subset that explains the most mutual
  information. For example, for size 2 it evaluates all 6 pairs:
  `[nodes, vcpu_size]`, `[nodes, provider]`, `[nodes, cpu_family]`, etc.
- This produces one optimal set per size (1, 2, 3, 4 in this case) — the
  Pareto-optimal sets.
- The smallest set whose total mutual information exceeds the threshold is
  selected.

!!! note

    Since Latin Hypercube sampling is random, the Pareto set can vary slightly
    from run to run. Over multiple runs you should see the Pareto set contain
    2 or 3 dimensions and always include `nodes`.

## Summary

| Step | What you did | `ado` concept |
| :--- | :----------- | :------------ |
| 1 | Installed the `ray_tune` operator | Operator |
| 2 | Located the existing discovery space | Discovery space |
| 3 | Ran a Latin Hypercube sampling operation with early stopping | Operation / `ray_tune` operator |
| 4 | Ranked dimensions by mutual information and read the Pareto selection | `InformationGain` stopper |

## What's next

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable no-inline-html -->
<!-- markdownlint-disable MD046 -->
<!-- prettier-ignore-start -->

<div class="grid cards" markdown>

- :octicons-rocket-24:{ .lg .middle } **Search using an optimizer**

    ---

    Use `ray_tune` to drive a Bayesian optimizer over a continuous space and
    locate the best configuration.

    [Search a space with an optimizer :octicons-arrow-right-24:](best-configuration-search.md)

- :octicons-rocket-24:{ .lg .middle } **Creating custom objective functions**

    ---

    Define a custom experiment that uses outputs from another experiment as its
    inputs — a common pattern for derived metrics.

    [Search a space based on a custom objective function :octicons-arrow-right-24:](search-custom-objective.md)

</div>

<!-- prettier-ignore-end -->
<!-- markdownlint-enable MD046 -->
<!-- markdownlint-enable no-inline-html -->
<!-- markdownlint-enable line-length -->
