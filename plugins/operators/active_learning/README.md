# ADO Active-Learning Operators

`ado-active-learning` is an operator plugin for the
[Accelerated Discovery Orchestrator (ADO)](https://github.com/IBM/ado),
providing two regression-based active-learning operators for finite Discovery
Spaces: **PKH** and **FLORA**.

Both are designed for scenarios where measuring an entity's target property is
slow or costly. Instead of measuring entities in a fixed or random order, each
operator trains a Random Forest regressor on the labels collected so far and
uses it to choose, one entity at a time, which unmeasured entity to measure
next.

## How it Works

Both operators run the same loop: pick an entity, wait for ADO to record its
target value, then use that new label - together with every prior label - to
choose the next entity. They differ in how they use the forest to make that
choice.

### PKH (Predictive Kernel Herding)

PKH treats forest leaves as a coarse summary of the entity pool's
distribution. After fitting, it builds two histograms per tree: how much of
the *full pool* falls in each leaf, and how much of the *labelled sample*
falls in each leaf. At each step, PKH selects the entity sitting in the leaf
with the largest deficit - where the pool is most over-represented relative to
what has been labelled so far, averaged across trees. This pulls the labelled
sample's distribution toward the full pool's distribution, the same
discrepancy-minimization idea behind kernel herding. The forest is refit every
`epochLength` selections, and leaf counts are updated cheaply in between.
PKH's objective is representativeness, not predictive accuracy: it is the
right choice when what you need is a labelled sample whose distribution
mirrors the full entity pool. When the goal is instead a model with the
lowest possible predictive error across the entire entity pool,
use FLORA, whose acquisition targets predictive risk directly.

### FLORA

FLORA uses the forest to track predictive disagreement instead of population
coverage. For every pool entity it computes the variance across the individual
trees' predictions — a proxy for how uncertain the forest is there — and
averages that disagreement within each leaf. At each step, FLORA selects the
entity whose leaf maximizes the expected reduction in predictive risk from one
more label, a score that grows with a leaf's disagreement and pool share and
shrinks the more that leaf is already labelled. The forest is refit on a
near-geometric schedule, so refits become less frequent as the labelled
sample grows.
FLORA's objective is predictive accuracy, not distributional coverage: it is
the right choice when the goal is a low-error predictive model. When the
goal is instead a labelled sample that represents the full entity pool's
distribution, use PKH, whose acquisition targets that discrepancy directly.

## Installation

```bash
uv pip install -e plugins/operators/active_learning
```

Confirm that ADO discovered both operators:

```bash
ado get operators
```

The output should include `pkh` and `flora`.

## Examples

[`examples/operation_pkh.yaml`](examples/operation_pkh.yaml) and
[`examples/operation_flora.yaml`](examples/operation_flora.yaml) are complete
operation files that run against
[`examples/discoveryspace.yaml`](examples/discoveryspace.yaml), a 100-entity
space built on ADO's `calculate_density` custom experiment:

```bash
uv pip install -e examples/density_example
uv pip install -e plugins/operators/active_learning
ado create space \
  -f plugins/operators/active_learning/examples/discoveryspace.yaml \
  --new-sample-store
ado create operation \
  -f plugins/operators/active_learning/examples/operation_pkh.yaml \
  --use-latest space
```

Swap in `operation_flora.yaml` to try FLORA instead.
