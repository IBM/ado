# ADO Active-Learning Operators

`ado-active-learning` is an operator plugin for the
[Accelerated Discovery Orchestrator (ADO)](https://github.com/IBM/ado),
providing regression-based active-learning operators for finite Discovery
Spaces. It currently provides **PKH**.

PKH is designed for scenarios where measuring an entity's target property is
slow or costly. Instead of measuring entities in a fixed or random order, it
trains a Random Forest regressor on the labels collected so far and uses it
to choose, one entity at a time, which unmeasured entity to measure next.

## How it Works

PKH picks an entity, waits for ADO to record its target value, then uses
that new label - together with every prior label - to choose the next
entity.

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
mirrors the full entity pool.

## Installation

```bash
uv pip install -e plugins/operators/active_learning
```

Confirm that ADO discovered the operator:

```bash
ado get operators
```

The output should include `pkh`.

## Examples

[`examples/operation_pkh.yaml`](examples/operation_pkh.yaml) is a complete
operation file that runs against
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
