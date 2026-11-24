# ADO Active Learning Operators

The `pkh` (Predictive Kernel Herding) and `flora` operators select entities from
a finite ADO discovery space within a fixed budget. They fit a surrogate model
and predict the remaining target values.

The discovery space must define exactly one experiment. It can return several
outputs. Set `targetOutput` to the numeric output to model. Both operators
reuse existing measurements of that output.

PKH compares the fraction of the sample in each Random Forest leaf with the
fraction of the full pool in that leaf. It favors entities in leaves where the
sample fraction is smaller. FLORA selects entities using the trees'
disagreement about the target and the number of measurements in each leaf.

## Installation

Use a current `ado` checkout. Install the core and plugin from its root:

```bash
pip install -e . -e plugins/operators/active_learning
```

For the optional AutoGluon predictor on Python 3.11–3.13, install the extra:

```bash
pip install -e "plugins/operators/active_learning[autogluon]"
```

When upgrading from 0.1.0, update any explicit PKH
`operation.module.operatorVersion` pin to `0.2.0` to run it with this release.
Omit the field to let `ado` resolve the installed operator version.

See the [PKH configuration guide](../../../docs/user-guide/operators/pkh.md),
the [FLORA guide](../../../docs/user-guide/operators/flora.md), the
[ideal-gas tutorial](../../../docs/user-guide/examples/pkh.md), and the
[titration tutorial](../../../docs/user-guide/examples/flora.md).
