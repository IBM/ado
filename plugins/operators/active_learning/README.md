# ADO Active Learning Operators

The `pkh` (Predictive Kernel Herding) operator selects entities from a finite
ADO discovery space within a fixed budget. It fits a surrogate model and
predicts the remaining target values.

The discovery space must define exactly one experiment. It can return several
outputs. `targetOutput` selects the numeric output to model, and PKH reuses its
existing measurements.

PKH selects entities to match the pool in a learned Random Forest leaf space.
This objective measures representativeness under the learned kernel. It does
not guarantee predictive accuracy.

## Installation

Use a current `ado` checkout. Install the core and plugin from its root:

```bash
pip install -e . -e plugins/operators/active_learning
```

For the optional AutoGluon predictor on Python 3.11–3.13, install the extra:

```bash
pip install -e "plugins/operators/active_learning[autogluon]"
```

See the [configuration guide](../../../docs/user-guide/operators/pkh.md)
and [ideal-gas tutorial](../../../docs/user-guide/examples/pkh.md).
