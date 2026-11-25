# ADO Active Learning Operators

The `pkh` (Predictive Kernel Herding) operator selects entities from a finite
ADO discovery space within a fixed budget. It fits a surrogate model and
predicts the remaining target values.

The discovery space must define exactly one experiment. It can return several
outputs. `targetOutput` selects the numeric output to model, and PKH reuses its
existing measurements.

PKH compares the fraction of the sample in each Random Forest leaf with the
fraction of the full pool in that leaf. It favors entities in leaves where the
sample fraction is smaller.

## Installation

Use a current `ado` checkout. Install the core and plugin from its root:

```bash
pip install -e . -e plugins/operators/active_learning
```

On Python 3.11–3.13, the base install includes AutoGluon for both
`random_forest` and `autogluon`. To include CatBoost, LightGBM, and XGBoost
in AutoGluon model selection, install the extra:

```bash
pip install -e "plugins/operators/active_learning[autogluon]"
```

See the [configuration guide](../../../docs/user-guide/operators/pkh.md)
and [ideal-gas tutorial](../../../docs/user-guide/examples/pkh.md).
