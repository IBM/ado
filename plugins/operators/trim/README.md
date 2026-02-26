# ADO TRIM Operator

`ado-trim` is an operator plugin for the
[Accelerated Discovery Orchestrator (ADO)](https://github.com/IBM/ado),
providing the Transfer Refined Iterative Modeling (TRIM) characterization
operator.

**TRIM** is designed to efficiently build a surrogate model of a complex system.
It is ideal for scenarios where exploring a parameter space is time-consuming or
expensive. TRIM intelligently samples just enough points to create a stable and
accurate predictive model, saving significant time and resources.

## How it Works

The `TRIM` operator works in two main phases:

1. **No-Priors Characterization**: If the system has not been measured before,
   TRIM starts by sampling a small, representative set of points using a
   space-filling algorithm to get a baseline understanding of the parameter
   space.

2. **Iterative Modeling**: This phase begins by using all currently available
   data to train a single preliminary surrogate model. The feature importance
   from this model is used to order for all remaining unmeasured points. TRIM
   then enters a loop where it:
   - Samples the next point and adds it to the dataset.
   - Trains a model on the gathered data.
   - Evaluates the expected improvement of a model trained on a larger dataset
     by comparing the new model's performance against that of previous models.

This loop continues until the improvement is below a threshold, at which point
TRIM automatically stops. Finally, it trains one high-quality model on all
collected data and saves it for your use. It also outputs a file containing the
measured values and predictions for all points in your space.

## Installation

You can install the `TRIM` operator and its dependencies (including `ado-core`)
directly from PyPI:

```bash
pip install ado-trim
```

## More Information

To learn more about TRIM and explore the full capabilities of ADO, including
detailed documentation, configuration guides, and additional examples, visit the
official ADO website:

- **TRIM Quickstart**: <https://ibm.github.io/ado/examples/trim/>
- **Configuring TRIM**: <https://ibm.github.io/ado/operators/trim/>
- **ADO Documentation**: <https://ibm.github.io/ado/>
