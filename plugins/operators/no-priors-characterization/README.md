# ADO No-Priors Characterization Operator

`ado-no-priors-characterization` is an operator plugin for the
[Accelerated Discovery Orchestrator (ADO)](https://github.com/IBM/ado),
providing initial exploration of discovery spaces using high-dimensional
sampling strategies.

**No-Priors Characterization** is designed for unbiased exploration when no
measured data exists yet, establishing an initial dataset for subsequent
model-based exploration.

## How it Works

The `No-Priors Characterization` operator uses different sampling strategies
to ensure good coverage of the discovery space:

- **`random`**: Random sampling across the space for unbiased exploration.
  This provides the baseline sampling approach.
- **`clhs`** (Concatenated Latin Hypercube Sampling): Ensures uniform coverage
  by enforcing stratification in each dimension independently. Each dimension
  cycles through all possible values before repeating.
- **`sobol`**: Sobol sequence sampling for quasi-random low-discrepancy coverage

The operator retrieves already-measured entities from the discovery space,
orders the unmeasured entities using the specified sampling strategy,
and yields entities in batches
for measurement.

## Installation

You can install the `No-Priors Characterization` operator and its dependencies
(including `ado-core`) directly from PyPI:

```bash
pip install ado-no-priors-characterization
```

## More Information

To learn more about No-Priors Characterization and explore the full
capabilities of ADO, including detailed documentation, configuration guides, and
additional examples, visit the official ADO website:

- **No-Priors Quickstart**: <https://ibm.github.io/ado/examples/no-priors-characterization/>
- **Configuring No-Priors**: <https://ibm.github.io/ado/operators/no-priors-characterization/>
- **ADO Documentation**: <https://ibm.github.io/ado/>
