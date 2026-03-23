# Exploring Parameter Spaces with No-Priors Characterization

<!-- markdownlint-disable no-blanks-blockquote -->

> [!NOTE] The scenario
>
> You have an experiment with multiple parameters,
> and you want to understand how these parameters influence the outcome.
> **In this example, `ado`'s no-priors characterization operator is used to
> systematically sample and measure the target property across the parameter
> space using various sampling strategies aimed at covering uniformly the
> parameter space.** Using the no-priors characterization
> operator involves:
>
> 1. Defining the parameter space to explore.
> 2. Creating an `operation` that uses no-priors characterization to sample
>    points using a chosen strategy.
> 3. Observing the sampling process as it measures the target output property with
>    the selected strategy.

> [!IMPORTANT] Prerequisites
>
> Get the example files and install dependencies:
>
> ```commandline
> git clone https://github.com/IBM/ado.git
> cd ado
> pip install plugins/operators/no-priors-characterization/
> pip install examples/no-priors-characterization/custom_experiments/
> ```

> [!CAUTION]
>
> All commands below assume you are running them from the
> **top-level of the `ado` repository**.

> [!TIP] TL;DR
>
> To create a `discoveryspace` and explore it with the no-priors
> characterization operator, execute the following from the root of the `ado`
> repository:
>
> ```bash
> : # Create the space to explore based on a custom experiment
> ado create space -f \
>   examples/no-priors-characterization/example_yamls/space_reaction.yaml \
>   --new-sample-store
> : # Explore it with no-priors characterization!
> ado create operation -f \
>   examples/no-priors-characterization/example_yamls/op_basic_sampling.yaml \
>     --use-latest space
> ```

<!-- markdownlint-enable no-blanks-blockquote -->

## What is No-Priors Characterization?

**No-Priors Characterization** is a sampling operator designed to explore a
parameter space systematically without requiring any prior knowledge or
existing data. It's perfect for initial exploration of a system where you want
to gather representative samples across the entire parameter space.

**Handling Existing Measurements**: If the discovery space already contains
measured entities for the target property, the operator automatically:

- Identifies which entities have already been measured
- Excludes them from sampling, so that the operator will measure the
  desired amount of entities

The operator supports three sampling strategies:

1. **Random Sampling (`random`)**: Uniformly random sampling across the
   parameter space. Fast and simple, but may not provide optimal coverage.

2. **Concatenated Latin Hypercube Sampling (`clhs`)**: An adaptation of Latin
   Hypercube Sampling for discrete spaces. Good coverage in each dimension is
   obtained by avoiding measuring parameters combinations with many common
   values. Particularly effective for high-dimensional spaces.

3. **Sobol Sampling (`sobol`)**: A quasi-random low-discrepancy sampling
   method that provides better space-filling properties than pure random
   sampling. It has been adapted for discrete parameter spaces. It falls back
   to Concatenated Latin Hypercube Sampling when collisions are detected
   during the discretization process.

<!-- markdownlint-disable no-blanks-blockquote -->
> [!CAUTION]
>
> In the current version of no-priors characterization, if not all
> measurements produce the observed target output property specified in the
> `operation.parameters.targetOutput` field, the operation may fail or produce
> incomplete results. Ensure all experiments return the expected target property.

<!-- markdownlint-enable no-blanks-blockquote -->

The operator samples a specified number of points in batches, measures them
using the configured experiment, and stores the results in the sample store.

## Creating a `discoveryspace`

A `discoveryspace` describes the parameters you want to explore (`entitySpace`)
and how to measure them (`measurementSpace`). In this example, we'll use two
custom Python functions as experiments and take inspiration from the Chemistry domain:

1. **`calculate_reaction_yield`**: Calculates chemical reaction yield based on
   temperature (K), concentration (mol/L), and catalyst amount (g) using an
   Arrhenius-like equation.

2. **`calculate_material_strength`**: Calculates material tensile strength (MPa)
   based on composition percentages, temperature (°C), and grain size (μm) using
   a Hall-Petch relationship.

First, create the `discoveryspace` by executing this command from the repository
root:

```commandline
ado create space -f \
  examples/no-priors-characterization/example_yamls/space_reaction.yaml \
  --new-sample-store
```

This will create a new space and a sample store to hold the measurement results.
The output will be similar to:

```terminaloutput
Success! Created space with identifier: space-bfed2d-19b49a
```

## Exploring with a No-Priors Characterization Operation

Next, we will run an `operation` that uses no-priors characterization to
explore the `discoveryspace`. We provide three example configurations with
different sampling strategies:

### Basic Sampling with CLHS

The configuration for a basic sampling operation using CLHS is in
`op_basic_sampling.yaml`:

<!-- prettier-ignore-start -->

```yaml
{%
  include-markdown "./example_yamls/op_basic_sampling.yaml"
%}
```
<!-- prettier-ignore-end -->

To run the operation, execute:

<!-- markdownlint-disable line-length -->

```commandline
ado create operation -f \
  examples/no-priors-characterization/example_yamls/op_basic_sampling.yaml \
  --use-latest space
```

<!-- markdownlint-enable line-length -->

### Exploration with Random Sampling

For an exploration with random sampling (uses random sampling with 20 samples
and batch size of 5 for quick initial exploration):

```commandline
ado create operation -f \
  examples/no-priors-characterization/example_yamls/op_quick_exploration.yaml \
  --use-latest space
```

**Note**: Each operation samples different points from the space based on its
strategy and parameters, even when using the same discovery space.

### Thorough Coverage with Sobol Sequence

For comprehensive coverage using Sobol sequences (uses Sobol sampling with 100
samples and batch size of 1 for detailed parameter space coverage):

```commandline
ado create operation -f \
  examples/no-priors-characterization/example_yamls/op_thorough_coverage.yaml \
  --use-latest space
```

### What to Expect in the Terminal

You will see output as the no-priors characterization operator samples and
measures points. The key stages are:

#### Initialization

The operator will log the start of the sampling process:

<!-- markdownlint-disable line-length -->

```commandline
2026-03-09 16:30:00,000 INFO      MainThread           no_priors_characterization.operator: Starting no-priors characterization with 30 samples using clhs strategy
```

<!-- markdownlint-enable line-length -->

#### Sampling and Measurement

For each batch of points, you will see output indicating the experiments being
submitted and completed:

<!-- markdownlint-disable line-length -->

```commandline
(RandomWalk pid=82843) Continuous batching: SUBMIT EXPERIMENT. Submitted experiment custom_experiments.calculate_reaction_yield for temperature.353-concentration.4.1-catalyst_amount.4.5. Request identifier: c72090
(RandomWalk pid=82843)
(RandomWalk pid=82843) Continuous batching: SUMMARY. Entities sampled and submitted: 2. Experiments completed: 1 Waiting on 1 active requests. There are 0 dependent experiments
(RandomWalk pid=82843) Continuous Batching: EXPERIMENT COMPLETION. Received finished notification for experiment in measurement request in group 1: request-c72090-experiment-calculate_reaction_yield-entities-temperature.353-concentration.4.1-catalyst_amount.4.5 (no_priors_characterization)-requester-randomwalk-1.6.1.dev9+03a65e7b.dirty-9a277d-time-2026-03-10 11:43:11.066810+00:00
```

<!-- markdownlint-enable line-length -->

#### Completion

The operation will end with a success message:

<!-- markdownlint-disable line-length -->

```commandline
Success! Created operation with identifier operation-no-priors-characterization-v0.1-8b23a245 and it finished successfully.
```

<!-- markdownlint-enable line-length -->

## Looking at the `operation` output

After the operation completes, you can view the sampled entities and their
measured values.

You can see the relationship between the space and operations with:

```commandline
ado show related space --use-latest
```

This will show the `discoveryspace` and the operations that were run.
To see the entities of the space that have been measured, you can run:

<!-- markdownlint-disable line-length -->

```commandline
ado show entities space --use-latest
```

<!-- markdownlint-enable line-length -->

This will display a table of the entities sampled and their measured reaction
yield values.

<!-- markdownlint-disable line-length -->

```text
┌───────┬──────────────────────────────────────────────────────────┬────────────────────────────┬─────────────────────────────────────────────┬─────────────┬───────────────┬─────────────────┬──────────┐
│ INDEX │ identifier                                               │ generatorid                │ experiment_id                               │ temperature │ concentration │ catalyst_amount │ yield    │
├───────┼──────────────────────────────────────────────────────────┼────────────────────────────┼─────────────────────────────────────────────┼─────────────┼───────────────┼─────────────────┼──────────┤
│ 0     │ temperature.300-concentration.1.0-catalyst_amount.2.0    │ no_priors_characterization │ custom_experiments.calculate_reaction_yield │ 300         │ 1.0           │ 2.0             │ 45.23    │
│ 1     │ temperature.350-concentration.2.5-catalyst_amount.5.0    │ no_priors_characterization │ custom_experiments.calculate_reaction_yield │ 350         │ 2.5           │ 5.0             │ 78.91    │
│ 2     │ temperature.400-concentration.0.5-catalyst_amount.1.0    │ no_priors_characterization │ custom_experiments.calculate_reaction_yield │ 400         │ 0.5           │ 1.0             │ 92.15    │
│ ...   │ ...                                                      │ ...                        │ ...                                         │ ...         │ ...           │ ...             │ ...      │
└───────┴──────────────────────────────────────────────────────────┴────────────────────────────┴─────────────────────────────────────────────┴─────────────┴───────────────┴─────────────────┴──────────┘
```

<!-- markdownlint-enable line-length -->

## Takeaways

- **Systematic Exploration**: The no-priors characterization operator provides
  systematic sampling of parameter spaces without requiring prior knowledge.
- **Multiple Strategies**: Choose from random, Sobol, or CLHS sampling based on
  your needs for speed vs. coverage quality.
- **Flexible Configuration**: Adjust the number of samples and batch size to
  balance thoroughness with computational resources.
- **Foundation for Further Analysis**: The sampled data can serve as a
  foundation for building surrogate models or for use with other operators like
  TRIM.
