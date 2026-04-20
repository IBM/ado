# Performing Efficient Space-Filling Sampling of a Configuration Space

<!-- markdownlint-disable no-blanks-blockquote -->

> [!NOTE] The scenario
>
> You have an experiment with multiple parameters and need an initial measured
> dataset that covers the configuration space efficiently.
> **In this example, `ado`'s `random_walk` operator with the no-priors sampler
> is used for efficient space-filling sampling of the target property across the
> parameter space, moving beyond standard random-walk or brute-force sampling.**
> Using the no-priors sampler with `random_walk` involves:
>
> 1. Defining the configuration space to explore.
> 2. Creating an `operation` that uses `random_walk` with the no-priors sampler
>    to order and submit points with a space-filling strategy.
> 3. Observing the measurement process as the selected strategy orders and
>    submits the points.

> [!IMPORTANT] Prerequisites
>
> Get the example files and install dependencies:
>
> ```commandline
> git clone https://github.com/IBM/ado.git
> cd ado
> pip install examples/no-priors-characterization/custom_experiments/
> ```

> [!CAUTION]
>
> All commands below assume you are running them from the
> **top-level of the `ado` repository**.

> [!TIP] TL;DR
>
> To create a `discoveryspace` and perform efficient space-filling sampling with
> the `random_walk` operator using the no-priors sampler, execute the following
> from the root of the `ado` repository:
>
> ```bash
> : # Create the space to explore based on a custom experiment
> ado create space -f \
>   examples/no-priors-characterization/example_yamls/space_reaction.yaml \
>   --new-sample-store
> : # Run a space-filling characterization operation
> ado create operation -f \
>   examples/no-priors-characterization/example_yamls/op_basic_sampling.yaml \
>     --use-latest space
> ```

<!-- markdownlint-enable no-blanks-blockquote -->

## What is Space-Filling Sampling with the No-Priors Sampler?

The **no-priors sampler** is an advanced sampler for the `random_walk` operator
that provides efficient space-filling exploration when you do not yet have a
useful prior model or historical dataset. It is a strong fit for the first phase
of an exploration, where you want representative coverage across a configuration
space before switching to model-based or target-driven workflows.

**Handling Existing Measurements**: If the discovery space already contains
measured entities for the target property, the sampler automatically:

- Identifies which entities have already been measured
- Excludes them from sampling, so that the operator will measure the
  desired amount of new entities

The sampler supports multiple sampling strategies:

1. **Random Sampling (`random`)**: A baseline random ordering across the
   candidate configuration space. Fast and simple, but usually less
   space-filling than the advanced strategies.

2. **Concatenated Latin Hypercube Sampling (`clhs`)**: An adaptation of Latin
   Hypercube Sampling for discrete spaces. It improves dimension-wise coverage
   by reducing repeated reuse of the same values early in the sampling process.
   This is often a strong default for high-dimensional spaces.

3. **Sobol Sampling (`sobol`)**: A quasi-random low-discrepancy sampling
   method that provides stronger space-filling properties than pure random
   sampling. It is adapted for discrete parameter spaces and falls back to CLHS
   when collisions are detected during discretization.

4. **One-Shift Sampling (`one_shift`)**: A heuristic for higher-dimensional
   spaces that attempts to maximize minimum distance between samples.

5. **Recursive Aggregation (`recursive_aggregation`)**: Another heuristic for
   higher-dimensional spaces with different coverage characteristics.

<!-- markdownlint-disable no-blanks-blockquote -->
> [!CAUTION]
>
> In the current version, if not all measurements produce the observed target
> output property specified in the sampler's `targetOutput` parameter, the
> operation may fail or produce incomplete results. Ensure all experiments
> return the expected target property.

<!-- markdownlint-enable no-blanks-blockquote -->

The sampler orders a specified number of new points, which `random_walk` then
measures in batches using the configured experiment, storing the results in the
sample store.

## Creating a `discoveryspace`

A `discoveryspace` describes the configuration space you want to explore
(`entitySpace`) and how to measure it (`measurementSpace`). In this example,
we use two custom Python functions as experiments and take inspiration from the
chemistry domain:

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

## Running a Space-Filling Sampling Operation

Next, we run an `operation` that uses `random_walk` with the no-priors sampler
to perform space-filling sampling of the `discoveryspace`. We provide three
example configurations with different strategies:

### Space-Filling Sampling with CLHS

The configuration for a CLHS-based space-filling operation is in
`op_basic_sampling.yaml`:

<!-- prettier-ignore-start -->

```yaml
{%
  include-markdown "./example_yamls/op_basic_sampling.yaml"
%}
```
<!-- prettier-ignore-end -->

This configuration uses the no-priors sampler with CLHS to prioritize early
coverage across the configuration space rather than relying on plain random
ordering.

<!-- markdownlint-disable line-length -->

```commandline
ado create operation -f \
  examples/no-priors-characterization/example_yamls/op_basic_sampling.yaml \
  --use-latest space
```

<!-- markdownlint-enable line-length -->

### Baseline Random Sampling

For a baseline comparison using random sampling with 20 samples and batch size
of 5:

```commandline
ado create operation -f \
  examples/no-priors-characterization/example_yamls/op_quick_exploration.yaml \
  --use-latest space
```

**Note**: Each operation samples different points from the space based on its
strategy and parameters, even when using the same discovery space.

Random sampling is useful as a baseline, but CLHS and Sobol generally provide
better space-filling behavior for initial characterization.

### Detailed Coverage with Sobol Sequence

For denser low-discrepancy coverage using Sobol sequences with 100 samples and
batch size of 1:

```commandline
ado create operation -f \
  examples/no-priors-characterization/example_yamls/op_thorough_coverage.yaml \
  --use-latest space
```

This is a good option when you want more uniform low-discrepancy coverage of
the available configuration space.

### What to Expect in the Terminal

You will see output as the `random_walk` operator with the no-priors sampler
orders, submits, and measures points. The key stages are:

#### Initialization

The operator will log the start of the sampling process:

<!-- markdownlint-disable line-length -->

```commandline
2026-03-09 16:30:00,000 INFO      MainThread           RandomWalk: Running random walk for 30 iterations. Sampler is custom sampler class: ...
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
(RandomWalk pid=82843) Continuous Batching: EXPERIMENT COMPLETION. Received finished notification for experiment in measurement request in group 1: request-c72090-experiment-calculate_reaction_yield-entities-temperature.353-concentration.4.1-catalyst_amount.4.5 (random_walk)-requester-randomwalk-1.6.1.dev9+03a65e7b.dirty-9a277d-time-2026-03-10 11:43:11.066810+00:00
```

<!-- markdownlint-enable line-length -->

#### Completion

The operation will end with a success message:

<!-- markdownlint-disable line-length -->

```commandline
Success! Created operation with identifier operation-random_walk-v0.1-8b23a245 and it finished successfully.
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
│ 0     │ temperature.300-concentration.1.0-catalyst_amount.2.0    │ random_walk                │ custom_experiments.calculate_reaction_yield │ 300         │ 1.0           │ 2.0             │ 45.23    │
│ 1     │ temperature.350-concentration.2.5-catalyst_amount.5.0    │ random_walk                │ custom_experiments.calculate_reaction_yield │ 350         │ 2.5           │ 5.0             │ 78.91    │
│ 2     │ temperature.400-concentration.0.5-catalyst_amount.1.0    │ random_walk                │ custom_experiments.calculate_reaction_yield │ 400         │ 0.5           │ 1.0             │ 92.15    │
│ ...   │ ...                                                      │ ...                        │ ...                                         │ ...         │ ...           │ ...             │ ...      │
└───────┴──────────────────────────────────────────────────────────┴────────────────────────────┴─────────────────────────────────────────────┴─────────────┴───────────────┴─────────────────┴──────────┘
```

<!-- markdownlint-enable line-length -->

## Comparison with Other Sampling Approaches

### When to Use the No-Priors Sampler

Use the no-priors sampler with `random_walk` when you want to:

- Build an initial measured dataset before surrogate modelling or optimization
- Cover a discrete or discretized configuration space more efficiently than
  plain random sampling
- Avoid repeatedly measuring entities that already have the target output
- Get better space-filling coverage than the base `random_walk` samplers

### Comparison with Base Random Walk Samplers

The base `random_walk` samplers (`random`, `sequential`, grouped modes) are
simpler and appropriate when:

- You want to iterate through existing entities in the sample store
- You need deterministic sequential traversal of a finite space
- You don't need optimized space-filling properties

The no-priors sampler adds:

- Active reordering of candidates using dedicated space-filling strategies
- Automatic exclusion of already-measured entities for a target output
- Multiple strategy options (CLHS, Sobol, etc.) for different coverage needs

### Comparison with LHC and Ray Tune

For continuous optimization or hyperparameter tuning, consider:

- **Latin Hypercube Sampling (LHC)** via ray-tune: Better for continuous spaces
  and when you want to leverage Ray's distributed execution
- **Ray Tune operators**: Appropriate for model hyperparameter optimization with
  adaptive search algorithms (e.g., Bayesian optimization, HyperBand)

The no-priors sampler is specifically designed for:

- Discrete or discretized configuration spaces
- Initial characterization before optimization
- Cases where you want space-filling coverage without a surrogate model

## Takeaways

- **Efficient space-filling**: The no-priors sampler helps cover a configuration
  space more effectively than plain random ordering.
- **Multiple strategies**: Choose from random, Sobol, CLHS, or higher-dimensional
  heuristics depending on the trade-off you want between baseline simplicity and
  coverage quality.
- **Flexible configuration**: Adjust the number of samples and batch size to
  balance throughput, coverage, and experimental resources.
- **Foundation for later workflows**: The resulting dataset is well suited for
  surrogate modelling, optimization, or follow-on operators such as TRIM.
- **Integrated with random_walk**: The sampler works within the standard
  `random_walk` operator flow, benefiting from its batching, filtering, and
  memoization capabilities.
