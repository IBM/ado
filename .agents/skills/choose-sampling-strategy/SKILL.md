---
name: choose-sampling-strategy
description: >-
  Describes how how to configure sampling of a discoveryspace in ado: which operator
  (random_walk, ray_tune, trim), which sampler or optimizer (grid, random,
  selector, Latin hypercube, Sobol, Optuna TPE, Bayesian optimization,
  Nevergrad), which stoppers, and how to stage multiple sampling campaigns on
  one space. Use when choosing a sampling method, search or optimization
  algorithm, or design of experiments; configuring random_walk samplers and
  filters or ray_tune search algorithms and stoppers; or extending, resuming,
  or warm-starting sampling of a space.
---

# Choosing a Sampling Strategy

This skill covers choosing which strategy to use. For writing and validating the
operation YAML see
[define-experiment-campaign](../define-experiment-campaign/SKILL.md).

Operator reference:

* [random_walk](../../../docs/user-guide/operators/random-walk.md),
* [ray_tune](../../../docs/user-guide/operators/ray-tune.md),
* [trim](../../../docs/user-guide/operators/trim.md).

## Inputs to the choice

1. **Goal**: see the table below.
2. **Space**: dimension types, whether it is finite, its size, and existing
   measurements (see
   [examining-discovery-spaces](../examining-discovery-spaces/SKILL.md)).
3. **Budget**: affordable measurements, cost per measurement, concurrency.
4. **Installed operators**: `uv run ado get operators`. `ray_tune` and the
   `rifferla` modify operator come from `ado-ray-tune`; `trim` (characterize)
   and `NoPriorsSampleSelector` come from `ado-trim`.

## Goal → strategy

| Goal | Strategy | Stop with |
| --- | --- | --- |
| Measure every entity | `random_walk` generator, `sequential`, `numberEntities: all` | sampler exhausted |
| Unbiased statistics / distribution of a metric | `random_walk` generator, `random` | `numberEntities` |
| Even coverage with a small budget; which dimensions matter | `ray_tune` `lhu_sampler`, or `random_walk` + `NoPriorsSampleSelector` | `InformationGainStopper` / `samples` |
| Even coverage, continuous dimensions | `ray_tune` + Optuna `QMCSampler` | `MaxSamplesStopper` |
| Predictive model of a metric over the whole space | `trim` | model convergence (built in) |
| Find the best entity (single objective) | `ray_tune` optimizer (see [Choosing an optimizer](#choosing-an-optimizer-guidance)) | `MaxSamplesStopper` + `SimpleStopper` or `GrowthStopper` |
| Pareto front (multiple objectives) | `ray_tune` + Optuna | `MaxSamplesStopper` |
| Decide if two metrics differ (e.g. two experiment versions) | `ray_tune` `lhu_sampler` | `BayesianMetricDifferenceStopper` |
| Apply experiments to entities already in the sample store | `random_walk` selector + `filterMode: unmeasured` | sampler exhausted |
| Grow or resume a previous sample | `random_walk` + `filterMode: unmeasured` | `numberEntities` |

## Constraints

Breaking these causes a validation error or a misconfigured operation.

* `random_walk` generator, `lhu_sampler` and `NoPriorsSampleSelector` require
  every dimension to be discrete or categorical, and bounded.
* `ray_tune` optimizers can handle continuous and unbounded numerical dimensions
* A `random_walk` selector can handle spaces with continuous, unbounded, numerical
and open categorical dimensions. The entities matching the space must already
be in the samplestore.
* `numberEntities` larger than the space (generator) or than the matching
  entities (selector with `filterMode: noFilter`) raises a ValueError. Use
  `numberEntities: all`.
* `bayesopt` supports only continuous dimensions. The `ray_tune` template
  default is Optuna `TPESampler` (`multivariate: true`).
* Multi-objective (list `metric` / `mode`) requires `optuna`.
* Nevergrad optimizer arguments (`budget`, `num_workers`) go in
  `params.optimizer_kwargs`; elsewhere they are dropped.
* Replayed (memoized) entities count towards `num_samples`,
  `MaxSamplesStopper` and `numberEntities`, including points an optimizer
  suggests more than once. With TPE in small discrete spaces set
  `SimpleStopper` `stop_on_repeat: false`.
* Entities rejected by a `random_walk` filter do not count towards
  `numberEntities`; the walk stops when `numberEntities` entities pass or the
  sampler is exhausted.
* An optimizer learns only from trials in its own operation. To reuse earlier
  results, list them in `points_to_evaluate`; measured points are replayed.
* `ado-ray-tune` installs `optuna` and `nevergrad`. `nevergrad` installs
  `bayesian-optimization`, so `bayesopt` is available without a separate install.
* Optuna's`GPSampler` needs `torch` and `CmaEsSampler` needs `cmaes`. These are
  optional packages that must be installed to use these samplers.

## random_walk: selector or generator

Sampler types, modes and filters are defined in
[random-walk.md](../../../docs/user-guide/operators/random-walk.md#sampling-types).

Use a `selector` when:

* New experiments were added (in a new space on the same sample store) and
  should be applied only to entities that already exist.
* The space has continuous, unbounded, or open categorical dimensions, but
  relevant entities are already in the sample store.
* Re-measuring for noise or repeatability: `filterMode: measured` +
  `singleMeasurement: false`.
* Completing partially measured entities: `filterMode: partial`.

`filterMode: unmeasured` skips entities that already have results for this
space's experiments, including results recorded through other spaces that use
the same experiments. Skipped entities do not count towards `numberEntities`,
so combine the filter with the sampler settings:

* To extend a sample, use `samplerType: generator`, `mode: random`,
  `numberEntities: N`, and `filterMode: unmeasured`. This measures N entities
  that had no results. Without the filter, already-measured entities are drawn,
  replayed, and counted towards N.
* To resume an interrupted sweep, rerun the grid with `samplerType: generator`,
  `mode: sequential`, `numberEntities: all`, and `filterMode: unmeasured`.
  Without the filter, the walk starts over and replays everything already
  measured.

## Space-filling: `lhu_sampler` or `NoPriorsSampleSelector`

| | `ray_tune` `lhu_sampler` | `random_walk` + `NoPriorsSampleSelector` |
| --- | --- | --- |
| Strategies | Latin hypercube | `sobol`, `clhs` (concatenated LHS), `random` |
| Existing data | ignored; measured points are replayed and use budget | proposes only entities without a `targetOutput` value |
| Budget counted as | trials | entities that returned `targetOutput` (`samples`) |
| Early stopping | `InformationGainStopper`, `BayesianMetricDifferenceStopper` | none |
| Feeds into | `rifferla`; dimension-importance report | `trim` (same algorithm as its first phase) |

Prefer `NoPriorsSampleSelector` to extend coverage of a partly measured space,
when failed measurements should not consume budget, or before `trim`. Prefer
`lhu_sampler` to stop once dimension importance converges, to compare two
metrics, or before `rifferla`.

Configuration:
[randomwalk_sobol_operation.yaml](../../../examples/trim/example_yamls/randomwalk_sobol_operation.yaml)
(set `numberEntities` ≥ `samples`; the walk's `batchSize` sets concurrency)
and [lhc_sampler.yaml](../../../examples/ml-multi-cloud/lhc_sampler.yaml).

## Choosing an optimizer (guidance)

Optimizer names and parameters:
[ray-tune.md](../../../docs/user-guide/operators/ray-tune.md#available-optimizers).

Default: Optuna `TPESampler` with `multivariate: true`, which is what the
`ray_tune` template uses. Set a `seed` to reproduce a run. It handles
categorical and mixed spaces, noisy metrics and concurrent trials. Its first
`n_startup_trials` (default 10) are random; raise this for many dimensions.

Deviate from the default when:

| Situation | Use |
| --- | --- |
| Few numeric dimensions, expensive measurements, small budget, low concurrency | Optuna `GPSampler` |
| All dimensions continuous, budget large relative to dimensions | Nevergrad `CMA` |
| No clear fit, or many concurrent trials | Nevergrad `NGOpt` with `budget` and `num_workers` |
| Multiple objectives with a large budget | Optuna `NSGAIISampler` |
| Baseline for comparing strategies | Optuna `RandomSampler` (seeded) or a random `random_walk` |

Avoid `hyperopt` (superseded by Optuna TPE) and `bohb` (needs schedulers,
which ado does not support). Use `random_walk` rather than Optuna
`GridSampler` for grids.

## Multi-stage campaigns

Stages can run on the same space, or on a new space sharing the sample store,
so earlier measurements are replayed rather than re-measured.

* **Screen then optimize:** `lhu_sampler` + `InformationGainStopper` →
  `rifferla` creates a narrower space → optimizer on the new space.
* **Warm start:** random, LHS or Sobol sample first, then an optimizer with
  the best few entities in `points_to_evaluate`.

To compare strategies on one space, compare by trial index (the operation
timeseries, `uv run ado show measurements operation`), not wall-clock or cost:
later operations replay earlier measurements. Repeat stochastic strategies
with several seeds, or give each strategy its own sample store.

## YAML examples

Validated by `tests/operators/test_choose_sampling_strategy_yaml_examples.py`.

* [random-walk-grid.yaml](yaml-examples/random-walk-grid.yaml): full sweep,
  resumable
* [random-walk-extend-unmeasured.yaml](yaml-examples/random-walk-extend-unmeasured.yaml):
  add N new random entities
* [random-walk-selector-new-experiment.yaml](yaml-examples/random-walk-selector-new-experiment.yaml):
  measure existing entities with new experiments
* [ray-tune-optuna-tpe.yaml](yaml-examples/ray-tune-optuna-tpe.yaml): TPE
  with stoppers
* [ray-tune-optuna-qmc.yaml](yaml-examples/ray-tune-optuna-qmc.yaml): Sobol
  sample of a continuous space
* [ray-tune-nevergrad-ngopt.yaml](yaml-examples/ray-tune-nevergrad-ngopt.yaml):
  Nevergrad NGOpt with budget and workers
* [operation_optuna_multi.yaml](../../../plugins/actuators/vllm_performance/yamls/operation_optuna_multi.yaml):
  multi-objective Optuna
* [examples/trim/example_yamls/](../../../examples/trim/example_yamls/): TRIM
  and `NoPriorsSampleSelector`
