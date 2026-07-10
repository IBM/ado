# Quickly building a predictive model for a configuration space

<!-- markdownlint-disable no-blanks-blockquote -->

> [!NOTE] The scenario
>
> You have a complex system with many tunable parameters, like a scientific
> simulation or a machine learning model, which is time-consuming and expensive
> to run. **In this example, `ado`'s TRIM operator is used to intelligently
> explore the parameter space of an experiment, measuring just enough samples to
> build a stable and accurate predictive model.** Using the TRIM operator
> involves:
>
> 1. Defining the parameter space to explore in a `discoveryspace`.
> 2. Creating an `operation` that uses TRIM to intelligently sample points,
>    measure them, and build a model.
> 3. Observing TRIM's progress as it first characterizes the space and then
>    iteratively refines its model. When the quality of this predictive model
>    does not improve, TRIM stops.

> [!IMPORTANT] Prerequisites
>
> Get the example files and install dependencies:
>
> ```commandline
> git clone https://github.com/IBM/ado.git
> cd ado
> pip install ado-trim
> pip install -e examples/trim/custom_experiments/
> ```

> [!NOTE]
>
> **Python Version Compatibility**: The TRIM operator is not available on Python
> 3.14 due to a dependency on `autogluon==1.5.0`, which requires
> `pyarrow==20.0.0` (incompatible with Python 3.14). If you need to use TRIM,
> please use Python 3.10-3.13.

> [!CAUTION]
>
> All commands below assume you are running them from the **top-level of the
> `ado` repository**.

> [!TIP] TL;DR
>
> To create a `discoveryspace` and explore it with the TRIM operator, execute
> the following from the root of the `ado` repository:
>
> ```bash
> : # Create the space to explore based on a custom experiment
> ado create space -f examples/trim/example_yamls/space_pressure.yaml --new-sample-store
> : # Explore it with TRIM!
> ado create operation -f examples/trim/example_yamls/op_pressure.yaml \
>     --use-latest space
> ```

<!-- markdownlint-enable no-blanks-blockquote -->

## What is TRIM?

**TRIM (Transfer Refined Iterative Modeling)** is a characterization operator
designed to efficiently build a surrogate model of a system. This surrogate
model is especially useful when evaluating points in the parameter space is
costly, as it enables predictions at unmeasured points instead of relying on
direct measurements. It works in two main phases:

1. **No-Priors Characterization**: If there isn't enough existing data, TRIM
   starts by sampling a small, representative set of points to get a baseline
   understanding of the space
2. **Iterative Modeling**: TRIM then enters a loop: it uses the data it has
   gathered to train a preliminary model (using `AutoGluon`), uses that model's
   intelligence to decide which point to sample next, measures that point, and
   then retrains the model. It stops automatically when it determines that
   further sampling won't significantly improve the model's accuracy, saving you
   time and resources

Finally, it trains one high-quality model on all the data it has collected and
saves it for you to use.

<!-- markdownlint-disable no-blanks-blockquote -->

> [!CAUTION]
>
> The current version of TRIM assumes that all measurements produce the observed
> target output property, here in the `operation.parameters.targetOutput` field.
> More details in
> [the relevant section](https://ibm.github.io/ado/operators/trim/#debugging-and-troubleshooting).

<!-- markdownlint-enable no-blanks-blockquote -->

## Creating a `discoveryspace`

A `discoveryspace` describes the parameters you want to explore (`entitySpace`)
and how to measure them (`measurementSpace`). In this example, we'll use a
custom Python function `calculate_pressure_ideal_gas` as our experiment.

First, create the `discoveryspace` by executing this command from the repository
root:

```commandline
ado create space -f examples/trim/example_yamls/space_pressure.yaml --new-sample-store
```

This will create a new space and a sample store to hold the measurement results.
The output will be similar to:

```terminaloutput
Success! Created space with identifier: space-bfed2d-19b49a
```

## Exploring with a TRIM Operation

Next, we will run an `operation` that uses TRIM to explore the `discoveryspace`.
The configuration for our operation is in `op_pressure.yaml`:

<!-- prettier-ignore-start -->

```yaml
{%
  include-markdown "./example_yamls/op_pressure.yaml"
%}
```

<!-- prettier-ignore-end -->

To run the operation, execute:

<!-- markdownlint-disable line-length -->

```commandline
ado create operation -f examples/trim/example_yamls/op_pressure.yaml --use-latest space
```

<!-- markdownlint-enable line-length -->

### What to Expect in the Terminal

You will see a lot of output as TRIM does its work. Let's break down the key
stages:

#### Stage 1: No-Priors Characterization

Since in our example we started with an empty sample store, TRIM cannot
immediately build a model. It will log this and begin the initial
characterization phase.

<!-- markdownlint-disable line-length -->

```commandline
2026-07-09 15:35:48,452 WARNING   MainThread           trim.samplers.no_priors_utils: get_df_at_least_one_measured_value: No measured properties found in the discovery space
Returning empty DataFrame

2026-07-09 15:35:48,470 WARNING   MainThread           trim.samplers.no_priors_utils: get_source_and_target: The source space is empty
2026-07-09 15:35:48,470 WARNING   MainThread           trim.operator  : trim                : Only 0 points in the source space.
Starting with no-prior characterization operation, it will sample 18 points.
Note: Trim sampler has been called with a minimum budget of 18 points.
```

<!-- markdownlint-enable line-length -->

It then runs a simple sampling operation (in this case, using Concatenated Latin
Hypercube Sampling or `clhs`) to gather the initial data points. You will see
output for each point being measured:

<!-- markdownlint-disable line-length -->

```commandline
(RandomWalk pid=74822) Continuous batching: SUBMIT EXPERIMENT. Submitted experiment
custom_experiments.calculate_pressure_ideal_gas for mol.0.7-temperature.272-volume.1. Request identifier:
4f70cf
(RandomWalk pid=74822)
(RandomWalk pid=74822) Continuous batching: SUMMARY. Entities sampled and submitted: 2. Experiments
completed: 1 Waiting on 1 active requests. There are 0 dependent experiments
(RandomWalk pid=74822) Continuous Batching: EXPERIMENT COMPLETION. Received finished notification for
experiment in measurement request in group 1:
request-4f70cf-experiment-calculate_pressure_ideal_gas-entities-mol.0.7-temperature.272-volume.1
(no_priors_characterization)-requester-random_walk@2.0.0-3378d5-time-2026-07-09 15:35:52.207903+01:00
```

<!-- markdownlint-enable line-length -->

#### Stage 2: Iterative Modeling

Once the initial characterization is complete, TRIM begins its main iterative
loop. In each iteration, it samples a new point, trains an `AutoGluon` model and
checks if the model's accuracy is still improving. The points to sample are
chosen by leveraging the information obtained in the no-prior characterization
stage.

You'll see logs indicating that a model is being trained and evaluated:

<!-- markdownlint-disable line-length -->

```commandline
(RandomWalk pid=76621) AutoGluon training complete, total runtime = 1.49s ... Best model:
WeightedEnsemble_L2 | Estimated inference throughput: 508.6 rows/s (8 batch size)
```

<!-- markdownlint-enable line-length -->

After a set number of iterations (defined by `iterationSize`), it will check the
stopping criterion:

<!-- markdownlint-disable line-length -->

```commandline
(RandomWalk pid=10736) 2026-01-16 14:57:48,947 INFO      AsyncIO Thread: default trim.trim_sampler: iterator            : Testing stopping criterion after measuring 14 points, mean_ratio={mean_ratio} and std_ratio={std_ratio}
(RandomWalk pid=10736) 2026-01-16 14:57:48,947 INFO      AsyncIO Thread: default trim.trim_sampler: iterator            : Stopping not triggered for i=14
```

<!-- markdownlint-enable line-length -->

#### Stage 3: Stopping and Finalizing

The iterative process continues until the model's performance stabilizes. At
that point, the stopping criterion is met, and TRIM will train one final model
on all the data it has gathered.

<!-- markdownlint-disable line-length -->

```commandline
(RandomWalk pid=10736) 2026-01-16 14:58:06,441 INFO      AsyncIO Thread: default trim.trim_sampler: iterator            : Stopping criteria hit after measuring 22 entities.
...
(RandomWalk pid=10736) 2026-01-16 14:58:06,468 INFO      AsyncIO Thread: default trim.trim_sampler: finalize_model      : Finalizing the predictive model:Fitting AutoGluon TabularPredictor on full Source Space data of 42 rows.Model will be saved in: trim_models_finalized
...
(RandomWalk pid=10736) Final model root_mean_squared_error=-48.72586662062896.Saving predicted model to: trim_models_finalized.
```

<!-- markdownlint-enable line-length -->

The operation will end with a success message:

<!-- markdownlint-disable line-length -->

```commandline
Success! Created operation with identifier operation-trim@2.0.3-cb3448b3 and it finished successfully.```

<!-- markdownlint-enable line-length -->

## Looking at the `operation` output

The TRIM operator saves the final trained `AutoGluon` model to the directory
specified by the `outputDirectory` field in your operation parameters. The model
can be then loaded as a `TabularPredictor` in your own code to make predictions
on any unmeasured points in your parameter space.

You can also view the entities that were sampled during the entire operation.
TRIM actually runs two sub-operations (one for characterization, one for
iterative modeling). You can see the relationship with:

```commandline
ado show related space --use-latest
```

This will show the `discoveryspace` and the sub-operations that were run. To see
the entities of the space that have been measured, you can run:

<!-- markdownlint-disable line-length -->

```commandline
ado show measurements space --use-latest
```

<!-- markdownlint-enable line-length -->

This will display a table of the entities sampled and their measured pressure
values.

<!-- markdownlint-disable line-length -->

```text
┌───────┬────────────────┬────────────────┬────────────────┬─────────────┬────────┬─────┬───────────────┐
│ INDEX │ identifier     │ generatorid    │ experiment_id  │ temperature │ volume │ mol │ pressure      │
├───────┼────────────────┼────────────────┼────────────────┼─────────────┼────────┼─────┼───────────────┤
│ 0     │ mol.0.1-tempe… │ no_priors_cha… │ custom_experi… │ 270         │ 1      │ 0.1 │ 224.49049068… │
│ 1     │ mol.0.1-tempe… │ no_priors_cha… │ custom_experi… │ 274         │ 2      │ 0.1 │ 113.90813786… │
│ 2     │ mol.0.1-tempe… │ no_priors_cha… │ custom_experi… │ 280         │ 3      │ 0.1 │ 77.601651101… │
│ 3     │ mol.0.1-tempe… │ no_priors_cha… │ custom_experi… │ 280         │ 4      │ 0.1 │ 58.201238326  │
│ 4     │ mol.0.1-tempe… │ no_priors_cha… │ custom_experi… │ 284         │ 6      │ 0.1 │ 39.355123058… │
│ 5     │ mol.0.1-tempe… │ no_priors_cha… │ custom_experi… │ 286         │ 9      │ 0.1 │ 26.421514541… │
│ 6     │ mol.0.1-tempe… │ no_priors_cha… │ custom_experi… │ 288         │ 7      │ 0.1 │ 34.2080747712 │
│ 7     │ mol.0.1-tempe… │ no_priors_cha… │ custom_experi… │ 292         │ 5      │ 0.1 │ 48.556461689… │
│ 8     │ mol.0.2-tempe… │ no_priors_cha… │ custom_experi… │ 276         │ 2      │ 0.2 │ 229.47916825… │
│ 9     │ mol.0.2-tempe… │ no_priors_cha… │ custom_experi… │ 276         │ 8      │ 0.2 │ 57.3697920642 │
│ 10    │ mol.0.2-tempe… │ no_priors_cha… │ custom_experi… │ 278         │ 1      │ 0.2 │ 462.28412156… │
│ 11    │ mol.0.2-tempe… │ no_priors_cha… │ custom_experi… │ 278         │ 3      │ 0.2 │ 154.09470718… │
│ 12    │ mol.0.2-tempe… │ no_priors_cha… │ custom_experi… │ 278         │ 6      │ 0.2 │ 77.047353593… │
│ 13    │ mol.0.2-tempe… │ no_priors_cha… │ custom_experi… │ 290         │ 4      │ 0.2 │ 120.559707961 │
│ 14    │ mol.0.2-tempe… │ no_priors_cha… │ custom_experi… │ 292         │ 9      │ 0.2 │ 53.951624099… │
│ 15    │ mol.0.2-tempe… │ no_priors_cha… │ custom_experi… │ 294         │ 7      │ 0.2 │ 69.8414859912 │
│ 16    │ mol.0.3-tempe… │ no_priors_cha… │ custom_experi… │ 276         │ 3      │ 0.3 │ 229.47916825… │
│ 17    │ mol.0.3-tempe… │ no_priors_cha… │ custom_experi… │ 278         │ 2      │ 0.3 │ 346.71309117… │
│ 18    │ mol.0.3-tempe… │ no_priors_cha… │ custom_experi… │ 278         │ 5      │ 0.3 │ 138.68523646… │
│ 19    │ mol.0.3-tempe… │ no_priors_cha… │ custom_experi… │ 286         │ 1      │ 0.3 │ 713.38089262… │
│ 20    │ mol.0.3-tempe… │ no_priors_cha… │ custom_experi… │ 288         │ 8      │ 0.3 │ 89.7961962744 │
│ 21    │ mol.0.3-tempe… │ no_priors_cha… │ custom_experi… │ 294         │ 6      │ 0.3 │ 122.22260048… │
│ 22    │ mol.0.3-tempe… │ no_priors_cha… │ custom_experi… │ 298         │ 9      │ 0.3 │ 82.590328672… │
│ 23    │ mol.0.4-tempe… │ no_priors_cha… │ custom_experi… │ 270         │ 4      │ 0.4 │ 224.49049068… │
│ 24    │ mol.0.4-tempe… │ no_priors_cha… │ custom_experi… │ 272         │ 7      │ 0.4 │ 129.23050469… │
│ 25    │ mol.0.4-tempe… │ no_priors_cha… │ custom_experi… │ 276         │ 3      │ 0.4 │ 305.97222434… │
│ 26    │ mol.0.4-tempe… │ no_priors_cha… │ custom_experi… │ 278         │ 2      │ 0.4 │ 462.28412156… │
│ 27    │ mol.0.4-tempe… │ no_priors_cha… │ custom_experi… │ 282         │ 1      │ 0.4 │ 937.87138331… │
│ 28    │ mol.0.4-tempe… │ no_priors_cha… │ custom_experi… │ 284         │ 8      │ 0.4 │ 118.06536917… │
│ 29    │ mol.0.4-tempe… │ no_priors_cha… │ custom_experi… │ 294         │ 5      │ 0.4 │ 195.55616077… │
│ 30    │ mol.0.5-tempe… │ no_priors_cha… │ custom_experi… │ 274         │ 3      │ 0.5 │ 379.69379288… │
│ 31    │ mol.0.5-tempe… │ no_priors_cha… │ custom_experi… │ 274         │ 9      │ 0.5 │ 126.56459762… │
│ 32    │ mol.0.5-tempe… │ no_priors_cha… │ custom_experi… │ 278         │ 1      │ 0.5 │ 1155.7103039… │
│ 33    │ mol.0.5-tempe… │ no_priors_cha… │ custom_experi… │ 278         │ 4      │ 0.5 │ 288.92757597… │
│ 34    │ mol.0.5-tempe… │ no_priors_cha… │ custom_experi… │ 278         │ 6      │ 0.5 │ 192.61838398… │
│ 35    │ mol.0.5-tempe… │ no_priors_cha… │ custom_experi… │ 278         │ 7      │ 0.5 │ 165.101471986 │
│ 36    │ mol.0.5-tempe… │ no_priors_cha… │ custom_experi… │ 280         │ 2      │ 0.5 │ 582.01238326  │
│ 37    │ mol.0.5-tempe… │ no_priors_cha… │ custom_experi… │ 290         │ 5      │ 0.5 │ 241.119415922 │
│ 38    │ mol.0.5-tempe… │ no_priors_cha… │ custom_experi… │ 292         │ 8      │ 0.5 │ 151.73894277… │
│ 39    │ mol.0.6-tempe… │ no_priors_cha… │ custom_experi… │ 272         │ 3      │ 0.6 │ 452.30676641… │
│ 40    │ mol.0.6-tempe… │ no_priors_cha… │ custom_experi… │ 274         │ 1      │ 0.6 │ 1366.8976543… │
│ 41    │ mol.0.6-tempe… │ no_priors_cha… │ custom_experi… │ 280         │ 9      │ 0.6 │ 155.20330220… │
│ 42    │ mol.0.6-tempe… │ no_priors_cha… │ custom_experi… │ 282         │ 2      │ 0.6 │ 703.40353748… │
│ 43    │ mol.0.6-tempe… │ no_priors_cha… │ custom_experi… │ 284         │ 2      │ 0.6 │ 708.39221505… │
│ 44    │ mol.0.6-tempe… │ no_priors_cha… │ custom_experi… │ 284         │ 5      │ 0.6 │ 283.35688602… │
│ 45    │ mol.0.6-tempe… │ no_priors_cha… │ custom_experi… │ 286         │ 7      │ 0.6 │ 203.82311217… │
│ 46    │ mol.0.6-tempe… │ no_priors_cha… │ custom_experi… │ 290         │ 4      │ 0.6 │ 361.679123883 │
│ 47    │ mol.0.6-tempe… │ no_priors_cha… │ custom_experi… │ 292         │ 6      │ 0.6 │ 242.78230844… │
│ 48    │ mol.0.6-tempe… │ no_priors_cha… │ custom_experi… │ 298         │ 8      │ 0.6 │ 185.82823951… │
│ 49    │ mol.0.7-tempe… │ no_priors_cha… │ custom_experi… │ 270         │ 3      │ 0.7 │ 523.811144934 │
│ 50    │ mol.0.7-tempe… │ no_priors_cha… │ custom_experi… │ 272         │ 1      │ 0.7 │ 1583.0736824… │
│ 51    │ mol.0.7-tempe… │ no_priors_cha… │ custom_experi… │ 284         │ 2      │ 0.7 │ 826.45758422… │
│ 52    │ mol.0.7-tempe… │ no_priors_cha… │ custom_experi… │ 286         │ 9      │ 0.7 │ 184.95060179… │
│ 53    │ mol.0.7-tempe… │ no_priors_cha… │ custom_experi… │ 292         │ 7      │ 0.7 │ 242.78230844… │
│ 54    │ mol.0.7-tempe… │ no_priors_cha… │ custom_experi… │ 298         │ 3      │ 0.7 │ 578.13230070… │
│ 55    │ mol.0.8-tempe… │ no_priors_cha… │ custom_experi… │ 270         │ 4      │ 0.8 │ 448.98098137… │
│ 56    │ mol.0.8-tempe… │ no_priors_cha… │ custom_experi… │ 272         │ 1      │ 0.8 │ 1809.2270656… │
│ 57    │ mol.0.8-tempe… │ no_priors_cha… │ custom_experi… │ 272         │ 5      │ 0.8 │ 361.84541313… │
│ 58    │ mol.0.8-tempe… │ no_priors_cha… │ custom_experi… │ 274         │ 8      │ 0.8 │ 227.81627573… │
│ 59    │ mol.0.8-tempe… │ no_priors_cha… │ custom_experi… │ 276         │ 6      │ 0.8 │ 305.97222434… │
│ 60    │ mol.0.8-tempe… │ no_priors_cha… │ custom_experi… │ 282         │ 2      │ 0.8 │ 937.87138331… │
│ 61    │ mol.0.8-tempe… │ no_priors_cha… │ custom_experi… │ 292         │ 9      │ 0.8 │ 215.80649639… │
│ 62    │ mol.0.8-tempe… │ no_priors_cha… │ custom_experi… │ 296         │ 5      │ 0.8 │ 393.77294958… │
│ 63    │ mol.0.8-tempe… │ no_priors_cha… │ custom_experi… │ 298         │ 3      │ 0.8 │ 660.72262937… │
│ 64    │ mol.0.9-tempe… │ no_priors_cha… │ custom_experi… │ 270         │ 7      │ 0.9 │ 288.630630882 │
│ 65    │ mol.0.9-tempe… │ no_priors_cha… │ custom_experi… │ 274         │ 1      │ 0.9 │ 2050.3464815… │
│ 66    │ mol.0.9-tempe… │ no_priors_cha… │ custom_experi… │ 280         │ 2      │ 0.9 │ 1047.6222898… │
│ 67    │ mol.0.9-tempe… │ no_priors_cha… │ custom_experi… │ 280         │ 8      │ 0.9 │ 261.905572467 │
│ 68    │ mol.0.9-tempe… │ no_priors_cha… │ custom_experi… │ 282         │ 4      │ 0.9 │ 527.55265311… │
│ 69    │ mol.0.9-tempe… │ no_priors_cha… │ custom_experi… │ 282         │ 6      │ 0.9 │ 351.70176874… │
│ 70    │ mol.0.9-tempe… │ no_priors_cha… │ custom_experi… │ 290         │ 5      │ 0.9 │ 434.01494865… │
│ 71    │ mol.0.9-tempe… │ no_priors_cha… │ custom_experi… │ 296         │ 7      │ 0.9 │ 316.42469163… │
│ 72    │ mol.0.9-tempe… │ no_priors_cha… │ custom_experi… │ 298         │ 9      │ 0.9 │ 247.77098601… │
└───────┴────────────────┴────────────────┴────────────────┴─────────────┴────────┴─────┴───────────────┘
```

<!-- markdownlint-enable line-length -->

## Takeaways

- **Automated Surrogate Modeling**: The TRIM operator automates the process of
  building a surrogate model for a complex system.
- **Efficient Sampling**: By using an iterative, model-guided approach, TRIM
  avoids wasting resources on samples that provide little new information.
- **Auto-Stopping**: The stopping criterion ensures the process terminates once
  the model's quality plateaus, saving time and compute.
- **Reusable Artifacts**: The final output is a trained `AutoGluon` model that
  can be used for further analysis and prediction.
