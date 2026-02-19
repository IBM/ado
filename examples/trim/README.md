# Efficiently Exploring Parameter Spaces with TRIM

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
> pip install plugins/operators/trim/
> pip install -e examples/trim/custom_experiments/
> ```

> [!CAUTION]
>
> All commands below assume you are running them from the
> **top-level of the `ado` repository**.

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
designed to efficiently build a surrogate model of a system. It's perfect for
situations where measuring points in your parameter space is costly.

It works in two main phases:

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

```yaml
{% 
  include-markdown "./example_yamls/op_pressure.yaml" 
%}
```

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
2026-01-16 14:56:57,589 WARNING   MainThread           trim.utils.space_df_connector: get_df_at_least_one_measured_value: No measured properties found in the discovery space
...
2026-01-16 14:56:57,656 WARNING   MainThread           trim.operator  : trim                : Only 0 points in the source space.
Starting with no-prior characterization operation, it will sample 20 points.
```

<!-- markdownlint-enable line-length -->

It then runs a simple sampling operation (in this case, using Concatenated Latin
Hypercube Sampling or `clhs`) to gather the initial data points. You will see
output for each point being measured:

<!-- markdownlint-disable line-length -->

```commandline
(RandomWalk pid=10734) Continuous batching: SUBMIT EXPERIMENT. Submitted experiment custom_experiments.calculate_pressure_ideal_gas for temperature.270.0-volume.5.0-mol.0.2. Request identifier: 3201d2
(RandomWalk pid=10734)
(RandomWalk pid=10734) Continuous batching: SUMMARY. Entities sampled and submitted: 1. Experiments completed: 0 Waiting on 1 active requests. There are 0 dependent experiments
(RandomWalk pid=10734) Continuous Batching: EXPERIMENT COMPLETION. Received finished notification for experiment...
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
(RandomWalk pid=10736) 2026-01-16 14:57:19,256 INFO      AsyncIO Thread: default trim.trim_sampler: iterator            : Fitting AutoGluon TabularPredictor, iteration 5...
...
(RandomWalk pid=10736) 2026-01-16 14:57:20,723 INFO      AsyncIO Thread: default trim.trim_sampler: iterator            : [Batch under consideration: 5] Training metric: root_mean_squared_error;
(RandomWalk pid=10736) Best model: NeuralNetTorch; score_val: -8.49; holdout_score: -669.00
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
Success! Created operation with identifier operation-trim-v0.1-8b23a245 and it finished successfully.
```

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

This will show the `discoveryspace` and the sub-operations that were run.
To see the entities of the space that have been measured, you can run:

<!-- markdownlint-disable line-length -->

```commandline
ado show entities space --use-latest
```

<!-- markdownlint-enable line-length -->

This will display a table of the entities sampled and their measured pressure
values.

<!-- markdownlint-disable line-length -->

```text
┌───────┬──────────────────────────────────┬─────────────┬─────────────────────────────────────────────────┬─────────────┬────────┬─────┬────────────────────┐
│ INDEX │ identifier                       │ generatorid │ experiment_id                                   │ temperature │ volume │ mol │ pressure           │
├───────┼──────────────────────────────────┼─────────────┼─────────────────────────────────────────────────┼─────────────┼────────┼─────┼────────────────────┤
│ 0     │ temperature.270-volume.1-mol.0.1 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 270         │ 1      │ 0.1 │ 224.49049068600002 │
│ 1     │ temperature.270-volume.2-mol.0.1 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 270         │ 2      │ 0.1 │ 112.24524534300001 │
│ 2     │ temperature.270-volume.8-mol.0.2 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 270         │ 8      │ 0.2 │ 56.122622671500004 │
│ 3     │ temperature.272-volume.8-mol.0.9 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 272         │ 8      │ 0.9 │ 254.4225561108     │
│ 4     │ temperature.274-volume.1-mol.0.2 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 274         │ 1      │ 0.2 │ 455.6325514664     │
│ 5     │ temperature.274-volume.4-mol.0.9 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 274         │ 4      │ 0.9 │ 512.5866203997     │
│ 6     │ temperature.274-volume.8-mol.0.5 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 274         │ 8      │ 0.5 │ 142.38517233325    │
│ 7     │ temperature.276-volume.3-mol.0.8 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 276         │ 3      │ 0.8 │ 611.9444486848     │
│ 8     │ temperature.276-volume.7-mol.0.4 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 276         │ 7      │ 0.4 │ 131.1309532896     │
│ 9     │ temperature.278-volume.4-mol.0.6 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 278         │ 4      │ 0.6 │ 346.7130911706     │
│ 10    │ temperature.278-volume.7-mol.0.2 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 278         │ 7      │ 0.2 │ 66.04058879440001  │
│ 11    │ temperature.280-volume.2-mol.0.8 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 280         │ 2      │ 0.8 │ 931.219813216      │
│ 12    │ temperature.280-volume.6-mol.0.9 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 280         │ 6      │ 0.9 │ 349.207429956      │
│ 13    │ temperature.280-volume.9-mol.0.3 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 280         │ 9      │ 0.3 │ 77.60165110133333  │
│ 14    │ temperature.282-volume.1-mol.0.7 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 282         │ 1      │ 0.7 │ 1641.2749207932    │
│ 15    │ temperature.282-volume.6-mol.0.5 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 282         │ 6      │ 0.5 │ 195.389871523      │
│ 16    │ temperature.282-volume.6-mol.0.7 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 282         │ 6      │ 0.7 │ 273.5458201322     │
│ 17    │ temperature.282-volume.7-mol.0.6 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 282         │ 7      │ 0.6 │ 200.97243928080002 │
│ 18    │ temperature.284-volume.3-mol.0.4 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 284         │ 3      │ 0.4 │ 314.8409844682667  │
│ 19    │ temperature.284-volume.7-mol.0.9 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 284         │ 7      │ 0.9 │ 303.5966635944     │
│ 20    │ temperature.286-volume.4-mol.0.1 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 286         │ 4      │ 0.1 │ 59.448407718700004 │
│ 21    │ temperature.286-volume.8-mol.0.3 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 286         │ 8      │ 0.3 │ 89.17261157805001  │
│ 22    │ temperature.286-volume.9-mol.0.7 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 286         │ 9      │ 0.7 │ 184.9506017915111  │
│ 23    │ temperature.288-volume.2-mol.0.3 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 288         │ 2      │ 0.3 │ 359.1847850976     │
│ 24    │ temperature.288-volume.7-mol.0.8 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 288         │ 7      │ 0.8 │ 273.6645981696     │
│ 25    │ temperature.290-volume.4-mol.0.7 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 290         │ 4      │ 0.7 │ 421.9589778635     │
│ 26    │ temperature.290-volume.5-mol.0.3 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 290         │ 5      │ 0.3 │ 144.67164955319998 │
│ 27    │ temperature.292-volume.8-mol.0.6 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 292         │ 8      │ 0.6 │ 182.0867313342     │
│ 28    │ temperature.292-volume.9-mol.0.4 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 292         │ 9      │ 0.4 │ 107.90324819804444 │
│ 29    │ temperature.294-volume.1-mol.0.4 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 294         │ 1      │ 0.4 │ 977.7808038768001  │
│ 30    │ temperature.294-volume.3-mol.0.5 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 294         │ 3      │ 0.5 │ 407.408668282      │
│ 31    │ temperature.294-volume.5-mol.0.5 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 294         │ 5      │ 0.5 │ 244.44520096920002 │
│ 32    │ temperature.296-volume.3-mol.0.1 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 296         │ 3      │ 0.1 │ 82.03603116426667  │
│ 33    │ temperature.296-volume.4-mol.0.3 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 296         │ 4      │ 0.3 │ 184.5810701196     │
│ 34    │ temperature.296-volume.5-mol.0.1 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 296         │ 5      │ 0.1 │ 49.22161869856     │
│ 35    │ temperature.296-volume.5-mol.0.7 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 296         │ 5      │ 0.7 │ 344.55133088992    │
│ 36    │ temperature.296-volume.9-mol.0.1 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 296         │ 9      │ 0.1 │ 27.345343721422225 │
│ 37    │ temperature.298-volume.2-mol.0.5 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 298         │ 2      │ 0.5 │ 619.427465041      │
│ 38    │ temperature.298-volume.5-mol.0.9 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 298         │ 5      │ 0.9 │ 445.98777482952    │
│ 39    │ temperature.298-volume.6-mol.0.2 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 298         │ 6      │ 0.2 │ 82.59032867213334  │
│ 40    │ temperature.298-volume.9-mol.0.9 │ unk         │ custom_experiments.calculate_pressure_ideal_gas │ 298         │ 9      │ 0.9 │ 247.7709860164     │
└───────┴──────────────────────────────────┴─────────────┴─────────────────────────────────────────────────┴─────────────┴────────┴─────┴────────────────────┘
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
