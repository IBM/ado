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
ado show entities space --use-latest space
```

<!-- markdownlint-enable line-length -->

This will display a table of the entities sampled and their measured pressure
values.

<!-- markdownlint-disable line-length -->

```text
                             identifier generatorid                                    experiment_id  temperature  volume  mol     pressure
0   temperature.270.0-volume.5.0-mol.0.2         unk  custom_experiments.calculate_pressure_ideal_gas        270.0     5.0  0.2    89.796196
1   temperature.296.0-volume.8.0-mol.0.6         unk  custom_experiments.calculate_pressure_ideal_gas        296.0     8.0  0.6   184.581070
2   temperature.274.0-volume.9.0-mol.0.9         unk  custom_experiments.calculate_pressure_ideal_gas        274.0     9.0  0.9   227.816276
3   temperature.272.0-volume.4.0-mol.0.7         unk  custom_experiments.calculate_pressure_ideal_gas        272.0     4.0  0.7   395.768421
4   temperature.292.0-volume.3.0-mol.0.4         unk  custom_experiments.calculate_pressure_ideal_gas        292.0     3.0  0.4   323.709745
5   temperature.276.0-volume.2.0-mol.0.3         unk  custom_experiments.calculate_pressure_ideal_gas        276.0     2.0  0.3   344.218752
6   temperature.288.0-volume.7.0-mol.0.5         unk  custom_experiments.calculate_pressure_ideal_gas        288.0     7.0  0.5   171.040374
7   temperature.284.0-volume.1.0-mol.0.1         unk  custom_experiments.calculate_pressure_ideal_gas        284.0     1.0  0.1   236.130738
8   temperature.286.0-volume.6.0-mol.0.8         unk  custom_experiments.calculate_pressure_ideal_gas        286.0     6.0  0.8   317.058174
9   temperature.278.0-volume.1.0-mol.0.6         unk  custom_experiments.calculate_pressure_ideal_gas        278.0     1.0  0.6  1386.852365
10  temperature.294.0-volume.3.0-mol.0.1         unk  custom_experiments.calculate_pressure_ideal_gas        294.0     3.0  0.1    81.481734
11  temperature.280.0-volume.8.0-mol.0.3         unk  custom_experiments.calculate_pressure_ideal_gas        280.0     8.0  0.3    87.301857
...
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
