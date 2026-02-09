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
<!-- markdownlint-disable-next-line no-inline-html -->

> [!IMPORTANT] Prerequisites
>
> - Get the example files and install dependencies:
>
> ```commandline
> git clone https://github.com/IBM/ado.git
> cd ado
> pip install plugins/operators/trim
> pip install -e examples/trim
> ```
>
> - All commands below assume you are running them from the
> **top-level of the `ado` repository**.
<!-- markdownlint-disable-next-line no-inline-html -->

> [!TIP] TL;DR
> To create a `discoveryspace` and explore it with the `trim` operator,
> execute the following from the root of the `ado` repository:
>
> ```bash
> : # Create the space to explore based on a custom experiment
> ado create space -f examples/trim/configs/space_pressure.yaml --new-sample-store
> : # Explore it with TRIM!
> ado create operation -f examples/trim/configs/op_pressure.yaml --use-latest space
> ```

## What is TRIM?

**TRIM (Transfer Refined Iterative Modeling)**
is a characterization operator designed to efficiently build a surrogate
 model of a system.
It's perfect for situations where measuring points in your parameter space is costly.

It works in two main phases:

1. **No-Priors Characterization**:
    If there isn't enough existing data, `trim` starts by sampling a small,
     representative set of points to get a baseline understanding of the space.
2. **Iterative Modeling**:
   `trim` then enters a loop: it uses the data
   it has gathered to train a preliminary model (using `AutoGluon`),
    uses that model's intelligence to decide which point to sample next,
     measures that point, and then retrains the model.
   It stops automatically when it determines that
   further sampling won't significantly
   improve the model's accuracy, saving you time and resources.

Finally, it trains one high-quality model on all the data it has collected
and saves it for you to use.

## Creating a `discoveryspace`

A `discoveryspace` describes the parameters you want to
 explore (`entitySpace`) and how to measure them (`measurementSpace`).
In this example, we'll use a custom Python function
 `calculate_pressure_ideal_gas` as our experiment.

First, create the `discoveryspace` by executing this command
 from the repository root:

```commandline
ado create space -f examples/trim/configs/space_pressure.yaml --new-sample-store
```

This will create a new space and a sample store to hold the measurement results.
It will confirm with:

```commandline
Success! Created space with identifier: space-bfed2d-19b49a
```

## Exploring with a `trim` Operation

Next, we will run an `operation` that uses `trim` to explore the `discoveryspace`.
The configuration for our operation is in `op_pressure.yaml`:

```yaml
# op_pressure.yaml
operation:
  module:
    operationType: characterize
    operatorName: trim
  parameters:
    targetOutput: pressure
    batchSize: 1
    iterationSize: 5
    outputDirectory: trim_models
    stoppingCriterion:
      enabled: true
      meanThreshold: 0.9
      stdThreshold: 0.75
    autoGluonArgs:
      fitArgs:
        time_limit: 20
        presets: medium
    noPriorsParameters:
      targetOutput: pressure
```

To run the operation, execute:

<!-- markdownlint-disable line-length -->
```commandline
ado create operation -f examples/trim/configs/op_pressure.yaml --use-latest space
```
<!-- markdownlint-enable line-length -->

### What to Expect in the Terminal

You will see a lot of output as `trim` does its work.
Let's break down the key stages,
in the case of no point present in the discovery space at the beginning of the operation:

#### Stage 1: No-Priors Characterization

Since we started with an empty sample store,
`trim` first sees that it doesn't have enough data.
It will log this and begin the initial characterization phase.

<!-- markdownlint-disable line-length -->

```commandline
2026-01-16 14:56:57,589 WARNING   MainThread           trim.utils.space_df_connector: get_df_at_least_one_measured_value: No measured properties found in the discovery space
...
2026-01-16 14:56:57,656 WARNING   MainThread           trim.operator  : trim                : Only 0 points in the source space.
Starting with no-prior characterization operation, it will sample 20 points.
```
<!-- markdownlint-enable line-length -->

It then runs a simple sampling operation
(in this case, using Concatenated Latin Hypercube Sampling or `clhs`)
to gather the initial data points.
You will see output for each point being measured:

<!-- markdownlint-disable line-length -->

```commandline
(RandomWalk pid=10734) Continuous batching: SUBMIT EXPERIMENT. Submitted experiment custom_experiments.calculate_pressure_ideal_gas for temperature.270.0-volume.5.0-mol.0.2. Request identifier: 3201d2
(RandomWalk pid=10734) 
(RandomWalk pid=10734) Continuous batching: SUMMARY. Entities sampled and submitted: 1. Experiments completed: 0 Waiting on 1 active requests. There are 0 dependent experiments
(RandomWalk pid=10734) Continuous Batching: EXPERIMENT COMPLETION. Received finished notification for experiment...
```
<!-- markdownlint-enable line-length -->

#### Stage 2: Iterative Modeling

Once the initial characterization is complete, `trim` begins
 its main iterative loop.
In each iteration, it samples a new point, trains an `AutoGluon`
 model and checks
 if the model's accuracy is still improving.
  The points to sample is chosen by leveraging
   the information obtained in the no-prior characterization stage.

You'll see logs indicating that a model is being trained and evaluated:

<!-- markdownlint-disable line-length -->

```commandline
(RandomWalk pid=10736) 2026-01-16 14:57:19,256 INFO      AsyncIO Thread: default trim.trim_sampler: iterator            : Fitting AutoGluon TabularPredictor, iteration 5...
...
(RandomWalk pid=10736) 2026-01-16 14:57:20,723 INFO      AsyncIO Thread: default trim.trim_sampler: iterator            : [Batch under consideration: 5] Training metric: root_mean_squared_error;
(RandomWalk pid=10736) Best model: NeuralNetTorch; score_val: -8.49; holdout_score: -669.00
```
<!-- markdownlint-enable line-length -->

After a set number of iterations (defined by `iterationSize`),
it will check the stopping criterion:

<!-- markdownlint-disable line-length -->

```commandline
(RandomWalk pid=10736) 2026-01-16 14:57:48,947 INFO      AsyncIO Thread: default trim.trim_sampler: iterator            : Testing stopping criterion after measuring 14 points, mean_ratio={mean_ratio} and std_ratio={std_ratio}
(RandomWalk pid=10736) 2026-01-16 14:57:48,947 INFO      AsyncIO Thread: default trim.trim_sampler: iterator            : Stopping not triggered for i=14
```
<!-- markdownlint-enable line-length -->

#### Stage 3: Stopping and Finalizing

The iterative process continues until the model's performance stabilizes.
At this point, the stopping criterion is met,
and `trim` proceeds to train one final model on all the data it has gathered.

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

The `trim` operator saves the final trained `AutoGluon` model
to the directory specified by `outputDirectory`
 in your operation file (here, `trim_models_finalized`).
 You can now load this `TabularPredictor` in your own code
  to make predictions on any unmeasured points in your parameter space.

You can also view the entities that were sampled during the entire operation.
 `trim` actually runs two sub-operations
 (one for characterization, one for iterative modeling).
  You can see the relationship with:

```commandline
ado show related operation --use-latest
```

This will show the `discoveryspace` and the sub-operations that were run.
To see the actual data points from the final iterative phase, you can run:

```commandline
# The identifier will be different for your run
ado show entities operation randomwalk-1.3.3.dev14+177ead3c.dirty-14a313
```

This will display a table of the entities sampled and their
 measured pressure values.

<!-- markdownlint-disable line-length -->
```text
             Measurements - randomwalk-1.3.3.dev14+177ead3c.dirty-14a313              
┏━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━┓
┃ index ┃ mol  ┃ temperature ┃ volume ┃ pressure ┃ request_id ┃ entity_index ┃ valid ┃
┡━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━┩
│ 21    │ 0.40 │ 296.00      │ 8.00   │ 123.05   │ de3011     │ 0            │ True  │
│ 20    │ 0.20 │ 276.00      │ 1.00   │ 458.96   │ eec630     │ 0            │ True  │
│ 19    │ 0.20 │ 286.00      │ 2.00   │ 237.79   │ 15f0f5     │ 0            │ True  │
│ 18    │ 0.10 │ 288.00      │ 6.00   │ 39.91    │ 938036     │ 0            │ True  │
│ 17    │ 0.70 │ 278.00      │ 2.00   │ 809.00   │ b6ede7     │ 0            │ True  │
...
```
<!-- markdownlint-enable line-length -->

## Takeaways

- **Automated Surrogate Modeling**:
    The `trim` operator automates the process of building a surrogate model
    for a complex system.
- **Efficient Sampling**:
    By using an iterative, model-guided approach,
    `trim` avoids wasting resources on samples that provide
     little new information.
- **Declarative Configuration**:
    The entire process is configured with a simple YAML file,
    with no need to write complex orchestration code.
- **Auto-Stopping**:
    The stopping criterion ensures the process terminates once
     the model's quality plateaus,
    saving time and compute.
- **Reusable Artifacts**:
   The final output is a trained `AutoGluon` model,
   a powerful and easy-to-use artifact for further analysis and prediction.
