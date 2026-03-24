<!-- markdownlint-disable-next-line first-line-h1 -->
An **actuator** is a code module that provides experiment protocols that can
measure properties of entities. See [actuators](../core-concepts/actuators.md)
for more details on what an actuator is and read
[discoveryspaces](../resources/discovery-spaces.md) to learn how they are used
to create `discoveryspaces`.

This section covers how you install and configure actuators,
[create new actuators to extend `ado`](creating-actuator-classes.md) as well as
specific documentation for various actuators available.

You can also add [your own custom experiments](creating-custom-experiments.md)
using the special actuator
[_custom_experiments_](creating-custom-experiments.md#using-your-custom-experiment).

> [!NOTE]  Actuators and Plugins
>
> Most actuators are plugins: pieces of code that can be installed
> independently from `ado` and that `ado` can dynamically discover. Custom
> experiments are also plugins.

## Listing available Actuators

To see a list of available actuators execute

<!-- markdownlint-disable-next-line code-block-style -->
```commandline
ado get actuators
```

You can also use `ado get actuators --details` which in addition
outputs the description of the actuators, the number of
experiments they provide and their version. Below is an example
of the output:

<!-- markdownlint-disable line-length -->

```commandline
┌────────────────────┬─────────────┬─────────────────────────────────────────────────────┬───────────────────────────┐
│ ACTUATOR ID        │ EXPERIMENTS │ DESCRIPTION                                         │ VERSION                   │
├────────────────────┼─────────────┼─────────────────────────────────────────────────────┼───────────────────────────┤
│ SFTTrainer         │ 5           │ An actuator for benchmarking fine-tuning of         │ 1.5.1.dev13+ga1833142b    │
│                    │             │ foundation models                                   │                           │
│ custom_experiments │ 6           │ Actuator for applying user supplied custom          │ 1.5.1.dev8+531c6444.dirty │
│                    │             │ experiments                                         │                           │
│ mock               │ 2           │ A actuator class for testing                        │ 1.5.1.dev8+531c6444.dirty │
│ replay             │ 0           │ Special actuator for handling externally defined    │ 1.5.1.dev8+531c6444.dirty │
│                    │             │ experiments (experiments we don't have code for)    │                           │
│ robotic_lab        │ 1           │ A template for creating an actuator                 │ 1.5.1.dev13+ga1833142b    │
└────────────────────┴─────────────┴─────────────────────────────────────────────────────┴───────────────────────────┘
```

<!-- markdownlint-enable line-length -->

## Listing available Experiments

To see the experiments each actuator provides

<!-- markdownlint-disable-next-line code-block-style -->
```commandline
ado get experiments
```

You can also get see the description of each experiment (if provided)
with `ado get experiments --details`.
The output will be similar to:

<!-- markdownlint-disable line-length -->
```terminaloutput
┌────────────────────┬─────────────────────────────────────┬─────────────────────────────────────────────────────────┐
│ ACTUATOR ID        │ EXPERIMENT ID                       │ DESCRIPTION                                             │
├────────────────────┼─────────────────────────────────────┼─────────────────────────────────────────────────────────┤
│ SFTTrainer         │ finetune_full_benchmark-v1.0.0      │ Measures the performance of full-finetuning a model for │
│                    │                                     │ a given (GPU model, number GPUS, batch_size,            │
│                    │                                     │ model_max_length, number nodes) combination.            │
│ SFTTrainer         │ finetune_full_stability-v1.0.0      │ Performs 5 full finetune runs of 5 steps each on a      │
│                    │                                     │ model and reports the fraction of those that resulted   │
│                    │                                     │ in GPU OOM, Other error, or No Error for a given (GPU   │
│                    │                                     │ model, number GPUS, batch_size, model_max_length)       │
│                    │                                     │ combination.                                            │
│ SFTTrainer         │ finetune_gptq-lora_benchmark-v1.0.0 │ Measures the performance of GPTQ-LORA tuning a model    │
│                    │                                     │ for a given (GPU model, number GPUS, batch_size,        │
│                    │                                     │ model_max_length, number nodes) combination.            │
│ SFTTrainer         │ finetune_lora_benchmark-v1.0.0      │ Measures the performance of LORA tuning a model for a   │
│                    │                                     │ given (GPU model, number GPUS, batch_size,              │
│                    │                                     │ model_max_length, number nodes) combination.            │
│ SFTTrainer         │ finetune_pt_benchmark-v1.0.0        │ Measures the performance of prompt-tuning a model for a │
│                    │                                     │ given (GPU model, number GPUS, batch_size,              │
│                    │                                     │ model_max_length, number nodes) combination.            │
│ custom_experiments │ acid_test                           │                                                         │
│ custom_experiments │ avoid_oom_recommender               │ An AutoConf recommender that suggests the minimum       │
│                    │                                     │ number of gpus per worker and number of workers         │
│                    │                                     │ necessary to execute a Tuning job whilekeeping the per  │
│                    │                                     │ GPU batch size constant                                 │
│ custom_experiments │ calculate_density                   │                                                         │
│ custom_experiments │ min_gpu_recommender                 │ An AutoConf plugin that suggests the minimum number of  │
│                    │                                     │ gpus per worker and number of workers necessary to      │
│                    │                                     │ execute a Tuning job                                    │
│ custom_experiments │ ml-multicloud-cost-v1.0             │                                                         │
│ custom_experiments │ nevergrad_opt_3d_test_func          │                                                         │
│ mock               │ test-experiment                     │                                                         │
│ mock               │ test-experiment-two                 │                                                         │
│ robotic_lab        │ peptide_mineralization              │ Measures adsorption of peptide lanthanide combinations  │
└────────────────────┴─────────────────────────────────────┴─────────────────────────────────────────────────────────┘
```
<!-- markdownlint-enable line-length -->

## Special actuators: replay and custom_experiments

`ado` has two special builtin actuators: `custom_experiments` and `replay`.

`custom_experiments` allows users to create experiments from python functions
without having to write a full Actuator. The
[creating custom experiments](creating-custom-experiments.md) page describes
this in detail.

The `replay` actuator allows you to use property values from experiments that
were performed outside of `ado` i.e. no Actuator exists to measure them. Often
you might want to perform some analysis on a `discoveryspace` using these values
or to perform a search using an objective-function defined on these values. See
the [replay actuator](replay.md) page to learn more about how to do this.

## Actuator Plugins

Anyone can extend `ado` with **actuator plugins**. All actuator plugins are
python packages (see [creating actuator classes](creating-actuator-classes.md))
and can be installed in the usual ways with `pip`.

### Actuator plugins distributed with `ado`

The following actuators are distributed with `ado`:

- [SFTTrainer](sft-trainer.md): An actuator for testing foundation model
  fine-tuning performance
- [vllm_performance](https://github.com/IBM/ado/tree/main/plugins/actuators/vllm_performance):
  An actuator for testing foundation model inference performance

## Installing actuator plugins

Refer to our [installing plugins](../getting-started/install.md#installing-plugins)
documentation.

### Dynamic installation of actuators on a remote Ray cluster

If you are running `ado` operations on a remote Ray cluster, as Ray jobs, you may
want, or need, to dynamically install an actuator plugin or its latest version.
This is described in the
[running ado on a remote ray cluster](../getting-started/remote_run.md#dynamic-installation-from-pypi).

Some additional notes about this process when you are developing an actuator:

- Make sure plugin code changes are committed (if using `setuptools_scm` for
  versioning)
    - If they are not committed then the version of the built wheel will not
    change i.e. it will be same as for a wheel built before the changes
    - If a wheel with this version was already installed in ray cluster by a
    previous job, Ray will use the cached version instead of your updated one
- Ensure new files to be packaged with the wheel are committed
    - The setup.py for the plugins only adds committed non-python files

## What's next

<!-- markdownlint-disable line-length MD046 -->
<!-- markdownlint-disable-next-line no-inline-html -->
<div class="grid cards" markdown>

- :octicons-workflow-24:{ .lg .middle } **Try our examples**

      ---

      Explore using some of these actuators with our [examples](../examples/examples.md).

      [Our examples :octicons-arrow-right-24:](../examples/examples.md)

- :octicons-rocket-24:{ .lg .middle } **Learn about Operators**

    ---

    Learn about extending ado with new [Operators](../operators/working-with-operators.md).

    [Creating new Operators :octicons-arrow-right-24:](../operators/working-with-operators.md)

</div>
<!-- markdownlint-enable line-length MD046 -->