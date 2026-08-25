<!-- markdownlint-disable-next-line first-line-h1 -->
An **actuator** is a code module that provides experiment protocols that can
measure properties of entities. See [actuators](../../concepts/actuators.md)
for more details on what an actuator is and read
[discoveryspaces](../../resources/discovery-spaces.md) to learn how they are used
to create `discoveryspaces`.

This section covers how you install and configure actuators,
[create new actuators to extend `ado`](../../developer-guide/creating-actuator-classes.md)
as well as specific documentation for various actuators available.

You can also add [your own custom experiments](../../developer-guide/creating-custom-experiments.md)
using the special actuator
[_custom_experiments_](../../developer-guide/creating-custom-experiments.md#using-your-custom-experiment).

> [!NOTE]  Actuators and Plugins
>
> Most actuators are plugins: pieces of code that can be installed
> independently from `ado` and that `ado` can dynamically discover. Custom
> experiments are also plugins.

## Listing available Actuators

To see a list of available actuators, including their description,
number of experiments, and version, execute

<!-- markdownlint-disable-next-line code-block-style -->
```commandline
ado get actuators --details
```

Below is an example of the output:

<!-- markdownlint-disable line-length -->

```commandline
┌───────┬────────────────────┬─────────────┬────────────────────────────────────────────────────┬─────────┐
│ INDEX │ ACTUATOR ID        │ EXPERIMENTS │ DESCRIPTION                                        │ VERSION │
├───────┼────────────────────┼─────────────┼────────────────────────────────────────────────────┼─────────┤
│ 0     │ custom_experiments │ 2           │ Actuator for applying user supplied custom         │ 2.0.0   │
│       │                    │             │ experiments                                        │         │
│ 1     │ mock               │ 2           │ A actuator class for testing                       │ 2.0.0   │
│ 2     │ replay             │ 0           │ Special actuator for handling externally defined   │ 2.0.0   │
│       │                    │             │ experiments (experiments we don't have code for)   │         │
│ 3     │ vllm_performance   │ 25          │ VLLM performance testing actuator for ado          │ 1.13.1  │
└───────┴────────────────────┴─────────────┴────────────────────────────────────────────────────┴─────────┘
```

<!-- markdownlint-enable line-length -->

## Listing available Experiments

To see the experiments each actuator provides, including their description,
execute

<!-- markdownlint-disable-next-line code-block-style -->
```commandline
ado get experiments --details
```

The output will be similar to:

<!-- markdownlint-disable line-length -->
```terminaloutput
┌───────┬────────────────────┬──────────────────────────────────────┬─────────┬──────────────────────────────────────────┐
│ INDEX │ ACTUATOR ID        │ EXPERIMENT ID                        │ VERSION │ DESCRIPTION                              │
├───────┼────────────────────┼──────────────────────────────────────┼─────────┼──────────────────────────────────────────┤
│ 0     │ custom_experiments │ avoid_oom_recommender                │ None    │ An AutoConf recommender that preserves   │
│       │                    │                                      │         │ the requested number of GPUs if it won't │
│       │                    │                                      │         │ cause GPU OOM, otherwise recommends the  │
│       │                    │                                      │         │ minimum number of GPUs needed. Keeps the │
│       │                    │                                      │         │ per-device batch size constant.          │
│ 1     │ custom_experiments │ min_gpu_recommender                  │ None    │ An AutoConf plugin that suggests the     │
│       │                    │                                      │         │ minimum number of gpus per worker and    │
│       │                    │                                      │         │ number of workers necessary to execute a │
│       │                    │                                      │         │ Tuning job                               │
│ 2     │ vllm_performance   │ vllm-bench-deployment                │ 1.0.0   │ VLLM performance testing across compute  │
│       │                    │                                      │         │ resource and workload configuration      │
│ 3     │ vllm_performance   │ geospatial-vllm-bench-deployment     │ 1.0.0   │ VLLM performance testing across compute  │
│       │                    │                                      │         │ resource and workload configuration for  │
│       │                    │                                      │         │ geospatial models                        │
│ 4     │ vllm_performance   │ test-agentic-tool-calling            │ 1.2.0   │ Test inference performance of an         │
│       │                    │                                      │         │ agent-style model deployed by vLLM       │
│       │                    │                                      │         │ across compute resource and workload     │
│       │                    │                                      │         │ configurations                           │
│ ...   │ ...                │ ...                                  │ ...     │ ...                                      │
└───────┴────────────────────┴──────────────────────────────────────┴─────────┴──────────────────────────────────────────┘
```
<!-- markdownlint-enable line-length -->

## Special actuators: replay and custom_experiments

`ado` has two special builtin actuators: `custom_experiments` and `replay`.

`custom_experiments` allows users to create experiments from python functions
without having to write a full Actuator. The
[creating custom experiments](../../developer-guide/creating-custom-experiments.md)
page describes this in detail.

The `replay` actuator allows you to use property values from experiments that
were performed outside of `ado` i.e. no Actuator exists to measure them. Often
you might want to perform some analysis on a `discoveryspace` using these values
or to perform a search using an objective-function defined on these values. See
the [replay actuator](replay.md) page to learn more about how to do this.

## Actuator Plugins

Anyone can extend `ado` with **actuator plugins**. All actuator plugins are
python packages (see [creating actuator classes](../../developer-guide/creating-actuator-classes.md))
and can be installed in the usual ways with `pip`.

### Actuator plugins distributed with `ado`

The following actuators are distributed with `ado`:

- [SFTTrainer](sft-trainer.md): An actuator for testing foundation model
  fine-tuning performance
- [vllm_performance](https://github.com/IBM/ado/tree/main/plugins/actuators/vllm_performance):
  An actuator for testing foundation model inference performance

### Dynamic installation of actuators on a remote Ray cluster

If you are running `ado` operations on a remote Ray cluster, as Ray jobs, you may
want, or need, to dynamically install an actuator plugin or its latest version.
This is described in the
[running ado on a remote ray cluster](../advanced/remote-execution.md#dynamic-installation-from-pypi).

Some additional notes about this process when you are developing an actuator:

- Make sure plugin code changes are committed before building a wheel for remote
  use.
    - Uncommitted changes produce a unique dev version (e.g.
      `X.Y.Z.devN+g<commit>.d<timestamp>`), so Ray will not serve a stale cached
      wheel. However, the safest approach is to commit before building.
- Ensure new files to be packaged with the wheel are committed
    - Only committed non-python files are included in the wheel

## What's next

<!-- markdownlint-disable line-length MD046 -->
<!-- markdownlint-disable-next-line no-inline-html -->
<div class="grid cards" markdown>

- :octicons-workflow-24:{ .lg .middle } **Try our examples**

      ---

      Explore using some of these actuators with our [examples](../examples/index.md).

      [Our examples :octicons-arrow-right-24:](../examples/index.md)

- :octicons-rocket-24:{ .lg .middle } **Learn about Operators**

    ---

    Learn about extending ado with new [Operators](../operators/working-with-operators.md).

    [Creating new Operators :octicons-arrow-right-24:](../operators/working-with-operators.md)

</div>
<!-- markdownlint-enable line-length MD046 -->
