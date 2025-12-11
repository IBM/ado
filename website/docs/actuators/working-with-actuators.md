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

!!! info end

    Most actuators are plugins: pieces of code that can be installed
    independently from `ado` and that `ado` can dynamically discover. Custom
    experiments are also plugins.

## Listing available Actuators

To see a list of available actuators execute

<!-- markdownlint-disable-next-line code-block-style -->
```commandline
ado get actuators
```

to see the experiments each provides

<!-- markdownlint-disable-next-line code-block-style -->
```commandline
ado get actuators --details
```

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

<!-- markdownlint-disable line-length -->
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
<!-- markdownlint-enable line-length -->