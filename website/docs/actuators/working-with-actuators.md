---
status: published #Status can be draft, reviewed or published. 
---

An **actuator** is a code module that provides experiment protocols that can measure properties of entities. 
See [core-concepts:actuators](../core-concepts/actuators.md) for more details on what an actuator is and see [discoveryspaces](../resources/discovery-spaces.md)
shows how they are used to create `discoveryspaces`.

This section covers how you install and configure actuators, [create new actuators to extend `ado`](creating-actuator-classes.md) as well as specific documentation
for various actuators available.

You can also add [your own custom experiments](creating-custom-experiments.md) using the special actuator [*custom_experiments*](creating-custom-experiments.md#using-your-custom-experiments-the-custom_experiments-actuator)

!!! info end

    Most actuators are plugins. They are pieces of code that can be installed independently from `ado` and that `ado` can dynamically discover.
    Custom experiments are also plugins. 

## Listing available Actuators

To see a list of available actuators execute

```commandline
ado get actuators
```

to see the experiments each provides 

```commandline
ado get actuators --details
```

## Builtin Actuators

`ado` comes with a set of builtin actuators related to materials discovery: `molecule-embeddings`, `mordred`, `st4sd` and `molformer-toxicity`. 
These are available automatically,

### Special actuators: replay and custom_experiments

Two of `ado`'s builtin actuators are special: `custom_experiments` and `replay`. 

`custom_experiments` creating experiments from python functions without having to write a full Actuator. 
The [creating custom experiments](creating-custom-experiments.md) page describes this in detail.

The `replay` actuator allows you to use property values from experiments that were performed outside of `ado` i.e. no Actuator exists to measure them. 
Often you might want to perform some analysis on a `discoveryspace` using these values or to perform a search using an objective-function defined on these values.
See the [replay actuator](replay.md) page to learn more about how to do this.

## Actuator Plugins

In addition to the builtin actuators, anyone can extend `ado` with **actuator plugins**.
All actuator plugins are python packages (see [creating actuator classes](creating-actuator-classes.md)) and can be
installed in the usual ways with `pip`.

### Actuator plugins distributed with `ado` 

A number of actuators are included in the `ado` source repository. These include

- caikit-config-explorer: An actuator for foundation model inference tests
- SFTTrainer: An actuator for foundation model fine-tuning tests

### Installing an actuator plugin locally from source

A typical pattern to install from source is:
```commandline
git clone $ACTUATOR_REPO
cd $ACTUATOR_REPO
pip install .
```

However, always check the installation instructions from the plugin repository for deviations from this pattern. 

!!! info end

    A source code repository may include multiple actuator plugins. In this case installing will usually install all the actuator plugins.

To install these see [installing plugins](../getting-started/install.md#installing-plugins) in our install documentation.

### Installing an actuator plugin from a pypi instance or wheel

An actuator plugin can be uploaded to a pypi package index. If it is called $ACTUATORMAME it can be installed with

```commandline
pip install $ACTUATORNAME --extra-index-url=$PRIVATE_PYPI_INDEX
```
here we show using `--extra-index-url` to install from a private pypi index. Omit this if the plugin is not hosted privately. 

Note: multiple plugins can be bundled together in one pypi package.

### Dynamic installation of actuators on a remote Ray cluster

If you are running `ado` operations on a remote ray cluster, as ray job, you may want, or need, to dynamically install an actuator plugin or its latest version.
The recommended way of doing this is to build a wheel and install it when submitting the ray job. 
This is described in the [running ado on a remote ray cluster](../getting-started/remote_run.md#installing-ado-and-required-plugins-on-a-remote-ray-cluster-from-source).

Some additional notes about this process when you are developing an actuator:

- Make sure plugin code changes are committed (if using `setuptools_scm` for versioning)
     - If they are not committed then the version of the built wheel will not change i.e. it will be same as for a wheel built before the changes
     - If a wheel with this version was already installed in ray cluster by a previous job, ray will use the cached version, and not your new version
- Make sure new files you want to package with the wheel are committed
     - The setup.py for the plugins only adds committed non-python files
 

## What's next

<div class="grid cards" markdown>

  -   :octicons-workflow-24:{ .lg .middle } __Try our examples__

      ---

      Explore using some of these actuators with our [examples](../examples/examples.md).

      [Our examples :octicons-arrow-right-24:](../examples/examples.md)

-   :octicons-rocket-24:{ .lg .middle } __Learn about Operators__

    ---

    Learn about extending ado with new [Operators](../operators/working-with-operators.md).

    [Creating new Operators :octicons-arrow-right-24:](../operators/working-with-operators.md)


</div>


