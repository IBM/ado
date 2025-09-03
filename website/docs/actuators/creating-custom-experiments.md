<!-- markdownlint-disable-next-line first-line-h1 -->
> [!NOTE]
>
> The
> [search a space with an optimizer](../examples/best-configuration-search.md)
> example contains the code described here

Often you might want to use an experiment that is a simple python function. A
typical example is a cost-function to be used with an optimization. `ado`
provides a way to add such functions as experiments without having to create an
Actuator class.

The process involves creating a python package with two files

- A python module with your functions
- A yaml file describing the experiments they provide

And then installing this package.

## Custom experiment package structure

To create a package with one or more custom experiments you **must** use the
following package structure

<!-- markdownlint-disable line-length -->
```text
$YOUR_REPO_NAME
   - setup.py
   - ado_actuators/    # This is `ado`'s namespaced package for actuator plugins and custom experiments
       - $YOUR_CUSTOM_EXPERIMENT_PLUGIN_PACKAGE/  # Your package with custom experiments
         - __init__.py
         - $EXPERIMENTS.py # Python file with your function in it
         - custom_experiments.yaml # A yaml file describing the custom experiments your package provides
```
<!-- markdownlint-enable line-length -->

## Writing the experiment catalog

The experiment catalog contains the following critical pieces of information:

1. What your experiment is called
2. Who is going to execute your experiment - for custom python functions this
   will **always** be the special actuator "custom_experiments"
3. What the python function that executes your experiment is called
4. What properties your experiment measures
5. The properties from other experiments this experiment requires as input - if
   your function does not require properties from another experiment you don't
   need this field

A catalog can define multiple experiments. Each one is a new element in the
top-level list.

An example experiment description file is:

```yaml
{%
  include  "../../../examples/optimization_test_functions/custom_experiments/ado_actuators/optimization_test_functions/custom_experiments.yaml"
%}
```

This YAML describes:

- a single experiment called `nevergrad_opt_3d_test_func`
- the measurement will be executed using a python function called
  `artificial_function`
  - the function is in the module
    `ado_actuators.optimization_test_functions.optimization_test_functions` i.e.
    `ado_actuators.$YOUR_CUSTOM_EXPERIMENT_PLUGIN_PACKAGE.$EXPERIMENTS`
    following above package layout)
  - this name will always start with `ado_actuators`
- The experiment has a set of required input properties - `x0`, `x1` and `x2` -
  and set of optional input properties - `name` and `num_blocks`
  - these input properties are what the python function is expected to use
- The experiment measures a single target property `function_value`
  - so applying this experiment will return a value of an observed property
    called `nevergrad_opt_3d_test_func.total_cost`

## Writing your custom experiment functions

The python function that implements the experiment described in the catalog must

1. Be called the name you gave in the catalog (`metadata.function` field)
2. Have a specific signature and return value

A snippet of the above function, `artifical_function`, showing the signature and
return value is:

<!-- markdownlint-disable line-length -->
```python
import typing
from orchestrator.schema.experiment import Experiment
from orchestrator.schema.entity import Entity
from orchestrator.schema.property_value import PropertyValue


def artificial_function(
        entity: Entity,
        experiment: Experiment,
        parameters=None,  # deprecated field
) -> typing.List[PropertyValue]:
    """

    :param entity: The entity to be measured
    :param experiment: The Experiment object representing the exact Experiment to perform
        Required as multiple experiments can measure this property
    :param parameters: A dictionary.
    :return: A list of PropertyValue objects
    """
    # parameters is a dictionary of key:value pairs of the experiment required/optional inputs
    # defined in custom_experiments.yaml
    parameters = experiment.propertyValuesFromEntity(entity)

    #Experiment logic elided
    ...

    # At end return the results
    pv = PropertyValue(
        value=value,
        property=experiment.observedPropertyForTargetIdentifier("function_value"),
        valueType=ValueTypeEnum.NUMERIC_VALUE_TYPE,
    )
    return [pv]
```
<!-- markdownlint-enable line-length -->

In the above function `entity` and `experiment` are `ado` objects describing
what to measure and what to measure it with. Since the custom experiment package
only defined one experiment (see
[Writing the experiment catalog](#writing-the-experiment-catalog)) the
`experiment` object will represent the `nevergrad_opt_3d_test_func` experiment.
The `entity` and `experiment` objects can be converted into a dictionary of
required and optional input properties names and values using
`experiment.propertyValuesFromEntity`

Once the values to return have been calculated the function has to create
`PropertyValue` objects as shown above.

To find out more about the class instances passed to this function check the
`ado` source code.

!!! warning end
    <!-- markdownlint-disable-next-line code-block-style -->
    If your function returns properties with different names than those
    you specified in the catalog for the experiment entry, they will be ignored.

## Using your custom experiments: the custom_experiments actuator

All your custom experiments in `ado` are accessed via a special actuator called
_custom_experiments_.

### Add your experiments to `ado`

First to add your experiments to `ado` run `pip install` in the same directory
as your
[custom experiment packages `setup.py`](#custom-experiment-package-structure)

Confirm the experiment has been added:

```commandline
ado describe actuators --details
```

If the custom experiment was the one defined in
[above](#writing-the-experiment-catalog) you would see a new experiment entry
for the _objective_functions_ actuator called `ml-multicloud-cost-v1.0`.

### Add a custom experiment to a `discoveryspace`

To use a custom experiment you declare it the `measurementspace` of a
`discoveryspaces` - exactly like other experiments. The only difference is you
used the `custom_experiments` actuator.

```yaml
{% include "../../../examples/optimization_test_functions/space.yaml" %}
```

Note `ado` will validate the measurement space as normal. So in this case if the
custom experiment `benchmark_performance` from the `replay` actuator is not
included the space creation will fail.

## Next Steps

Follow the
[search a space with an optimizer](../examples/best-configuration-search.md)
example to see how the custom experiment described here works in practice.
