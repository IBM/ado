<!-- markdownlint-disable code-block-style -->
<!-- markdownlint-disable-next-line first-line-h1 -->
!!! info end

    A complete example operator is provided
    [here](https://github.com/IBM/ado/tree/main/plugins/operators/profile_space).
    This example operator is functional, and useful out of the box. It can be used
    as the basis to create new operators. It references this document to help tie
    details here to the implementation.

Developers can write their own [operator](working-with-operators.md) plugins to
add new operations that work on `discoveryspaces` to `ado`. Operator plugins are
written in Python and can live in their own repository.

The main part of writing an operator plugin, from an integration standpoint, is
**writing a Python function that implements a specific interface.** `ado` will
call this function to execute an operation with your operator. From this
function you then call your operator logic (or in many cases it can just live in
this function).

This page gives an overview of how to get started creating your own operator.
After reading this page the best resource is to check
[our example operator](https://github.com/IBM/ado/tree/main/plugins/operators/profile_space).

## Knowledge required

- Knowledge of Python
- Knowledge of [pydantic](https://docs.pydantic.dev/latest/) is useful, but not
  necessary

## `ado` operator functions

An operator function is a decorated Python function with a specific signature.
To execute your operator `ado` will call your function and expect it to return
output in a given way. Below is an example of such a decorated function. The
next sections describe the decorator, its parameters, and the structure of the
operation function itself.

<!-- markdownlint-disable line-length -->
```python
from orchestrator.modules.operators.collections import
    characterize_operation  # Import the decorator from this module depending on the type of operation your operator performs


@characterize_operation(
    name="my_operator",  # The name of your operator.
    description="Example operator",  # What this operator does
    configuration_model=MyOperatorOptions,  # A pydantic model that describes your operators input parameters
    configuration_model_default=MyOperatorOptions.default_parameters(),  # An example of your operators input parameters
    version="1.0",  # Version of the operator

)
def detect_anomalous_series(
        discoverySpace: DiscoverySpace,
        operationInfo: typing.Optional[FunctionOperationInfo] = None,
        **parameters,
) -> OperationOutput:
    # Your operation logic - can also call other Python modules etc.
    ...
    return operationOutput
```
<!-- markdownlint-disable line-length -->

### Operator Type

The first thing you need to do is decide what type of operator you are
creating. The choices are
[explore, characterize, learn, modify, fuse, export, or compare](working-with-operators.md).
You then import the decorator for this operator type from
`orchestrator.modules.operators.collections` and use it to decorate your
operator function.

For example, if your operator **compares** `discoveryspaces` you would do

```python
from orchestrator.modules.operators.collections import compare_operation

@compare_operation(...)
def my_comparison_operation():
```

The decorator parameters are the same for all operator/operation types.

### Operator function parameters

All operator functions take one or more `discoveryspaces` along with a
dictionary containing the inputs for the operation.

If your operation type is `explore`, `characterize`, `learn` or `modify`, your
function should have a parameter `discoverySpace` i.e.

```python
def detect_anomalous_series(
    discoverySpace: DiscoverySpace,
    operationInfo: typing.Optional[FunctionOperationInfo] = None,
    **parameters,
) -> OperationOutput:
   ...
```

If it is `fuse` or `compare` your function should have a parameter
`discoverySpaces` which is a list of `discoveryspaces` i.e.

```python
def detect_anomalous_series(
    discoverySpaces: list[DiscoverySpace],
    operationInfo: typing.Optional[FunctionOperationInfo] = None,
    **parameters,
) -> OperationOutput:
   ...
```

Operator functions also take an optional third parameter, `operationInfo`, that
holds information for `ado`. You do not have to interact with the parameter
unless you are writing an [explore operator](#creating-explore-operators).

## Describing your operation input parameters

From the previous section, the `parameters` variable will contain the parameters
values that should be used for a specific operation. However, how does `ado`
know what the valid input parameters are for your operator so the contents of
this variable will make sense?

The answer is that the input parameters to your operator are described by a
pydantic model that you give to the function decorator. Here's the example from
the previous section with the relevant fields called out:

```python
@characterize_operation(
    name="my_operator",
    description="Example operator",
    configuration_model=MyOperatorOptions,  # <- A pydantic model that describes your operators input parameters
    configuration_model_default=MyOperatorOptions(), # <- An example of your operators input parameters
    version="1.0",
)
```

Here `MyOperatorOptions` is a pydantic model that describes your operators input
parameters. The `parameters` dictionary that is passed to your operation
function will be a dump of this model. So the typical first step in the function
is to create the model for your inputs

```python
inputs = MyOperatorOptions.model_validate(parameters)
```

### Providing an example operation configuration

The decorators `configuration_model_default` parameter takes a example of your
operators parameters. If your operator's parameter model has defaults for all
fields then the simplest approach is to use those as the value of
`configuration_model_default`:

```python
    configuration_model_default=MyOperatorOptions(), # <- This will use the defaults specified for all fields of your operators parameters
```

### How your Operators input parameters model is stored and output

When outputting the default options via `ado template operator`,
`model_dump_json()` is used with no options.

When an `operation` is created using your Operator, the parameters are stored in
[the metastore](../resources/metastore.md) using `model_dump_json()` with **no
options**.

### Operation function logic

We've covered how your operator will be called. However, where do you put your
code?

If you are not creating an explore operator you can implement as you like e.g.
within the operator function or in a class or function in a separate module
called from the operator function.

If your operator type involves sampling and measuring entities e.g. it is an
optimizer, your code has some additional packaging requirements which are
discussed in [explore operators](#creating-explore-operators).

## Returning data from your operation: Operation Outputs

> [!NOTE]
>
> Any `ado` resources created will be stored in the project the operation was
> created in.

The operator function must return data using the
`orchestrators.core.operation.operation.OperationOutput` pydantic model.

```python
class OperationOutput(pydantic.BaseModel):
    metadata: typing.Annotated[
        dict,
        pydantic.Field(
            default_factory=dict,
            description="Additional metadata about the operation. ",
        ),
    ]
    resources: typing.Annotated[
        list[orchestrator.core.resources.ADOResource],
        pydantic.Field(
            default_factory=list,
            description="Array of ADO resources generated by the operation",
        ),
    ]
    exitStatus: typing.Annotated[
        OperationResourceStatus,
        pydantic.Field(
            description="Exit status of the operation. Default to success if not applied",
        ),
    ]
```

The key fields to set are:

- **resources**: A list of `ado` resources your operation created.
- **existStatus**: Indicates if the operation worked or not

### Returning non-ado resource data

If you have non-ado resource data you want to return from your operation, for
example pandas DataFrames, paths to files, text, lists etc. you can use `ado`s
[`datacontainer`](../resources/datacontainer.md) resource.

### Example

The following code snippet shows returning a dataframe, a dictionary with some
key:value pairs, and an URL:

```python
tabular_data = TabularData.from_dataframe(df)
location = ResourceLocation.locationFromURL(someURL)
data_container = DataContainer(tabularData={"main_dataframe":tabular_data},
                               data={"important_dict":results_dict},
                               locationData={"important_location": location})

return OperationOutput(resources=[DataContainerResource(config=data_container)])
```

### Storing returned resources

All resources returned by the operation will automatically be stored in the
project the operation was created in. In addition, the relationships between the
operation and the resources it creates are also automatically added. This means
`ado show related operation $OPERATION_ID` will list the resources the operation
created.

### Expected return types

Certain operation types are expected to return outputs as follows:

- fuse, modify: a new DiscoverySpaceResource and optionally a
  SampleStoreResource
- compare: a new DataContainerResource
- characterize: a new DataContainerResource

## How to update your operator input parameters

During development, there will be times when you might need to update the input
parameter model for your operator, adding, removing or modifying fields. In
these cases, it's important not to break backwards compatibility (where
possible) while making sure that users are aware of the changes to the model and
do not rely indefinitely on the model being auto upgraded.

In ado, we recommend using Pydantic before validators coupled with the
`ado upgrade` command. At a high level, you should:

1. Use a before validator to create a temporary upgrade path for your model.
2. Enable a warning in this validator using the provided support functions
   (described below). This warning will inform users that an upgrade is needed.
   The support function will automatically print the command to upgrade stored
   model versions and remove the warning. It will also display a message
   indicating that auto-upgrade functionality will be removed in a future
   release.
3. Remove the upgrade path in the specified future version.

Let's see a practical example. Consider this class as the input parameter class
in `my_operator` v1:

```python
import pydantic

class MyOperatorOptions(pydantic.BaseModel):
    my_parameter_name: int
```

And consider two cases:

- We want to deprecate a field.
- We want to apply changes to a field without deprecating it.

### Deprecating a field in your operator input parameters

Let's imagine we want to change the name of the `my_parameter_name` field to be
`my_improved_parameter_name`. The model for our operator v2 would then be:

```python
import pydantic

class MyOperatorOptions(pydantic.BaseModel):
    my_improved_parameter_name: int
```

To enable upgrading of the previous model versions when fields are being
deprecated, we recommended using a
[Pydantic Before Model Validator](https://docs.pydantic.dev/latest/concepts/validators/#model-before-validator).
This allows the dictionary content of the model to be changed as appropriate
before validation is applied. To ensure the users are aware of the change, we
will also use the `warn_deprecated_operator_parameters_model_in_use` method in
the validator:

```python
import pydantic

class MyOperatorOptions(pydantic.BaseModel):
    my_improved_parameter_name: int

    @pydantic.model_validator(mode="before")
    @classmethod
    def rename_my_parameter_name(cls, values: dict):
        from orchestrator.modules.operators.base import (
            warn_deprecated_operator_parameters_model_in_use,
        )

        old_key = "my_parameter_name"
        new_key = "my_improved_parameter_name"
        if old_key in values:

            # Notify the user that the my_parameter_name
            # field is deprecated
            warn_deprecated_operator_parameters_model_in_use(
                affected_operator="my_operator",
                deprecated_from_operator_version="v2",
                removed_from_operator_version="v3",
                deprecated_fields=old_key,
                latest_format_documentation_url="https://example.com",
            )

            # The user has set both the old
            # and the new key - the new key
            # takes precedence.
            if new_key in values:
                values.pop(old_key)
            # Set the old value in the
            # new field
            else:
                values[new_key] = values.pop(old_key)

        return values
```

When a model with the old field will be loaded, the user will see the following
warning:

```text
WARN:   The parameters for the my_operator operator have been updated as of my_operator v2.
        They are being temporarily auto-upgraded to the latest version.
        This behavior will be removed with my_operator v3.
HINT:   Run ado upgrade operations to upgrade the stored operations.
        Update your operation YAML files to use the latest format: https://example.com
```

### Updating a field in your operator input parameters without deprecating it

Let's imagine we want to change the type of the `my_parameter_name` field to be
`str`. The model for our operator v2 would then be:

```python
import pydantic

class MyOperatorOptions(pydantic.BaseModel):
    my_parameter_name: str
```

To enable upgrading of the previous model versions when fields are not being
deprecated, we recommended using a
[Pydantic Before Field Validator](https://docs.pydantic.dev/latest/concepts/validators/#field-before-validator).
This allows the specific field to be changed as appropriate before validation is
applied. To ensure the users are aware of the change, we will also use the
`warn_deprecated_operator_parameters_model_in_use` method in the validator:

> [!NOTE]
>
> The method being called is the same as the one for
> [warning about deprecated fields](#deprecating-a-field-in-your-operator-input-parameters),
> but we omit the `deprecated_fields` parameter.

```python
import pydantic

class MyOperatorOptions(pydantic.BaseModel):
    my_parameter_name: str

    @pydantic.field_validator("my_parameter_name", mode="before")
    @classmethod
    def convert_my_parameter_name_to_string(cls, value: int | str):
        from orchestrator.modules.operators.base import (
            warn_deprecated_operator_parameters_model_in_use,
        )

        if isinstance(value, int):
            # Notify the user that the parameters of my_operator
            # have been updated
            warn_deprecated_operator_parameters_model_in_use(
                affected_operator="my_operator",
                deprecated_from_operator_version="v2",
                removed_from_operator_version="v3",
                latest_format_documentation_url="https://example.com",
            )
            value = str(value)

        return value
```

When a model using `int`s will be loaded, the user will see the following
warning:

```text
WARN:   The parameters for the my_operator operator have been updated as of my_operator v3.
        They are being temporarily auto-upgraded to the latest version.
        This behavior will be removed with my_operator v3.
HINT:   Run ado upgrade operations to upgrade the stored operations.
        Update your operation YAML files to use the latest format: https://example.com
```

## Nesting Operations

Operators can use other operators. For example your operator can create
operations using other operators and consume the results. You access other
operators via the relevant collection in
`orchestrator.modules.operators.collections`. For example to use the RandomWalk
operator

```python
from orchestrator.modules.operators.collections import explore

@learn_operation(...)
def my_learning_operation(...):
    ...
    #Note: The name of the function called (here random_walk() ) is the operator name
    random_walk_output = explore.random_walk(...Args...)
    ...
```

> [!IMPORTANT]
>
> The name used to call an operator function is the name of the
> operator. This is the name given to the decorator `name` parameter and is the
> name shown by `ado get operators`

You access the data of the operation from the OperationOutput instance it
returns. Any `ado` resources the nested operation creates will have been
automatically added to the correct project by `ado`.

## Handling Keyboard Interrupts (SIGINT)

> [!NOTE]
>
> If your operator does not create any ado resources, you don't need
> to do anything.

Your operator must ensure that all resources it creates, along with their
relationships, are recorded in the project database if a keyboard interrupt
(CTRL+C) occurs during execution. For details on how resources are handled under
normal conditions, see [Storing Returned
Resources](#storing-returned-resources).

By default, ado ensures that when a keyboard interrupt (CTRL+C) occurs:

- Any nested operations created by your operator are stored.
- The relationship to the nested operation that was executing at the time of the
  interrupt is stored.

However, the following are **not stored by default**:

- Non-operation resources (e.g., spaces, data containers) and their
  relationships created before the interrupt.
- Relationships to nested operations that were already completed.

To handle these cases, wrap your operator logic in a try/except block as shown
below

```python
from orchestrator.modules.operators.base import InterruptedOperationError

try:
    # operator logic
    ...
except KeyboardInterrupt as error:
    # Assumes created_resources is an array containing all ado resource already created
    raise InterruptedOperationError(resources=created_resources) from error
except (
    InterruptedOperationError
) as nested_operation_error:  # This is when a nested operation was interrupted first
    # IMPORTANT: You must add the identifier of the interrupted nested operation
    raise InterruptedOperationError(
        resources=created_resources, identifier=nested_operation_error.identifier
    ) from error
```

## Creating Explore Operators

Explore operators sample and measure entities. In `ado` all explore operation
run as distributed ray jobs with:

- actuator ray actors for performing measurements
- discovery space manager actor for storing and notifying about measurement
  results

This means explore operators need to be implemented differently to the others,
in particular

- The logic of your explore operator must be implemented as a ray actor (a
  class)
- The explore operator functions must call this class i.e. you won't have any
  operator logic in the function

### Explore operation functions

All explore operation functions follow this pattern:

```python
@explore_operation(
    name="ray_tune",
    description=RayTune.description(),
    configuration_model=RayTuneConfiguration,
    configuration_model_default=RayTuneConfiguration(),
)
def ray_tune(
        discoverySpace: DiscoverySpace,
        operationInfo: FunctionOperationInfo = FunctionOperationInfo(),
        **kwargs: typing.Dict,
) -> OperationOutput:
    """
    Performs an optimization on a given discoverySpace

    """

    from orchestrator.core.operation.config import OperatorModuleConf
    from orchestrator.module.operator.orchestrate import orchestrate_explore_operation


    ## This describes where the class that implements your explore operation is
    module = OperatorModuleConf(
        moduleName="ado_ray_tune.operator",  # The name of the package containing your explore actor
        moduleClass="RayTune",  # The name of your explore actor class
    )

    # Tell ado to execute your class
    return orchestrate_explore_operation(
        discovery_space=discoverySpace,
        module=module,
        parameters=kwargs,
        operation_info=operationInfo,  # Important: This is where you must pass the operationInfo parameter to ado
    )
```

### Explore operator classes

TBA

## Operator plugin packages

Operator plugin packages follow a standard python structure

```terminaloutput
$YOUR_REPO_NAME
│  └── $YOUR_PLUGIN_PACKAGE        # Your plugin
│      ├── __init__.py
│      └── ...
└── pyproject.toml
```

The key to making it an ado plugin is having a
`[project.entry-points."ado.operators"]` section in the `pyproject.toml` e.g.

```yaml
[project]
name = "ado-ray-tune" #Note: this is the distribution name of the Python package. Your ado operator(s) can have different ado identifier
version = "0.1.0"
dependencies = [
  #Dependencies
]

[project.entry-points."ado.operators"]
ado-ray-tune = "ado_ray_tune.operator_function" # The key is the distribution name of your Python package and the value is the Python module in your package containing your decorated operator function
```

This references the Python module (file) that contains your operator function

> [!NOTE]
>
> You can define multiple operator functions in the referenced module.
