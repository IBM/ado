<!-- markdownlint-disable code-block-style -->
<!-- markdownlint-disable first-line-h1 -->

!!! info end

    A complete example operator is provided
    [here](https://github.com/IBM/ado/tree/main/plugins/operators/profile_space).
    This example operator is functional, and useful out of the box. It can be used
    as the basis to create new operators. It references this document to help tie
    details here to the implementation.

Developers can write their own [operator](../user-guide/operators/working-with-operators.md)
plugins to add new operations that work on `discoveryspaces` to `ado`.
Operator plugins are
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
import typing
from ado.modules.operators.collections import (
    characterize_operation,
)  # Import the decorator from this module depending on the type of operation your operator performs


@characterize_operation(
    name="my_operator",  # The name of your operator.
    description="Example operator",  # What this operator does
    configuration_model=MyOperatorOptions,  # A pydantic model that describes your operators input parameters
    example_configuration=MyOperatorOptions.example_configuration(),  # An example of your operators input parameters
    version="1.0",  # Version of the operator
)
def detect_anomalous_series(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
    **parameters: typing.Any,
) -> OperationOutput:
    # Your operation logic - can also call other Python modules etc.
    ...
    return operationOutput
```

<!-- markdownlint-disable line-length -->

### Operator Type

The first thing you need to do is decide what type of operator you are creating.
The choices are
[explore, characterize, learn, modify, fuse, export, or compare](../user-guide/operators/working-with-operators.md).
You then import the decorator for this operator type from
`ado.modules.operators.collections` and use it to decorate your
operator function.

For example, if your operator **compares** `discoveryspaces` you would do

```python
from ado.modules.operators.collections import compare_operation

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
import typing


def detect_anomalous_series(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
    **parameters: typing.Any,
) -> OperationOutput: ...
```

If it is `fuse` or `compare` your function should have a parameter
`discoverySpaces` which is a list of `discoveryspaces` i.e.

```python
import typing


def detect_anomalous_series(
    discoverySpaces: list[DiscoverySpace],
    operationInfo: FunctionOperationInfo | None = None,
    **parameters: typing.Any,
) -> OperationOutput: ...
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
    example_configuration=MyOperatorOptions(), # <- An example of your operators input parameters
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

The decorators `example_configuration` parameter takes an example of your
operators parameters. If your operator's parameter model has defaults for all
fields then the simplest approach is to use those as the value of
`example_configuration`:

```python
example_configuration = (
    MyOperatorOptions(),
)  # <- This will use the defaults specified for all fields of your operators parameters
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
`ado.core.operation.operation.OperationOutput` pydantic model.

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
        list[ado.core.resources.ADOResource],
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
- **exitStatus**: Indicates if the operation worked or not

### Returning non-ado resource data

If you have non-ado resource data you want to return from your operation, for
example pandas DataFrames, paths to files, text, lists etc. you can use `ado`'s
[`datacontainer`](../resources/datacontainer.md) resource.

### Example

The following code snippet shows returning a dataframe, a dictionary with some
key:value pairs, and an URL:

```python
tabular_data = TabularData.from_dataframe(df)
location = ResourceLocation.locationFromURL(someURL)
data_container = DataContainer(
    tabularData={"main_dataframe": tabular_data},
    data={"important_dict": results_dict},
    locationData={"important_location": location},
)

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
        from ado.modules.operators.base import (
            warn_deprecated_operator_parameters_model_in_use,
        )
        from ado.utilities.dictionaries import (
            get_nested_value,
            has_nested_field,
            remove_nested_field,
            set_nested_value,
        )

        old_key = "my_parameter_name"
        new_key = "my_improved_parameter_name"
        if has_nested_field(values, old_key):
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
            if has_nested_field(values, new_key):
                remove_nested_field(values, old_key)
            # Set the old value in the
            # new field
            else:
                set_nested_value(values, new_key, get_nested_value(values, old_key))
                remove_nested_field(values, old_key)

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
        from ado.modules.operators.base import (
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
`ado.modules.operators.collections`. For example to use the RandomWalk
operator

```python
from ado.modules.operators.collections import explore

@learn_operation(...)
def my_learning_operation(...):
    ...
    #Note: The name of the function called (here random_walk() ) is the operator name
    random_walk_output = explore.random_walk(...Args...)
    ...
```

> [!IMPORTANT]
>
> The name used to call an operator function is the name of the operator. This
> is the name given to the decorator `name` parameter and is the name shown by
> `ado get operators`

You access the data of the operation from the OperationOutput instance it
returns. Any `ado` resources the nested operation creates will have been
automatically added to the correct project by `ado`.

## Handling Keyboard Interrupts (SIGINT)

> [!NOTE]
>
> If your operator does not create any ado resources, you don't need to do
> anything.

Your operator must ensure that all resources it creates, along with their
relationships, are recorded in the project database if a keyboard interrupt
(CTRL+C) occurs during execution. For details on how resources are handled under
normal conditions, see
[Storing Returned Resources](#storing-returned-resources).

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
from ado.modules.operators.base import InterruptedOperationError

try:
    # operator logic
    ...
except KeyboardInterrupt as error:
    # Assumes created_resources lists all ADO resources already created, and
    # operation_id is the identifier string of this operation (from your context).
    raise InterruptedOperationError(
        operation_identifier=operation_id,
        resources=created_resources,
    ) from error
except InterruptedOperationError as nested_operation_error:
    # Nested operation was interrupted first; propagate using its operation identifier.
    raise InterruptedOperationError(
        operation_identifier=nested_operation_error.operation_identifier,
        resources=created_resources,
    ) from nested_operation_error
```

## Creating Explore Operators

Explore operators sample entities from a discovery space and submit them for
measurement. Unlike other operator types, the logic runs inside a **Ray actor**
and requires a class to be implemented.

### Implementation

1. Create a class that subclasses `ado.modules.operators.base.Explore`
2. Decorate it with `@explore_operation`
3. Implement, at least, the following methods:
   - **`operator_metadata()`** — a classmethod returning an `OperatorMetadata`
     instance that describes your operator.
   - **`run()`** — an async method containing your operator logic
   - **`onUpdate()`**, **`onCompleted`** and **`onError`** - methods that handle
     notifications about completed measurements

A simple example is show below:

```python
import asyncio
import pydantic
from ado.core.datacontainer.resource import DataContainer
from ado.core.datacontainer.resource import DataContainerResource
from ado.core.operation.config import DiscoveryOperationEnum, OperatorMetadata
from ado.core.operation.operation import OperationOutput
from ado.core.operation.resource import (
    OperationExitStateEnum,
    OperationResourceEventEnum,
    OperationResourceStatus,
)
from ado.modules.operators.base import Explore, measure_or_replay
from ado.modules.operators.collections import explore_operation


class MySearchParameters(pydantic.BaseModel):
    num_entities: int = 10


@explore_operation
class MySearchOperator(Explore):
    def __init__(
        self,
        operationActorName,
        namespace,
        discovery_space_manager,
        actuators,
        params=None,
    ):
        self.params = MySearchParameters(**(params or {}))
        # Queue for completed-measurement notifications received via onUpdate
        self.completed_measurements_queue = asyncio.Queue()
        super().__init__(
            operationActorName=operationActorName,
            namespace=namespace,
            discovery_space_manager=discovery_space_manager,
            actuators=actuators,
        )

    @classmethod
    def operator_metadata(cls) -> OperatorMetadata:
        return OperatorMetadata(
            name="my_search",
            version="0.1.0",
            description="A minimal example explore operator.",
            configuration_model=MySearchParameters,
            example_configuration=MySearchParameters(),
            type=DiscoveryOperationEnum.EXPLORE,
        )

    # --- callbacks from DiscoverySpaceManager --------------------------------
    # onUpdate: called when a measurement completes
    # onError:  called on an unrecoverable error

    def onUpdate(self, measurementRequest) -> None:
        self.completed_measurements_queue.put_nowait(measurementRequest)

    def onCompleted(self) -> None:
        pass

    def onError(self, error: Exception) -> None:
        self.completed_measurements_queue.put_nowait(error)

    # -------------------------------------------------------------------------

    async def run(self) -> OperationOutput | None:
        measurement_queue = await self.ds_manager.measurement_queue.remote()
        ds = await self.ds_manager.discoverySpace.remote()
        experiments = ds.measurementSpace.independentExperiments

        error_message = ""

        # Sample entities and submit them for measurement
        submitted = 0
        async for entities in ...:  # use your chosen sampling strategy
            for experiment in experiments:
                request_ids = measure_or_replay(
                    requestIndex=submitted,
                    requesterid=self.operationIdentifier(),
                    entities=entities,
                    experimentReference=experiment.reference,
                    actuators=self.actuators,
                    measurement_queue=measurement_queue,
                    memoize=False,
                )
                submitted += len(request_ids)

        # Wait for all submitted measurements to complete
        completed = 0
        while not error_message and completed < submitted:
            item = await self.completed_measurements_queue.get()
            if isinstance(item, Exception):
                error_message = f"Discovery space manager error: {item}"
                break
            if item.operation_id == self.operationIdentifier():
                completed += 1

        self.ds_manager.unsubscribeFromUpdates.remote(subscriberName=self.actorName)

        if error_message:
            return OperationOutput(
                exitStatus=OperationResourceStatus(
                    event=OperationResourceEventEnum.FINISHED,
                    exit_state=OperationExitStateEnum.FAIL,
                    message=error_message,
                )
            )
        # exitStatus defaults to success when omitted
        summary = DataContainer(data={"entities_submitted": submitted})
        return OperationOutput(resources=[DataContainerResource(config=summary)])
```

### Tips

**Submit measurements in batches.** Submitting a batch at once, then waiting for
all of them to complete before sampling the next batch, is the simplest pattern.
A more advanced approach is to submit the next entity as soon as one measurement
finishes (continuous batching), which keeps actuators busy and reduces idle
time.

**Use `measure_or_replay` for all submissions.** Do not call actuators directly.
`measure_or_replay` handles memoisation (reusing a prior measurement result when
`memoize=True`) and routes the request to the correct actuator.

**Use `self.operationIdentifier()` as the `requesterid`.** The update
notifications you receive via `onUpdate` include the `operation_id` that created
the request. Filtering on
`measurement_request.operation_id == self.operationIdentifier()` lets you ignore
notifications from other concurrent operations sharing the same space.

**Unsubscribe before returning.** Call
`self.ds_manager.unsubscribeFromUpdates.remote(subscriberName=self.actorName)`
at the end of `run()` as a courtesy — it stops the `DiscoverySpaceManager` from
dispatching further `onUpdate` and `onCompleted` calls to an operator that has
already finished.

### Error handling

**Errors from `measure_or_replay`.** The function raises `KeyError` (no actuator
can handle the experiment) or `MeasurementError` (experiment is deprecated for
the actuator version in use). These don't have to be caught as they are handled
by `ado`. It records the operation with exit state `ERROR` including the full
exception message. You only need to catch them explicitly if you want finer
control.

**Errors from the discovery space manager.** If the discovery space manager
encounters an unrecoverable problem it calls `onError`. This arrives
asynchronously and without explicit handling run() could wait forever for a new
measurement result to arrive. A recommended pattern to handle this is shown in
the example above: `onError` puts the exception onto the same `asyncio.Queue`
that `onUpdate` uses for completed measurements. In the wait loop, check whether
the item dequeued is an `Exception` to detect this sentinel and exit early, then
return a failed `OperationOutput`.

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
