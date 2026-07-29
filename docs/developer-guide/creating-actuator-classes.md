<!-- markdownlint-disable first-line-h1 -->

>[!INFO]
>
> A [complete template actuator](https://github.com/IBM/ado/tree/main/plugins/actuators/example_actuator)
> is available.
> This example actuator is functional out-of-the-box
> and can be used as the basis to create new actuators.

[Custom experiments](creating-custom-experiments.md) cover many use cases for
extending `ado` with new experiments. However, sometimes you need more control
over how the experiments are run. For example, you might need to connect to,
configure and manage an external environment, like a Kubernetes cluster.

For such situations developers can write their own
[actuators](../concepts/actuators.md). Actuators allow you to control and
customize the entire experiment submission process giving great flexibility and
power. You can also expose customization options to users via
[`actuatorconfigurations`](#enabling-custom-configuration-of-an-actuator).
Like custom experiments actuator are supplied as plugin **python
packages**.

This page gives an overview of how to get started creating your own actuator.
It's not intended to be comprehensive. After reading this page the best resource
is to check
[our example actuator](https://github.com/IBM/ado/tree/main/plugins/actuators/example_actuator)
or to check an existing actuator plugin.

## Knowledge required

- Knowledge of Python
- Knowledge of [pydantic](https://docs.pydantic.dev/latest/) is useful, but not
  necessary

## The Actuator Class

The main part of writing an actuator plugin is writing (at least) one Python
class that implements a specific interface.

- is a subclass of `ado.modules.actuators.base.StandardActuator`.
- defines a class attribute `identifier`, which is human-readable name of the
  actuator
- implements the `catalog()` method
- _either_
  - simple case: override `_experiment_implementations()`
  - complex case: overrides `_get_request_executor`

The simple case is for when your actuators experiments are independent of each
other i.e. executing one experiment does not care about other experiments. The
complex case is for when your experiments require access to shared state.

### The catalog method

The `catalog()` method returns an `ExperimentCatalog` instance detailing the
experiments your actuator provides.

Each entry in the catalog is an `Experiment` model instance that describes the
name, version, input properties and output properties of an experiment. It
**does not contain** the implementation of the experiment.

Our
[example implementation](https://github.com/IBM/ado/tree/main/plugins/actuators/example_actuator)
demonstrates reading the catalog from YAML. The catalog can also be built in
code.

### Simple case: Independent experiments

If your experiments can be executed independently, requiring no shared state,
you can override the method `_experiment_implementations`.

In this case, implement each experiment as a python function i.e. one python
function for each `Experiment` entry your `ExperimentCatalog`. The
`_experiment_implementations()` method then returns a dict that maps each
experiment identifier to the corresponding function e.g.

```python
def _experiment_implementations(self) -> dict[str, Callable[..., dict[str, Any]]]:

    return {"myexperiment": my_experiment_fn()}
```

The parameter names of the function must be the same as the input property
identifiers of the `Experiment`. The output of the function must be a dict that
maps the target property identifiers of the Experiment to their measured values.

For example, for an Experiment instance like

```yaml
# it's properties which  match what is defined here
peptide_mineralization:
  identifier: peptide_mineralization
  actuatorIdentifier: "robotic_lab"
  requiredProperties:
    - identifier: "peptide_identifier"
      propertyDomain:
        variableType: "CATEGORICAL_VARIABLE_TYPE"
        values: ["test_peptide", "test_peptide_new"]
    - identifier: "peptide_concentration"
      propertyDomain:
        values: [0.1, 0.4, 0.6, 0.8]
        variableType: "DISCRETE_VARIABLE_TYPE"
  targetProperties: # What properties experiment will measure
    - identifier: adsorption_timeseries
    - identifier: adsorption_plateau_value
```

The function would look like

```python
def peptide_mineralization_fn(peptide_identifier, peptide_concentration):

    ...
    return {"adsoprtion_timeseries": timeseries, "adsorption_plateau_value": plateau}
```

### Complex case: Experiments with shared state

If your experiments require access to shared state e.g. a queue object, an
environment manager object, then you can override the method
`_get_request_executor`.

This method that takes a `MeasurementRequest` instance that describes the
experiment to run. Note, the `use_ray` parameter is used by default
implementation and can be safely ignored when overridden.

```python
    def _get_request_executor(
        self,
        request: MeasurementRequest,
        use_ray: bool = False,
    ) -> Callable[[], MeasurementRequest]:
```

The `get_request_executor` method must return a zero-argument `Callable` that
executes the requested experiment and returns a completed `MeasurementRequest`
(measurements and status set). The function must be picklable.

Other than that the function returned may do anything. This can include
executing the experiment in ray workers, creating pods, or submitting jobs to
batch-schedulers.

We recommend using `functools.partial` if you want to customize a module-level
function for this purpose. For example, you can define a generic function that
has parameters for experiment description, and certain instance variables. Then
use `functools.partial` to bind values to these parameters and return the
result.

### Executing experiments

You can execute experiments with your Actuator in scripts using the `execute`
method

e.g.

```python
result = actuator.execute(entities,
                        experiment_reference,
                        requesterid: "script",
                        requestIndex: 0,)
```

where `entities` is a list of one or more `Entity` instances representing the
points you want to measure, and `experiment_reference`, is an
`ExperimentReference` instance describing the experiment to execute.
`requesterid` and `requestindex` are tracking information that will be contained
in the returned result.

## The Actuator Plugin Package

### pyproject.toml

The `pyproject.toml` file for an actuator plugin should contain fields similar
to the following:

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable code-block-style -->

```toml
[project]
name = "robotic_lab"  # Change to your preferred name, along with the actual package
description = "A template for creating an actuator"  # Change to describing your actuator
dependencies = [
    "ado-core"
]
dynamic = ["version"]

[project.entry-points."ado.actuators"]
robotic_lab = "robotic_lab_actuator.actuator:RoboticLab"

[build-system]
requires = ["hatchling", "uv-dynamic-versioning>=0.7.0"]
build-backend = "hatchling.build"

[tool.hatch.version]
source = "uv-dynamic-versioning"

[tool.hatch.build.targets.wheel]
packages = ["src/robotic_lab_actuator"]
```

<!-- markdownlint-enable code-block-style -->
<!-- markdownlint-enable line-length -->

### Telling ado about your actuator class(es)

Actuator plugins must register their actuator classes using the `ado.actuators`
entry point in `pyproject.toml`. This allows ado to automatically discover and
load your actuator when the plugin is installed.

This is done via the following lines in the `pyproject.toml`

<!-- markdownlint-disable code-block-style -->

```toml
[project.entry-points."ado.actuators"]
my-actuator = "myplugin.actuators:MyActuator"
```

<!-- markdownlint-enable code-block-style -->

The entry point format is:

- **Entry point name**: : A unique identifier within the ado.actuators group
  (e.g., `my-actuator`)
- **Module path**: The path to your actuator class (e.g.,
  `myplugin.actuators:MyActuator`)

## Enabling custom configuration of an actuator

Actuators may require a custom configuration (i.e., parameters) to be provided.
For example, an actuator calling an inference server can require an endpoint to
connect and its related authorisation token.

`ado` provides this capability through the `GenericActuatorParameters` class,
which allows developers to define a Pydantic model of the parameters expected by
the actuator. This model will be validated at runtime.

To write your own actuator parameters class, simply create a class that inherits
from `GenericActuatorParameters` and add a reference to it in the
`parameters_class` class variable of your Actuator, as such:

<!-- markdownlint-disable code-block-style -->

```python
from ado.core.actuatorconfiguration.config import GenericActuatorParameters
from ado.modules.actuators.base import ActuatorBase
from typing import Annotated
import pydantic


class InferenceActuatorParameters(GenericActuatorParameters):
    model_config = pydantic.ConfigDict(extra="forbid")

    endpoint: Annotated[
        str,
        pydantic.Field(
            description="Endpoint to an inference service",
            validate_default=True,
        ),
    ] = None
    authToken: Annotated[
        str,
        pydantic.Field(
            description="The token to access the inference service",
            validate_default=True,
        ),
    ] = None


class Actuator(ActuatorBase):
    identifier = "my_actuator"
    parameters_class = InferenceActuatorParameters
```

<!-- markdownlint-enable code-block-style -->

### Example custom configurations

Users can obtain an example configuration for your actuator using:

<!-- markdownlint-disable code-block-style -->

```commandline
ado template actuatorconfiguration --actuator-identifier $YOUR_ACTUATOR_ID`
```

<!-- markdownlint-enable code-block-style -->

This example is generated by calling `model_construct()` on your actuator
parameter class. This means

- default values you specify for fields are output
- you need default values for all fields
- the defaults are not validated

This is useful when your configuration has required fields, i.e., you need the
user to supply them and can't set a default value for them. This way, the
generated example template will include those fields, but `ado` will catch any
missing or incorrect values when the user is creating the
`actuatorconfiguration` resource.

For example, you can declare a required field like this

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable code-block-style -->

```python
authToken: typing.Annotated[
    str,
    pydantic.Field(
        description="The token to access the inference service",
        validate_default=True,  # <--- This will check if the value is None and raise an error if it is i.e. if the example value was not changed
    ),
] = None  # <--- value that will be written for examples. It is actually invalid
```

<!-- markdownlint-enable code-block-style -->
<!-- markdownlint-enable line-length -->

If you have no required fields, you may want `ado` to validate your default
values before outputting them. This is useful for e.g. tests, to ensure there
isn't an error with the defaults. To do this you can override the
`default_parameters` method in your Actuator to turn validation on e.g.

<!-- markdownlint-disable code-block-style -->

```python
@override
def default_parameters(self) -> GenericActuatorParameters:
    return MyActuatorParams()
```

<!-- markdownlint-enable code-block-style -->

### Using custom ActuatorConfiguration parameters

Once users have set the relevant values for your actuator in a YAML file they
can create an `actuatorconfiguration` resource from them

<!-- markdownlint-disable code-block-style -->

```commandline
ado create actuatorconfiguration -f $FILLED_IN_TEMPLATE
```

<!-- markdownlint-enable code-block-style -->

The
[actuatorconfiguration resource documentation](../resources/actuatorconfig.md)
contains for more information on how users will create and supply actuator
parameters to your actuator.

### How the custom configuration is stored and output

When storing an instance of your custom configuration model in
[the metastore](../resources/metastore.md), the serialized representation is
obtained using `model_dump_json()` with **no options**.

When outputting for `ado get actuatorconfiguration`, the serialized
representation is also obtained with `model_dump_json()`, and the schema with
`model_json_schema()`. In this case various options to `model_dump_json` or
`model_json_schema` may be used, e.g. `exclude_unset=True`.

When outputting for `ado template actuatorconfiguration`, `model_construct()` is
used by default as described in the previous section.

## How to update your actuator's custom configuration

During development, there will be times when you might need to update the input
parameter model for your actuator, adding, removing or modifying fields. In
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

Let's see a practical example using `MyActuatorParams`. We will consider two
cases:

- We want to deprecate a field.
- We want to apply changes to a field without deprecating it.

### Deprecating a field in your actuator's custom configuration

Let's imagine we want to change the name of the `authToken` field to be
`authorization_token`. The model for our actuator v2 would then be:

<!-- markdownlint-disable code-block-style -->

```python
from ado.core.actuatorconfiguration.config import GenericActuatorParameters
from typing import Annotated
import pydantic


class InferenceActuatorParameters(GenericActuatorParameters):
    model_config = pydantic.ConfigDict(extra="forbid")

    endpoint: Annotated[
        str,
        pydantic.Field(
            description="Endpoint to an inference service",
            validate_default=True,
        ),
    ] = None
    authorization_token: Annotated[
        str,
        pydantic.Field(
            description="The token to access the inference service",
            validate_default=True,
        ),
    ] = None
```

<!-- markdownlint-enable code-block-style -->

To enable upgrading of the previous model versions when fields are being
deprecated, we recommended using a
[Pydantic Before Model Validator](https://docs.pydantic.dev/latest/concepts/validators/#model-before-validator).
This allows the dictionary content of the model to be changed as appropriate
before validation is applied. To ensure the users are aware of the change, we
will also use the `warn_deprecated_actuator_parameters_model_in_use` method in
the validator:

<!-- markdownlint-disable code-block-style -->

```python
from typing import Annotated, Any

import pydantic

from ado.core.actuatorconfiguration.config import GenericActuatorParameters


class InferenceActuatorParameters(GenericActuatorParameters):
    model_config = pydantic.ConfigDict(extra="forbid")

    endpoint: Annotated[
        str,
        pydantic.Field(
            description="Endpoint to an inference service",
            validate_default=True,
        ),
    ] = None
    authorization_token: Annotated[
        str,
        pydantic.Field(
            description="The token to access the inference service",
            validate_default=True,
        ),
    ] = None

    @pydantic.model_validator(mode="before")
    @classmethod
    def rename_authToken(cls, values: Any) -> Any:  # noqa: ANN401

        # We expect either a GenericActuatorParameters or a dict instance
        if not isinstance(values, GenericActuatorParameters) and not isinstance(
            values, dict
        ):
            raise ValueError(f"Unexpected type {type(values)} in validator")

        from ado.core.actuatorconfiguration.config import (
            warn_deprecated_actuator_parameters_model_in_use,
        )
        from ado.utilities.dictionaries import (
            get_nested_value,
            has_nested_field,
            remove_nested_field,
            set_nested_value,
        )

        old_key = "authToken"
        new_key = "authorization_token"

        if isinstance(values, GenericActuatorParameters):
            # The old key is not present - all good
            if not hasattr(values, old_key):
                return values

            # Notify the user that the authToken
            # field is deprecated
            warn_deprecated_actuator_parameters_model_in_use(
                affected_actuator="my_actuator",
                deprecated_from_actuator_version="v2",
                removed_from_actuator_version="v3",
                deprecated_fields=old_key,
                latest_format_documentation_url="https://example.com",
            )

            # The user has set both the old
            # and the new key - the new key
            # takes precedence.
            if hasattr(values, new_key):
                delattr(values, old_key)
            # Set the old value in the
            # new field
            else:
                setattr(values, new_key, getattr(values, old_key))
                delattr(values, old_key)

        else:
            # The old key is not present - all good
            if not has_nested_field(values, old_key):
                return values

            # Notify the user that the authToken
            # field is deprecated
            warn_deprecated_actuator_parameters_model_in_use(
                affected_actuator="my_actuator",
                deprecated_from_actuator_version="v2",
                removed_from_actuator_version="v3",
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

<!-- markdownlint-enable code-block-style -->

When a model with the old field is loaded, the user will see the following
warning:

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable code-block-style -->

```text
WARN:   The parameters for the my_actuator actuator have been updated as of my_actuator v2.
        They are being temporarily auto-upgraded to the latest version.
        This behaviour will be removed with my_actuator v3.
HINT:   Run ado upgrade actuatorconfigurations to upgrade the stored actuatorconfigurations.
        Update your actuatorconfiguration YAML files to use the latest format: https://example.com
```

<!-- markdownlint-enable code-block-style -->
<!-- markdownlint-enable line-length -->

### Updating a field in your actuator's configuration without deprecating it

Let's imagine we want to change the type of the `endpoint` field to be
`pydantic.HttpUrl`. The model for our actuator v2 would then be:

<!-- markdownlint-disable code-block-style -->

```python
from ado.core.actuatorconfiguration.config import GenericActuatorParameters
from typing import Annotated
import pydantic


class InferenceActuatorParameters(GenericActuatorParameters):
    model_config = pydantic.ConfigDict(extra="forbid")

    endpoint: Annotated[
        pydantic.HttpUrl,
        pydantic.Field(
            description="Endpoint to an inference service",
            validate_default=True,
        ),
    ] = None
    authToken: Annotated[
        str,
        pydantic.Field(
            description="The token to access the inference service",
            validate_default=True,
        ),
    ] = None
```

<!-- markdownlint-enable code-block-style -->

To enable upgrading of the previous model versions when fields are not being
deprecated, we recommended using a
[Pydantic Before Field Validator](https://docs.pydantic.dev/latest/concepts/validators/#field-before-validator).
This allows the specific field to be changed as appropriate before validation is
applied. To ensure the users are aware of the change, we will also use the
`warn_deprecated_actuator_parameters_model_in_use` method in the validator:

> [!NOTE]
>
> The method being called is the same as the one for
> [warning about deprecated fields](#deprecating-a-field-in-your-actuators-custom-configuration),
> but we omit the `deprecated_fields` parameter.

<!-- markdownlint-disable code-block-style -->

```python
from ado.core.actuatorconfiguration.config import GenericActuatorParameters
from typing import Annotated
import pydantic


class InferenceActuatorParameters(GenericActuatorParameters):
    model_config = pydantic.ConfigDict(extra="forbid")

    endpoint: Annotated[
        pydantic.HttpUrl,
        pydantic.Field(
            description="Endpoint to an inference service",
            validate_default=True,
        ),
    ] = None
    authToken: Annotated[
        str,
        pydantic.Field(
            description="The token to access the inference service",
            validate_default=True,
        ),
    ] = None

    @pydantic.field_validator("endpoint", mode="before")
    @classmethod
    def convert_endpoint_to_url(cls, value: str | pydantic.HttpUrl):
        from ado.core.actuatorconfiguration.config import (
            warn_deprecated_actuator_parameters_model_in_use,
        )

        if isinstance(value, str):
            # Notify the user that the parameters of my_actuator
            # have been updated
            warn_deprecated_actuator_parameters_model_in_use(
                affected_actuator="my_actuator",
                deprecated_from_actuator_version="v2",
                removed_from_actuator_version="v3",
                latest_format_documentation_url="https://example.com",
            )
            value = pydantic.HttpUrl(value)

        return value
```

<!-- markdownlint-enable code-block-style -->

When a model using `str`s will be loaded, the user will see the following
warning:

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable code-block-style -->

```text
WARN:   The parameters for the my_actuator actuator have been updated as of my_actuator v1.
        They are being temporarily auto-upgraded to the latest version.
        This behavior will be removed with my_actuator v2.
HINT:   Run ado upgrade actuatorconfigurations to upgrade the stored actuatorconfigurations.
        Update your actuatorconfiguration YAML files to use the latest format: https://example.com
```

<!-- markdownlint-enable code-block-style -->
<!-- markdownlint-enable line-length -->

## Ensure actuator cleanup

An actuator implementation can create resources that need to be cleaned up at
execution completion. Two options are provided for doing this:

### Python [atexit](https://docs.python.org/3/library/atexit.html) based cleanup

The `atexit` module defines functions to register and unregister cleanup
functions. Functions thus registered are automatically executed upon normal
interpreter termination. atexit runs these functions in the reverse order in
which they were registered; if you register A, B, and C, at interpreter
termination time they will be run in the order C, B, A. This method works well
for clean up resources used by the actuator implementation itself, but not for
cleaning up resources created by custom Ray actors created by the actuators.

### Custom Ray actors cleanup

This option uses a
[named detached actor](https://docs.ray.io/en/latest/ray-core/actors/named-actors.html).
This actor is started in the Ray namespace of the `operation` using the actuator
with the name of `resource_cleaner` and can be used by any custom actor
implementing `cleanup` method.

To ensure the cleanup actor has been created when you retrieve it, the safest
approach is to only access it within your actuator class implementation or
actors that were directly created by it.

Below is an example of registering a custom class for cleanup:

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable code-block-style -->

```python
from ado.modules.operators.orchestrate import CLEANER_ACTOR, ResourceCleaner
import ray

...
try:
    cleaner_handle = ray.get_actor(name=CLEANER_ACTOR)
    cleaner_handle.add_to_cleanup.remote(handle="your actor handle")
except Exception as e:
    print(
        f"Failed to register custom actors for clean up {e}. Make sure you clean it up"
    )
```

<!-- markdownlint-enable code-block-style -->
<!-- markdownlint-enable line-length -->

Once the registration is in place, the `cleanup` method of this actor is invoked
at the end of execution

## Signaling progress from your actuator

Actuator developers can provide rich, real-time progress output to users running
experiments, using utilities available in
`ado.modules.operators.console_output.py`. This is critical for
long-running operations (such as deployment, environment setup, or
benchmarking), and helps users visually associate progress with specific
requests.

### How progress signaling works

When performing asynchronous tasks inside your actuator (or its experiment
executor), emit progress or spinner messages to a centralized console queue
using provided Rich message helpers:

- **RichConsoleSpinnerMessage**: Shows an animated spinner with a label (for
  things like environment creation or deployment in progress)
- **RichConsoleProgressMessage**: Shows a progress bar reflecting integer
  percentage (for measurable steps such as data transfer, job startup, etc)

You should send these messages to the `RichConsoleQueue` actor and update or
stop them when state changes.

> [!INFO]
> Use the `request id` of the MeasurementRequest you're operating on
> as the message `id` (and include it in the message `label`).
> This allows your actuator to support progress for multiple experiments
> running concurrently, and the UI will clearly indicate which progress
> output is tied to which experiment request.

### Example usage

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable code-block-style -->

```python
from ado.modules.operators.console_output import (
    RichConsoleSpinnerMessage,
    RichConsoleProgressMessage,
)

# Get the console queue where you post progress messages to show
console = ray.get_actor(name="RichConsoleQueue")
request_id = request.requestid  # or similar

# Start a spinner
console.put.remote(
    message=RichConsoleSpinnerMessage(
        id=request_id,
        label=f"({request_id}) Waiting for environment...",
        state="start",
    )
)
# ... do work ...
# Stop the spinner (replace with progress or mark complete)
console.put.remote(
    message=RichConsoleSpinnerMessage(
        id=request_id,
        label=f"({request_id}) Environment ready.",
        state="stop",
    )
)
# Start a bar showing progress
console.put.remote(
    message=RichConsoleProgressMessage(
        id=request_id,
        label=f"({request_id}) Uploading data...",
        progress=0,  # percent
    )
)
# ... sleep then calculate how much upload is complete ...
console.put.remote(
    message=RichConsoleProgressMessage(
        id=request_id,
        label=f"({request_id}) Uploading data...",
        progress=35,  # percent
    )
)
```

<!-- markdownlint-enable code-block-style -->
<!-- markdownlint-enable line-length -->

---

## Experiment executor

The actuator submit method invokes a Ray remote function `run_experiment`
implemented by an experiment_executor. The actual name of this function and its
parameters can be defined by the actuator implementer. Typically, the set of
parameters includes:

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable code-block-style -->

```python
request: MeasurementRequest,  # measurement request
experiment: Union[Experiment, ParameterizedExperiment],  # experiment definition
state_update_queue: ado.modules.actuators.measurement_queue.MeasurementQueue,  # state update queue
```

<!-- markdownlint-enable code-block-style -->
<!-- markdownlint-enable line-length -->

<!-- markdownlint-enable line-length -->

Any additional parameters can be added to these, as required for actuator
implementation

Implementation of `run_experiment` does the following:

1. For each Entity in the request it retrieves the values required to run the
   experiment
2. Run experiment with the retrieved entities
3. Create a MeasurementResult to hold the results
4. Compute the overall request status
5. Put completed request to the `state_update_queue`

### Helper functions for Experiment executor

To simplify Experiment executor implementation, we provide several helper
functions and methods:

- `Experiment.propertyValuesFromEntity` - Get the input values for the
  experiment based on the entity and the experiment definition
- `ado.utilities.support.observed_property_values_from_dict` - Extract
  the values related to an experiment from a dictionary of measurements and
  convert to PropertyValues
- `ado.utilities.support.create_measurement_result` - Create
  measurement result
- `ado.utilities.support.compute_measurement_status` - Compute
  execution status
- `ado.utilities.async_task_runner.AsyncTaskRunner` - wait for the
  completion of an async function and get execution result
