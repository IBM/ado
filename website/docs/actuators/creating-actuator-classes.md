<!-- markdownlint-disable-next-line first-line-h1 -->
!!! info end

    A complete template actuator can be found 
    [here](https://github.com/IBM/ado/tree/main/plugins/actuators/example_actuator).
    This example actuator is functional out-of-the-box 
    and can be used as the basis to create new actuators.

Developers can write their own [actuator](../core-concepts/actuators.md) plugins
to add new experiments (a.k.a. tests, experiment protocols) in new domains to
`ado`. Actuator plugins are written in Python and can live in their own
repository.

The main part of writing an actuator plugin is writing a Python class that
implements a specific interface. `ado` then interacts with your plugin, and the
experiments it provides, via an instance of this class.

This page gives an overview of how to get started creating your own actuator.
It's not intended to be comprehensive. After reading this page the best resource
is to check
[our example actuator](https://github.com/IBM/ado/tree/main/plugins/actuators/example_actuator)
or to check an existing actuator plugin.

## Knowledge required

- Knowledge of Python
- Knowledge of [pydantic](https://docs.pydantic.dev/latest/) is useful, but not
  necessary

## Actuator plugin package structure

To create an actuator plugin you **must** use the following package structure

<!-- markdownlint-disable-next-line code-block-style -->
```text
$YOUR_REPO_NAME
├── ado_actuators # This is ado's namespaced package for actuator plugins
│   └── $YOUR_PLUGIN_PACKAGE        # Your plugin
│       ├── __init__.py
│       ├── actuator_definitions.yaml
│       └── ...
└── pyproject.toml
```

The above is structure creates a Python `namespace` package. In this case the
namespace package is called "ado_actuators", which is the namespace for `ado`
plugins. Namespace packages allow developers to independently create and
distribute Python modules that will be installed under a common package name.

When you `pip install` the above package `ado` will detect it when its next run.
If you want to import the installed package in e.g. the Python console you use

<!-- markdownlint-disable-next-line code-block-style -->
```python
import ado_actuators.$YOUR_PLUGIN_NAME
```

!!! warning end

    *NOTE*: Do not place an `__init__.py` under `ado_actuators/` - 
    this will overwrite all installed plugins.

!!! info end

    You can have multiple plugins under `ado_actuators` in $YOUR_REPO_NAME above.
    When you install your package all the plugins will be installed.

### pyproject.toml

The `pyproject.toml` file for an actuator plugin should contain the following fields

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line code-block-style -->
```toml
[build-system]
requires = ["setuptools", "setuptools_scm"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
include-package-data = true # This is on by default, including it for clarity

[tool.setuptools_scm]

[tool.setuptools.packages.find]
where = ["."]

[tool.setuptools.package-data]
# Note: This is optional.
# If you don't specify every non Python file that's in SCM will be added
robotic_lab_actuator = [
    "actuator_definitions.yaml", # Required: The file that describes the actuator classes the plugin provides
    "experiments.yaml" # Optional file that contains definitions for experiment catalog
]

[project]
name="robotic_lab" # Change to your preferred name, along with the actual package
description="A template for creating an actuator" # Change to describing your actuator
dependencies=[
    "black"
]
dynamic = ["version"]
```
<!-- markdownlint-enable line-length -->

## The Actuator Class

Your actuator plugin package must contain at least one class that is a subclass
of `orchestrator.modules.actuators.base.ActuatorBase`. Each of these subclasses
is an interface to a set of experiments (or tests)

The subclass has to implement two methods:

- `catalog`: This returns an
  `orchestrator.modules.actuators.catalog.ExperimentCatalog` instance detailing
  the experiments your actuator provides.
- `submit`: This is an `async` method that `ado` will call to run an Experiment
  on an Entity.

In addition, the case must be decorated with `@ray.remote`

A sketch example:

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line code-block-style -->
```python
import orchestrator.modules.actuators.base
from orchestrator.schema.entity import Entity
from orchestrator.schema.experiment import Experiment
from orchestrator.modules.actuators.catalog import ExperimentCatalog

class MyActuator(orchestrator.modules.actuators.base):

  async def submit(self, entities: [Entity], experiment: Experiment) -> list[str]: #Returns a list of identifiers for the created experiments
    ...

  def catalog(self, **kwargs) -> ExperimentCatalog:
    pass
```
<!-- markdownlint-enable line-length -->

### Telling ado about your actuator class(es)

Actuator plugins must include a file called `actuator_definitions.yaml` that is
installed with the plugin. This file lists all the actuator classes that are
available in the plugin.

An example:

<!-- markdownlint-disable-next-line code-block-style -->
```yaml
- module:
    moduleClass: MyActuator
    moduleName: ado_actuators.myplugin.actuators
```

### What an actuator is expected to do on `submit`

The key method of an `actuator` is the `submit` method as this is what runs an
experiment. On a call to this method three things are expected to happen in the
Actuator:

- One or more `MeasurementRequest` instances are created representing an
  execution of the experiment that was requested
  - `One or more` as the actuator can launch a separate experiment for each
    entity or one for them all. Which method is used depends on developer choice
- Launch the experiment asynchronously and return the `MeasurementRequest`
  identifier(s)
  - i.e. It is expected that the `submit` method will return almost immediately
    and the requested experiments will be executed asynchronously
- When an experiment has finished
  - Add the results to the relevant `MeasurementRequest` instance
  - Put the `MeasurementRequest` on the `MeasurementQueue` that was provided to
    the actuator on `__init__`

From the `submit` callers point of view this means:

1. It expects to immediately get back a set of strings that are
   MeasurementRequest ids
2. At some later time it will find MeasurementRequests with these ids on the
   `MeasurementQueue` containing the experiment results.

For everything else the actuator developer is free to implement as they want.

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

<!-- markdownlint-disable-next-line code-block-style -->
```python
from orchestrator.core.actuatorconfiguration.config import GenericActuatorParameters
from orchestrator.modules.actuators.base import ActuatorBase
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


@ray.remote
class Actuator(ActuatorBase):
    identifier = "my_actuator"
    parameters_class = InferenceActuatorParameters
```

### Example custom configurations

Users can obtain an example configuration for your actuator using:

<!-- markdownlint-disable-next-line code-block-style -->
```commandline
ado template actuatorconfiguration --actuator-identifier $YOUR_ACTUATOR_ID`
```

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
<!-- markdownlint-disable-next-line code-block-style -->
```python
authToken: typing.Annotated[
    str,
    pydantic.Field(
        description="The token to access the inference service",
        validate_default=True,  # <--- This will check if the value is None and raise an error if it is i.e. if the example value was not changed
    ),
] = None  # <--- value that will be written for examples. It is actually invalid
```
<!-- markdownlint-enable line-length -->

If you have no required fields, you may want `ado` to validate your default
values before outputting them. This is useful for e.g. tests, to ensure there
isn't an error with the defaults. To do this you can override the
`default_parameters` method in your Actuator to turn validation on e.g.

<!-- markdownlint-disable-next-line code-block-style -->
```python
@override
def default_parameters(self) -> GenericActuatorParameters:
    return MyActuatorParams()
```

### Using custom ActuatorConfiguration parameters

Once users have set the relevant values for your actuator in a YAML file they
can create an `actuatorconfiguration` resource from them

<!-- markdownlint-disable-next-line code-block-style -->
```commandline
ado create actuatorconfiguration -f $FILLED_IN_TEMPLATE
```

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

<!-- markdownlint-disable-next-line code-block-style -->
```python
from orchestrator.core.actuatorconfiguration.config import GenericActuatorParameters
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

To enable upgrading of the previous model versions when fields are being
deprecated, we recommended using a
[Pydantic Before Model Validator](https://docs.pydantic.dev/latest/concepts/validators/#model-before-validator).
This allows the dictionary content of the model to be changed as appropriate
before validation is applied. To ensure the users are aware of the change, we
will also use the `warn_deprecated_actuator_parameters_model_in_use` method in
the validator:

<!-- markdownlint-disable-next-line code-block-style -->
```python
from typing import Annotated, Any

import pydantic

from orchestrator.core.actuatorconfiguration.config import GenericActuatorParameters

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
    def rename_authToken(cls, values: Any):

        # We expect either a GenericActuatorParameters or a dict instance
        if not isinstance(values, GenericActuatorParameters) and not isinstance(
            values, dict
        ):
            raise ValueError(f"Unexpected type {type(values)} in validator")

        from orchestrator.core.actuatorconfiguration.config import (
            warn_deprecated_actuator_parameters_model_in_use,
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
            if old_key not in values:
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
            if new_key in values:
                values.pop(old_key)
            # Set the old value in the
            # new field
            else:
                values[new_key] = values.pop(old_key)

        return values
```

When a model with the old field is loaded, the user will see the following
warning:

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line code-block-style -->
```text
WARN:   The parameters for the my_actuator actuator have been updated as of my_actuator v2.
        They are being temporarily auto-upgraded to the latest version.
        This behaviour will be removed with my_actuator v3.
HINT:   Run ado upgrade actuatorconfigurations to upgrade the stored actuatorconfigurations.
        Update your actuatorconfiguration YAML files to use the latest format: https://example.com
```
<!-- markdownlint-enable line-length -->

### Updating a field in your actuator's configuration without deprecating it

Let's imagine we want to change the type of the `endpoint` field to be
`pydantic.HttpUrl`. The model for our actuator v2 would then be:

<!-- markdownlint-disable-next-line code-block-style -->
```python
from orchestrator.core.actuatorconfiguration.config import GenericActuatorParameters
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

<!-- markdownlint-disable-next-line code-block-style -->
```python
from orchestrator.core.actuatorconfiguration.config import GenericActuatorParameters
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
        from orchestrator.core.actuatorconfiguration.config import (
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

When a model using `str`s will be loaded, the user will see the following
warning:

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line code-block-style -->
```text
WARN:   The parameters for the my_actuator actuator have been updated as of my_actuator v1.
        They are being temporarily auto-upgraded to the latest version.
        This behavior will be removed with my_actuator v2.
HINT:   Run ado upgrade actuatorconfigurations to upgrade the stored actuatorconfigurations.
        Update your actuatorconfiguration YAML files to use the latest format: https://example.com
```
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
<!-- markdownlint-disable-next-line code-block-style -->
```python
from orchestrator.modules.operators.orchestrate import CLEANER_ACTOR, ResourceCleaner
import ray
...
try:
    cleaner_handle = ray.get_actor(name=CLEANER_ACTOR)
    cleaner_handle.add_to_cleanup.remote(handle='your actor handle')
except Exception as e:
    print(f"Failed to register custom actors for clean up {e}. Make sure you clean it up")
```
<!-- markdownlint-enable line-length -->

Once the registration is in place, the `cleanup` method of this actor is invoked
at the end of execution

## Signaling progress from your actuator

Actuator developers can provide rich, real-time progress output
to users running experiments, using utilities available in
`orchestrator.modules.operators.console_output.py`.
This is critical for long-running operations (such as deployment,
environment setup, or benchmarking),
and helps users visually associate progress with specific requests.

### How progress signaling works

When performing asynchronous tasks inside your actuator
(or its experiment executor),
emit progress or spinner messages to a centralized console queue
using provided Rich message helpers:

- **RichConsoleSpinnerMessage**: Shows an animated spinner with a label
(for things like environment creation or deployment in progress)
- **RichConsoleProgressMessage**: Shows a progress bar reflecting integer percentage
(for measurable steps such as data transfer, job startup, etc)

You should send these messages to the `RichConsoleQueue` actor
and update or stop them when state changes.

!!! tip end

    Use the `request id` of the MeasurementRequest you're operating on 
    as the message `id` (and include it in the message `label`).
    This allows your actuator to support progress for multiple experiments
    running concurrently, and the UI will clearly indicate which progress
    output is tied to which experiment request. 

### Example usage

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line code-block-style -->
```python
from orchestrator.modules.operators.console_output import RichConsoleSpinnerMessage, RichConsoleProgressMessage
# Get the console queue where you post progress messages to show
console = ray.get_actor(name="RichConsoleQueue") 
request_id = request.requestid  # or similar

# Start a spinner
console.put.remote(message=RichConsoleSpinnerMessage(
    id=request_id,
    label=f"({request_id}) Waiting for environment...",
    state="start",
))
# ... do work ...
# Stop the spinner (replace with progress or mark complete)
console.put.remote(message=RichConsoleSpinnerMessage(
    id=request_id,
    label=f"({request_id}) Environment ready.",
    state="stop",
))
# Start a bar showing progress
console.put.remote(message=RichConsoleProgressMessage(
    id=request_id,
    label=f"({request_id}) Uploading data...",
    progress=0,  # percent
))
# ... sleep then calculate how much upload is complete ...
console.put.remote(message=RichConsoleProgressMessage(
    id=request_id,
    label=f"({request_id}) Uploading data...",
    progress=35,  # percent
))
```
<!-- markdownlint-enable line-length -->

---

## Experiment executor

The actuator submit method invokes a Ray remote function `run_experiment`
implemented by an experiment_executor. The actual name of this function and its
parameters can be defined by the actuator implementer. Typically the set of
parameters includes:

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line code-block-style -->
```python
request: MeasurementRequest,  # measurement request
experiment: Union[Experiment, ParameterizedExperiment],  # experiment definition
state_update_queue: orchestrator.modules.actuators.measurement_queue.MeasurementQueue,  # state update queue
```
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
- `orchestrator.utilities.support.dict_to_measurements` - Extract the values
  related to an experiment from a dictionary of measurements and convert to
  PropertyValues
- `orchestrator.utilities.support.create_measurement_result` - Create
  measurement result
- `orchestrator.utilities.support.compute_measurement_status` - Compute
  execution status
- `orchestrator.utilities.async_task_runner.AsyncTaskRunner` - wait for the
  completion of an async function and get execution result
