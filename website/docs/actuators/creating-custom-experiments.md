<!-- markdownlint-disable-next-line first-line-h1 -->
> [!NOTE]
>
> For a fully working example, see
> [search a space with an optimizer](../examples/best-configuration-search.md)

`ado` enables you to use Python functions as experiments by registering
 them as custom experiments using a decorator.

## The structure of a custom experiment package

Your custom experiment should be in a standard Python package e.g.

```text
$YOUR_REPO_NAME/
  pyproject.toml
  my_custom_experiment/ # Change to whatever name you like
    __init__.py
    # Python file with your decorated function(s) - can have any name
    experiments.py  
```

In addition, you must register an entry-point to the group `ado.custom_experiments`
in your `pyproject.toml` so `ado` can find your custom_experiment automatically:

```toml
[project.entry-points."ado.custom_experiments"]
#This should be python file with your decorated function(s).
my_experiment = "my_custom_package.experiments" 
```

>[!NOTE]
>
> 1. You can have more than one decorated function in a module.
> 2. If you want to have functions in different modules you
> need to register each module as an entrypoint.

## Decorating your custom experiment function

**To define a custom experiment, decorate your function with `@custom_experiment`.**

In the simplest case:

- type the parameters (using python `typing`)
- return the output in a dictionary of key value pairs
- define the keys of this dictionary in the `output_property_identifiers`
parameter of the decorator

```python
from typing import Dict, Any
from orchestrator.modules.actuators.custom_experiments import custom_experiment


@custom_experiment(
    output_property_identifiers=["density"]
)
def calculate_density(mass: float, volume: float) -> Dict[str, Any]:
    density_value = mass / volume if volume else None
    return {"density": density_value}
```

**Experiment Naming:**  

The experiment will be registered with the name of the decorated
Python function (e.g., `calculate_density`).

**Required Properties:**

Each positional parameter in the signature will become a
required property.

**Return Value:**

The function must return a dictionary whose keys are output names
e.g. "density" above, and the value is the measured value.

**Property Domains:**

`ado` will infer the domains of your positional (non-keyword) inputs as follows:

- floats -> continuous domain over the real numbers
- ints -> discrete domain over the integers
- literal -> categorical domain whose values are the literal values

> [!IMPORTANT]
>
> If a positional parameter has a different type to above e.g.string
> `ado` cannot automatically determine a domain and you will get an
> exception on trying to use the function.
> In this case see
> [define the domain of input parameters](#defining-the-domains-of-required-properties)

### Keyword parameters and optional properties

Keyword parameters in your function signature will be converted to
optional properties of the custom experiment.
The `parameterization` for the optional properties is the value of
keyword in the signature.

The domain inference rules are the same as given above with one addition,
types other than float,int and literal, are assigned an open categorical domain
with a single "known" value, the keyword parameters default.

---

## Using your custom experiment

### Adding your custom experiment to `ado`

To add your experiments to `ado`:

1. Install your package (e.g. `pip install -e .` in your package’s root).
2. Run:

   ```shell
   ado describe actuators --details
   ```

All custom experiments are made available in `ado` through
the special actuator called `custom_experiments`.
Your experiment will be listed under the `custom_experiments` actuator
using the function's name.

### Testing your custom experiment

You can test your custom experiment
using the [`run_experiment`](run_experiment.md) command line tool.
Save the following YAML to a file `point.yaml`

```yaml
entity:
  mass:8
  volume:4
experiments:
- actuatorIdentifier: custom_experiments
  experimentIdentifier: calculate_density
```

then execute:

```commandline
run_experiment point.yaml
```

### Using your custom experiment in a `discoveryspace`

To use a custom experiment in `discoveryspace`
you specify it in its `measurementspace` - exactly like other experiments.

Here is a toy example using the `calculate_density` custom experiment
defined above:

```yaml
sampleStoreIdentifier: dfe035
entitySpace:
- identifier: mass
  propertyDomain:
    domainRange: [1,10]
    interval: 1
- identifier: volume
  propertyDomain:
    domainRange: [1,10]
experiments:
- actuatorIdentifier: custom_experiments
  experimentIdentifier: calculate_density
```

---

## Advanced configuration of custom experiments

The simplest case described in
[decorating you custom experiment function](#decorating-your-custom-experiment-function)
is enough to get started with a custom experiment.
However, if your function has particular types or if you want to refine
domain information you need to access more advanced features of the decorator.

### Defining the domains of required properties

Python functions don't carry any domain information so in many cases
the domain inferred from the type will be too broad.
In this case you can define the domains explicitly in the decorator.

> [!IMPORTANT]
>
> Once you define one required property explicitly you must define them all explicitly.

Defining the domain explicitly enables:

- Better input validation when creating `spaces`
- Automated construction of relevant discovery spaces (via `ado template`)
- Control of what are considered required and optional properties
- Finer grained control of the domain (e.g. you can have a float
parameter but make the domain discrete)

In the following example, we explicitly indicate that the mass and volume parameters
of our `calculate_density` function are  positive numbers.

```python
from typing import Dict, Any
from orchestrator.modules.actuators.custom_experiments import custom_experiment
from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum
from orchestrator.schema.property import ConstitutiveProperty

mass = ConstitutiveProperty(
    identifier="mass",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[1, 100]
    )
)
volume = ConstitutiveProperty(
    identifier="volume",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[1, 100]
    )
)


@custom_experiment(
    required_properties=[mass, volume],
    output_property_identifiers=["density"]
)
def calculate_density(mass, volume) -> Dict[str, Any]:
    density_value = mass / volume if volume else None
    return {"density": density_value}
```

> [!NOTE]
>
> Every non-keyword parameter in your python function is **required**.
> However, you can make any keyword parameter required by adding it to
> required_properties parameter of the decorator

### Defining the domains of optional properties

Similarly to required properties you can define domains for
the **optional properties** via the `optional_properties` parameter to the decorator.
This is also a list of `ConstitutiveProperty` instances
which define the parameters domains.
Default values for the optional properties must be given either in the function signature
i.e. as keyword args, or via the `parameterization` parameter to the decorator.

> [!IMPORTANT]
>
> Once you define one optional property explicitly you must define them
> all explicitly.
> Similarly once you define the parameterization of an optional
> property explicitly you must define the all explicitly.

```python

round_result = ConstitutiveProperty(
    identifier="round_result",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.BINARY_VARIABLE_TYPE,
    )
)

@custom_experiment(
    required_properties=[mass, volume],
    #round_result will get its default value from the keyword arg
    optional_properties=[round_result], 
    output_property_identifiers=["density"],
    metadata={"description": "Calculates density from mass and volume"}
)
def calculate_density(mass, volume, round_result: bool = False):
    density_value = mass / volume if volume else None
    if round_result and density_value is not None:
        density_value = round(density_value, 2)
    return {"density": density_value}
```

The above registers `round_result` as an optional properties
of the experiment,  with its value in the function signature as the default parameterization.

### Supplying metadata

You can also supply a `metadata` dictionary to the "metadata" parameter of
the decorator.
Use this to record experiment-level documentation, categories, etc.
This is illustrated in the above example.

---

## Using your decorated function in code

The decorated function is wrapped to take `ado` internal
data structures, and you would not typically need to
call it directly. However, the decorated experiment function is
still regular Python and can be called:

```python
# Access the original function (undecorated)
original = calculate_density._original_func
print(original(8, 4))  # {'density': 2}
```

## Next Steps

See  
[search a space with an optimizer](../examples/best-configuration-search.md)
for a complete practical workflow using custom experiments.
