---
name: create-custom-experiment
description: Create and test custom experiments in ado. Covers package structure, decorator usage, installation, and testing via run_experiment API and discoveryspace dry-run.
---

# Create Custom Experiment

Custom experiments enable using Python functions as experiments in ado by
registering them with the `@custom_experiment` decorator.

## Package Structure

Standard Python package with entry-point registration:

```text
$YOUR_REPO_NAME/
  pyproject.toml
  my_custom_experiment/
    __init__.py
    experiments.py  # Decorated function(s)
```

## pyproject.toml Configuration

Required entry-point in `pyproject.toml`:

```toml
[project]
name = "my_custom_experiment"
description = "Custom experiment description"
dependencies = []  # Add any dependencies
dynamic = ["version"]

[project.entry-points."ado.custom_experiments"]
my_experiment = "my_custom_experiment.experiments"

[build-system]
requires = ["hatchling", "uv-dynamic-versioning>=0.7.0"]
build-backend = "hatchling.build"

[tool.hatch.build]
include = ["my_custom_experiment/**"]

[tool.hatch.build.targets.wheel]
only-include = ["my_custom_experiment"]

[tool.hatch.version]
source = "uv-dynamic-versioning"

[tool.uv-dynamic-versioning]
vcs = "git"
style = "pep440"
pattern = "default-unprefixed"
fallback-version = "0.0.0"
tagged-metadata = true
dirty = true
bump = true
```

## Decorator Usage

### Simple Example

```python
from typing import Any
from orchestrator.modules.actuators.custom_experiments import custom_experiment

@custom_experiment(output_property_identifiers=["density"])
def calculate_density(mass: float, volume: float) -> dict[str, Any]:
    density_value = mass / volume if volume else None
    return {"density": density_value}
```

**Key points:**

- Function name becomes experiment identifier
- Positional parameters → required properties
- Return dict with keys matching `output_property_identifiers`
- Type annotations determine domain inference:
  - `float` → continuous domain
  - `int` → discrete domain
  - `Literal` → categorical domain

### Advanced Example with Explicit Domains

```python
from orchestrator.modules.actuators.custom_experiments import custom_experiment
from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum
from orchestrator.schema.property import ConstitutiveProperty

@custom_experiment(
    required_properties=[
        ConstitutiveProperty(
            identifier="x0",
            propertyDomain=PropertyDomain(
                variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
                domainRange=[-5, 5]
            )
        ),
        ConstitutiveProperty(
            identifier="x1",
            propertyDomain=PropertyDomain(
                variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
                domainRange=[-5, 5]
            )
        )
    ],
    optional_properties=[
        ConstitutiveProperty(
            identifier="num_iterations",
            propertyDomain=PropertyDomain(
                variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
                domainRange=[1, 100],
                interval=1
            )
        )
    ],
    parameterization={"num_iterations": 10},
    output_property_identifiers=["result"],
    metadata={"description": "Example optimization function"}
)
def my_experiment(x0: float, x1: float, num_iterations: int) -> dict[str, Any]:
    # Implementation
    result = x0**2 + x1**2
    return {"result": result}
```

## Installation

Install in editable mode from package root:

```bash
# From the directory containing pyproject.toml
uv pip install -e .
```

Verify installation:

```bash
uv run ado get experiments
```

Custom experiment appears under `custom_experiments` actuator.

## Testing

### Method 1: run_experiment API

Create test point YAML (`point.yaml`):

```yaml
entity:
  mass: 10.0
  volume: 2.0
experiments:
  - actuatorIdentifier: custom_experiments
    experimentIdentifier: calculate_density
```

Execute:

```bash
uv run run_experiment point.yaml
```

**Options:**

- `--remote <ENDPOINT>`: Execute via REST API
- `--timeout <SECONDS>`: Timeout for remote execution (default: 300)
- `--no-validate`: Skip entity validation
- `--actuator-configuration-id <ID>`: Specify actuator configuration

### Method 2: discoveryspace Dry-Run

Create space YAML (`space.yaml`):

```yaml
entitySpace:
  - identifier: mass
    propertyDomain:
      variableType: CONTINUOUS_VARIABLE_TYPE
      domainRange: [1, 10]
  - identifier: volume
    propertyDomain:
      variableType: CONTINUOUS_VARIABLE_TYPE
      domainRange: [1, 10]
experiments:
  - actuatorIdentifier: custom_experiments
    experimentIdentifier: calculate_density
```

**Important:** In space YAML files, `variableType` must use full enum
constant names:

- `CONTINUOUS_VARIABLE_TYPE` (for float ranges)
- `DISCRETE_VARIABLE_TYPE` (for integer ranges with interval)
- `CATEGORICAL_VARIABLE_TYPE` (for fixed set of values)
- `BINARY_VARIABLE_TYPE` (for true/false values)

Do not use shortened forms like `continuous`, `discrete`, or `categorical` -
these will cause validation errors.

Test with dry-run:

```bash
uv run ado create space -f space.yaml --dry-run
```

Validates:

- Space configuration
- Experiment availability
- Property domain compatibility

## Validation Steps

1. **Verify experiment registration:**

   ```bash
   uv run ado get experiments
   ```

2. **Check experiment details:**

   ```bash
   uv run ado get experiments --details
   ```

3. **Test with point:**

   ```bash
   uv run run_experiment point.yaml
   ```

4. **Validate space configuration:**

   ```bash
   uv run ado create space -f space.yaml --dry-run
   ```

## Common Patterns

### Multiple Experiments in One Module

```python
@custom_experiment(output_property_identifiers=["result1"])
def experiment_one(x: float) -> dict[str, Any]:
    return {"result1": x * 2}

@custom_experiment(output_property_identifiers=["result2"])
def experiment_two(y: float) -> dict[str, Any]:
    return {"result2": y ** 2}
```

### Multiple Modules

Register each module separately:

```toml
[project.entry-points."ado.custom_experiments"]
module_one = "my_package.experiments_one"
module_two = "my_package.experiments_two"
```

### Ray Configuration

```python
@custom_experiment(
    output_property_identifiers=["loss"],
    use_ray=True,
    ray_options={
        "num_cpus": 2,
        "num_gpus": 0.5,
        "runtime_env": {"env_vars": {"OMP_NUM_THREADS": "2"}}
    }
)
def heavy_experiment(x: float) -> dict[str, Any]:
    # Implementation
    pass
```

## Troubleshooting

### Experiment Not Found

```bash
# Reinstall package
uv pip install -e . --force-reinstall

# Verify entry-point
uv run ado get experiments
```

### Domain Inference Issues

Define domains explicitly using `required_properties` parameter.

### Validation Errors

Check:

- Return dict keys match `output_property_identifiers`
- Parameter types match domain definitions
- All required parameters provided in test point

### Domain Compatibility Errors

When `ado create space --dry-run` fails with domain compatibility:

- Entity space values must be subset of experiment's required property
  domain
- For categorical properties, ensure all entity space values are in
  experiment's allowed values
- Check experiment definition for exact domain constraints

## References

- [AGENTS.md](../../../AGENTS.md) - Development guidelines
- [creating-custom-experiments.md](../../../website/docs/actuators/creating-custom-experiments.md)
  \- Full documentation
- [run_experiment.md](../../../website/docs/actuators/run_experiment.md)
  \- run_experiment details
- [density_example/](../../../examples/density_example/) - Simple example
- [optimization_test_functions/](../../../examples/optimization_test_functions/)
  \- Advanced example
- [examples/gp_tabarena_test/](../../../examples/gp_tabarena_test/) -
  GP regression with TabArena datasets
