---
name: ado-custom-experiment-definition
description: Define custom experiments for ado with proper type hints, property domains, and decorator usage. Use when creating new custom experiments or modifying existing ones.
---

# ado Custom Experiment Definition

This skill helps you define custom experiments for ado using the `@custom_experiment`
decorator with proper type hints, property domains, and validation.

## Core Concepts

### Custom Experiment Decorator

All custom experiments use the `@custom_experiment` decorator from `orchestrator.modules.actuators.custom_experiments`:

```python
from orchestrator.modules.actuators.custom_experiments import custom_experiment
from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum
from orchestrator.schema.property import ConstitutiveProperty

@custom_experiment(
    required_properties=[...],
    optional_properties=[...],
    output_property_identifiers=[...],
    metadata={...},
    use_ray=True,
    ray_options={...}
)
def my_experiment(param1: type1, param2: type2, ...) -> Dict[str, Any]:
    # Implementation
    return {"output1": value1, "output2": value2}
```

## Property Domains

### Automatic Validation

**IMPORTANT:** When you define properties with `PropertyDomain`, the
`@custom_experiment` decorator **automatically validates** all parameters.
You do NOT need to write manual validation code.

**What is validated automatically:**

- Type checking (int, float, bool, str)
- Range validation (domainRange)
- Categorical values (values list)
- Interval constraints (for discrete variables)
- Required vs optional parameters

**Example - No manual validation needed:**

```python
num_qubits_property = ConstitutiveProperty(
    identifier="num_qubits",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        domainRange=[3, 11],  # Automatically validates 3-10
        interval=1,
    ),
)

@custom_experiment(
    required_properties=[num_qubits_property],
    output_property_identifiers=["result"],
)
def my_experiment(num_qubits: int) -> Dict[str, Any]:
    # No need for: if not 3 <= num_qubits <= 10: raise ValueError(...)
    # The decorator already validated this!
    return {"result": compute(num_qubits)}
```

**When to add manual validation:**

- Complex business logic not expressible in PropertyDomain
- Cross-parameter constraints (e.g., param1 must be less than param2)
- External resource validation (e.g., file exists, API accessible)

### Defining Required Properties

Required properties must be explicitly defined using `ConstitutiveProperty`:

```python
# Discrete variable (integers)
num_items = ConstitutiveProperty(
    identifier="num_items",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        domainRange=[1, 11],  # 1-10 (exclusive upper bound)
        interval=1
    ),
    metadata={"description": "Number of items to process"}
)

# Continuous variable (floats)
temperature = ConstitutiveProperty(
    identifier="temperature",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[0.0, 100.0]
    ),
    metadata={"description": "Temperature in Celsius"}
)

# Categorical variable (strings)
algorithm = ConstitutiveProperty(
    identifier="algorithm",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
        values=["method_a", "method_b", "method_c"]
    ),
    metadata={"description": "Algorithm to use"}
)

# Binary variable (boolean)
use_cache = ConstitutiveProperty(
    identifier="use_cache",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.BINARY_VARIABLE_TYPE
    ),
    metadata={"description": "Whether to use caching"}
)

# Discrete with larger interval (steps)
num_shots = ConstitutiveProperty(
    identifier="num_shots",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        domainRange=[100, 10001],  # 100-10000 (exclusive upper)
        interval=100,  # Steps of 100: 100, 200, 300, ..., 10000
    ),
    metadata={"description": "Number of measurement shots"}
)

# Categorical with numeric values
priority = ConstitutiveProperty(
    identifier="priority",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
        values=[1, 2, 3, 5, 8]  # Fibonacci-like priorities
    ),
    metadata={"description": "Task priority level"}
)
```

### Variable Types

- `CONTINUOUS_VARIABLE_TYPE`: Float values in a range
- `DISCRETE_VARIABLE_TYPE`: Integer values with optional interval
- `CATEGORICAL_VARIABLE_TYPE`: Fixed set of string/numeric values
- `BINARY_VARIABLE_TYPE`: Boolean (True/False or 0/1)
- `OPEN_CATEGORICAL_VARIABLE_TYPE`: Categorical with unknown values
- `UNKNOWN_VARIABLE_TYPE`: Type not specified

### PropertyDomain Examples by Type

#### Discrete Numeric Variables

```python
# Simple range with interval=1
batch_size = ConstitutiveProperty(
    identifier="batch_size",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        domainRange=[16, 129],  # 16-128
        interval=1
    )
)

# Power-of-2 steps
buffer_size = ConstitutiveProperty(
    identifier="buffer_size",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        domainRange=[256, 4097],  # 256, 512, 1024, 2048, 4096
        interval=256
    )
)

# Large range with coarse granularity
population_size = ConstitutiveProperty(
    identifier="population_size",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        domainRange=[1000, 100001],
        interval=1000  # Steps of 1000
    )
)
```

#### Continuous Variables

```python
# Standard continuous range
learning_rate = ConstitutiveProperty(
    identifier="learning_rate",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[0.0001, 0.1]
    )
)

# Physical quantities
temperature_kelvin = ConstitutiveProperty(
    identifier="temperature",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[273.15, 373.15]  # 0-100°C
    )
)

# Probability/percentage
dropout_rate = ConstitutiveProperty(
    identifier="dropout_rate",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[0.0, 1.0]
    )
)
```

#### Categorical Variables

```python
# String categories
optimizer_type = ConstitutiveProperty(
    identifier="optimizer",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
        values=["adam", "sgd", "rmsprop", "adagrad"]
    )
)

# Numeric categories
num_layers = ConstitutiveProperty(
    identifier="num_layers",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
        values=[2, 4, 8, 16]  # Specific layer counts
    )
)

# Mixed type categories (converted to strings internally)
activation = ConstitutiveProperty(
    identifier="activation",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
        values=["relu", "tanh", "sigmoid", "linear"]
    )
)
```

#### Binary Variables

```python
# Boolean flag
enable_feature = ConstitutiveProperty(
    identifier="enable_feature",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.BINARY_VARIABLE_TYPE
    )
)

# On/off switch
use_gpu = ConstitutiveProperty(
    identifier="use_gpu",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.BINARY_VARIABLE_TYPE
    )
)
```

## Complete Experiment Example

```python
from typing import Dict, Any, Literal
import logging

from orchestrator.modules.actuators.custom_experiments import custom_experiment
from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum
from orchestrator.schema.property import ConstitutiveProperty

logger = logging.getLogger(__name__)

# Define properties
batch_size_property = ConstitutiveProperty(
    identifier="batch_size",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        domainRange=[1, 129],  # 1-128
        interval=1
    ),
    metadata={"description": "Batch size for processing"}
)

learning_rate_property = ConstitutiveProperty(
    identifier="learning_rate",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[0.0001, 0.1]
    ),
    metadata={"description": "Learning rate"}
)

optimizer_property = ConstitutiveProperty(
    identifier="optimizer",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
        values=["adam", "sgd", "rmsprop"]
    ),
    metadata={"description": "Optimizer type"}
)

@custom_experiment(
    required_properties=[
        batch_size_property,
        learning_rate_property,
        optimizer_property,
    ],
    output_property_identifiers=[
        "accuracy",
        "loss",
        "training_time"
    ],
    metadata={
        "description": "Train a model with specified hyperparameters",
        "version": "1.0.0",
        "author": "Your Name"
    },
    use_ray=True,
    ray_options={"num_cpus": 2, "num_gpus": 1}
)
def train_model(
    batch_size: int,
    learning_rate: float,
    optimizer: Literal["adam", "sgd", "rmsprop"]
) -> Dict[str, Any]:
    """
    Train a model with specified hyperparameters.

    Args:
        batch_size: Batch size for training
        learning_rate: Learning rate
        optimizer: Optimizer type

    Returns:
        Dictionary with accuracy, loss, and training_time
    """
    logger.info(f"Training with batch_size={batch_size}, lr={learning_rate}, optimizer={optimizer}")

    # Your training logic here
    accuracy = 0.95
    loss = 0.05
    training_time = 120.5

    return {
        "accuracy": accuracy,
        "loss": loss,
        "training_time": training_time
    }
```

## Optional Properties

Optional properties have default values in the function signature:

```python
epochs_property = ConstitutiveProperty(
    identifier="epochs",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        domainRange=[1, 101],
        interval=1
    ),
    metadata={"description": "Number of training epochs"}
)

verbose_property = ConstitutiveProperty(
    identifier="verbose",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.BINARY_VARIABLE_TYPE
    ),
    metadata={"description": "Enable verbose output"}
)

@custom_experiment(
    required_properties=[batch_size_property],
    optional_properties=[epochs_property, verbose_property],
    output_property_identifiers=["accuracy"],
)
def train_with_optional(
    batch_size: int,
    epochs: int = 10,      # Default value for optional parameter
    verbose: bool = False  # Default value for optional parameter
) -> Dict[str, Any]:
    if verbose:
        logger.info(f"Training with batch_size={batch_size}, epochs={epochs}")
    return {"accuracy": 0.95}
```

**Key points about optional properties:**

- Must have default values in function signature
- PropertyDomain still validates when value is provided
- Can be omitted from entity space in discoveryspace YAML
- If included in entity space, the space value overrides the default

## Type Inference

If you don't specify properties explicitly, ado will infer them from type hints:

```python
@custom_experiment(
    output_property_identifiers=["result"]
)
def simple_experiment(
    x: float,  # Inferred as CONTINUOUS_VARIABLE_TYPE
    y: int,    # Inferred as DISCRETE_VARIABLE_TYPE
    z: bool    # Inferred as BINARY_VARIABLE_TYPE
) -> Dict[str, Any]:
    return {"result": x + y + int(z)}
```

**Note**: Explicit property definitions are recommended for better control
over domains.

## Ray Configuration

Control parallel execution with Ray:

```python
@custom_experiment(
    # ... other parameters ...
    use_ray=True,  # Enable Ray (default)
    ray_options={
        "num_cpus": 2,           # CPUs per task
        "num_gpus": 1,           # GPUs per task
        "resources": {"custom": 1},  # Custom resources
        "runtime_env": {         # Runtime environment
            "env_vars": {"OMP_NUM_THREADS": "2"}
        }
    }
)
def my_experiment(...):
    pass
```

Set `use_ray=False` for sequential execution (only one instance at a time).

## Return Value Requirements

The return value of a custom experiment must be a dictionary that follows
these rules:

1. Must return a dictionary (not None, not a list, not a primitive type)
2. Keys must include at least one from `output_property_identifiers`
3. Values should be JSON-serializable (int, float, str, list, dict)
4. Extra keys not in `output_property_identifiers` are ignored

```python
# Good - contains required output properties
return {
    "accuracy": 0.95,
    "loss": 0.05,
    "extra_info": "ignored"  # This is fine, will be ignored
}

# Bad - missing required output properties
return {
    "wrong_key": 0.95  # Error: no valid output properties
}

# Bad - not a dictionary
return 0.95  # Error: must return a dictionary
```

### Empty Dictionary as the Return Value

**Important**: Returning an empty dictionary `{}` causes ado to treat the
experiment run as **failed**.

When an experiment fails:

1. **No data is recorded**: Failed experiments do not store measurements in the
   database
2. **Retry behavior**: Operators that support retrying (if configured) will retry
   the failed experiment
3. **No memoization**: Operators that support memoization will not cache failed
   experiment results. In future runs, these experiments will be executed again
   rather than using cached values

**When to return an empty dictionary:**

- When the experiment encounters an unrecoverable error
- When input parameters are invalid and cannot produce meaningful results
- When external dependencies fail (e.g., API unavailable, file not found)

**Example of intentional failure:**

```python
@custom_experiment(
    output_property_identifiers=["result"]
)
def my_experiment(param: float) -> Dict[str, Any]:
    if param < 0:
        logger.error(f"Invalid parameter: {param} must be non-negative")
        return {}  # Intentionally fail the experiment

    result = compute_result(param)
    return {"result": result}
```

**Best practice**: Use try-except blocks to handle exceptions and decide
whether to return an empty dictionary (failure) or partial results:

```python
@custom_experiment(
    output_property_identifiers=["primary_metric", "secondary_metric"]
)
def robust_experiment(param: float) -> Dict[str, Any]:
    try:
        primary = compute_primary(param)
        secondary = compute_secondary(param)
        return {
            "primary_metric": primary,
            "secondary_metric": secondary
        }
    except CriticalError as e:
        logger.error(f"Critical error: {e}")
        return {}  # Fail completely
    except MinorError as e:
        logger.warning(f"Minor error: {e}, returning partial results")
        return {
            "primary_metric": compute_primary(param),
            # secondary_metric omitted but primary_metric is valid
        }
```

## Package Structure and Import Conventions

### Recommended Directory Structure

```text
my_experiment_package/
├── pyproject.toml          # Package configuration
├── README.md               # Documentation
├── my_experiment/
│   ├── __init__.py        # Package exports
│   ├── experiments.py     # Decorated experiment functions
│   ├── utils.py          # Helper functions
│   ├── circuit_builder.py # Domain-specific modules
│   └── analysis.py       # Analysis functions
├── configs/               # Example configurations
│   ├── test_point.yaml
│   └── exploration.yaml
└── tests/
    └── test_experiments.py
```

### Import Path Conventions

**In `my_experiment/__init__.py`:**

```python
"""My custom experiment package for ado."""

from .experiments import experiment_one, experiment_two

__all__ = [
    "experiment_one",
    "experiment_two",
]

__version__ = "0.1.0"
```

**In `my_experiment/experiments.py`:**

```python
"""Experiment definitions with @custom_experiment decorator."""

from typing import Dict, Any
import logging

# Import from ado/orchestrator
from orchestrator.modules.actuators.custom_experiments import custom_experiment
from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum
from orchestrator.schema.property import ConstitutiveProperty

# Use relative imports for package modules
from .utils import helper_function, validate_input
from .circuit_builder import build_circuit
from .analysis import analyze_results

logger = logging.getLogger(__name__)

# Define properties
my_property = ConstitutiveProperty(...)

@custom_experiment(...)
def experiment_one(...) -> Dict[str, Any]:
    """Experiment implementation."""
    pass
```

**In `my_experiment/utils.py`:**

```python
"""Utility functions for experiments."""

import numpy as np
from typing import Any

def helper_function(param: float) -> float:
    """Helper function documentation."""
    return param * 2.0

def validate_input(value: Any) -> bool:
    """Validate input data."""
    return isinstance(value, (int, float)) and value > 0
```

**Import Best Practices:**

1. **Absolute imports for external packages:**

   ```python
   import numpy as np
   from qiskit import QuantumCircuit
   from orchestrator.modules.actuators.custom_experiments import custom_experiment
   ```

2. **Relative imports for package modules:**

   ```python
   from .utils import helper_function
   from .analysis import analyze_results
   ```

3. **Avoid circular imports:**
   - Don't import experiments.py in utils.py
   - Keep dependencies unidirectional: experiments → utils → core

4. **Type hints for clarity:**

   ```python
   from typing import Dict, Any, List, Optional, Literal
   ```

### pyproject.toml

```toml
[project]
name = "my-experiment-package"
version = "0.1.0"
description = "Custom experiments for ado"
dependencies = [
    "ado-core>=1.2.3",
    "numpy>=1.24.0",
    # ... other dependencies
]

[project.entry-points."ado.custom_experiments"]
my_experiments = "my_experiment.experiments"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

## Testing

Test your experiment before using it in ado:

```python
# Direct function call
result = my_experiment(batch_size=32, learning_rate=0.001, optimizer="adam")
assert "accuracy" in result
```

### Using run_experiment CLI

Create a test point YAML file that specifies the experiment:

```yaml
# test_point.yaml
sampleStoreIdentifier: default

entitySpace:
  - identifier: batch_size
    value: 32
  - identifier: learning_rate
    value: 0.001
  - identifier: optimizer
    value: "adam"

experiments:
  - actuatorIdentifier: custom_experiments
    experimentIdentifier: my_experiment

metadata:
  name: test-point
```

Run the experiment (no `--experiment` flag needed):

```bash
run_experiment test_point.yaml
```

The experiment is specified in the YAML file, so no command-line flag is required.

## Common Patterns

### Manual Validation (When Needed)

**Remember:** PropertyDomain handles basic validation automatically.
Only add manual validation for complex logic.

```python
def validate_cross_parameter_constraints(param1: int, param2: int):
    """Validate constraints between multiple parameters."""
    if param1 >= param2:
        raise ValueError(f"param1 ({param1}) must be less than param2 ({param2})")

def validate_external_resource(file_path: str):
    """Validate external resources."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required file not found: {file_path}")

@custom_experiment(...)
def my_experiment(param1: int, param2: int, config_file: str) -> Dict[str, Any]:
    # PropertyDomain already validated param1 and param2 ranges
    # Only validate complex constraints
    validate_cross_parameter_constraints(param1, param2)
    validate_external_resource(config_file)
    # Implementation
```

### External Library Integration

```python
import external_library

@custom_experiment(...)
def experiment_with_library(param: float) -> Dict[str, Any]:
    # Use external library
    result = external_library.process(param)

    # Convert to ado format
    return {
        "output": float(result.value),
        "metadata": str(result.info)
    }
```

## Best Practices

1. **Always provide metadata**: Include description, version, author
2. **Use explicit property definitions**: Better than type inference
3. **Add logging**: Use `logger.info()` for important steps
4. **Handle exceptions appropriately**: Use try-except blocks and decide whether
   to fail (return `{}`) or return partial results
5. **Validate inputs**: Check parameter constraints early
6. **Document functions**: Include docstrings with Args and Returns
7. **Test thoroughly**: Test with various parameter combinations including edge
   cases
8. **Keep functions focused**: One experiment per function
9. **Use type hints**: Helps with validation and IDE support
10. **Return serializable data**: Avoid complex objects
11. **Be intentional about failures**: Only return empty dictionaries when the
    experiment truly cannot produce valid results

## Next Steps

Once you're comfortable with basic custom experiment implementation, you may
want to:

- **Integrate external libraries**: See [ado Library Integration](../ado-library-integration/SKILL.md)
  for wrapping third-party libraries like Qiskit, TensorFlow, or scikit-learn
- **Formulate problems**: See [Formulate ado Problems](../formulate-ado-problems/SKILL.md)
  for creating DiscoverySpace and Operation YAML files
- **Debug experiments**: See [ado Custom Experiment Debugging](../ado-custom-experiment-debugging/SKILL.md)
  for troubleshooting common issues

## References

- [Creating Custom Experiments](../../../website/docs/actuators/creating-custom-experiments.md)
- [Quantum Circuit Example](../../../plugins/custom_experiments/quantum_circuit_exp/)
- [Plugin Development Guidelines](../../rules/plugin-development.mdc)
- Schema definitions:
  - `orchestrator/schema/domain.py`: PropertyDomain, VariableTypeEnum
  - `orchestrator/schema/property.py`: ConstitutiveProperty
  - `orchestrator/modules/actuators/custom_experiments.py`: custom_experiment decorator
