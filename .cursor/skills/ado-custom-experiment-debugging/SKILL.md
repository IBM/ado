---
name: ado-custom-experiment-debugging
description: Debug common issues with ado custom experiments including validation errors, import problems, Ray issues, and result formatting. Use when troubleshooting experiment failures or unexpected behavior.
---

# ado Custom Experiment Debugging

This skill helps you debug common issues when developing and running ado
custom experiments.

## Common Issues and Solutions

### 1. Experiment Not Found

**Symptom**: `ValueError: Requested experiment X is not in
 the CustomExperiments actuator catalog`

**Causes**:

- Package not installed
- Entry point not registered
- Module import error

**Solutions**:

```bash
# 1. Verify package is installed
pip list | grep your-package-name

# 2. Reinstall in editable mode
cd /path/to/your/package
pip install -e .

# 3. Check entry points are registered
python -c "import importlib.metadata; print(list(importlib.metadata.entry_points(group='ado.custom_experiments')))"

# 4. Verify experiments are loaded
ado get actuators --details
```

**Check pyproject.toml**:

```toml
[project.entry-points."ado.custom_experiments"]
my_experiment = "my_package.experiments"  # Must point to module with
                                          # @custom_experiment decorated functions
```

### 2. Import Errors

**Symptom**: `ImportError: cannot import name 'my_experiment'` or `ModuleNotFoundError`

**Causes**:

- Missing dependencies
- Incorrect module path
- Circular imports

**Solutions**:

```bash
# 1. Check dependencies
pip install -r requirements.txt

# 2. Test import directly
python -c "from my_package.experiments import my_experiment"

# 3. Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

**Fix circular imports**:

```python
# Bad - circular import
from .utils import helper
from .experiments import my_experiment  # experiments imports utils

# Good - import at function level
def my_experiment(...):
    from .utils import helper  # Import inside function
    result = helper(...)
```

### 3. Validation Errors

**Symptom**: `ValueError: Arguments do not match required/optional properties`

**Causes**:

- Parameter value outside domain
- Wrong parameter type
- Missing required parameter

**Debug**:

```python
# Enable verbose validation
from orchestrator.schema.point import SpacePoint

point = SpacePoint(entity={"param1": value1, "param2": value2})
entity = point.to_entity()

# This will print detailed validation errors
experiment._experiment.validate_entity(entity, verbose=True)
```

**Common fixes**:

```python
# Issue: Value outside domain
num_qubits_property = ConstitutiveProperty(
    identifier="num_qubits",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        domainRange=[2, 6],  # 2-5 (exclusive upper bound!)
        interval=1
    )
)

# Fix: Adjust domain or input value
# If you want 2-5 inclusive, use domainRange=[2, 6]
# If you pass 6, it will fail because upper bound is exclusive
```

### 4. Empty Dictionary Returns

**Symptom**: Experiment runs but no data is recorded

**Cause**: Experiment returned `{}` (empty dictionary)

**Debug**:

```python
# Add logging to see what's being returned
@custom_experiment(...)
def my_experiment(...) -> Dict[str, Any]:
    result = compute_something()
    logger.info(f"Returning: {result}")  # Check what's returned
    return result
```

**Common causes**:

```python
# Cause 1: Exception caught but empty dict returned
try:
    result = compute()
    return {"metric": result}
except Exception as e:
    logger.error(f"Error: {e}")
    return {}  # This causes experiment to fail

# Cause 2: No valid output properties
return {
    "wrong_key": 123  # Not in output_property_identifiers
}

# Cause 3: All values are None
return {
    "metric": None  # This counts as empty
}
```

**Fix**:

```python
# Return partial results when possible
try:
    primary = compute_primary()
    secondary = compute_secondary()
    return {"primary": primary, "secondary": secondary}
except PrimaryError:
    # Still return secondary if available
    return {"secondary": compute_secondary()}
except Exception as e:
    logger.error(f"Complete failure: {e}")
    return {}  # Only fail completely when necessary
```

If an experiment is failing due to an error that is deterministic consider
adopting a special property e.g. `is_valid` with boolean values indicating
whether this experiment is valid or not. This serves two purposes:

1. avoid repeating invalid experiments when re-running operations with
   an optimizer that supports memoization
2. allow future analyses to tell between invalid measurements and
   those which failed due to a transient exception

### 5. Ray Execution Issues

**Symptom**: `RayTaskError` or experiments hang

**Causes**:

- Insufficient resources
- Serialization errors
- When running on Kubernetes, Ray worker nodes are being evicted

**Debug**:

```bash
# Check Ray status
ray status

# Check Ray dashboard
# Open http://localhost:8265 in browser

# View Ray logs
tail -f /tmp/ray/session_latest/logs/*
```

**Solutions**:

```python
# Issue: Serialization error
# Ray cannot serialize certain objects (file handles, database connections, etc.)

# Bad
db_connection = create_connection()  # Created at module level

@custom_experiment(...)
def my_exp(...):
    db_connection.query(...)  # Won't work with Ray

# Good
@custom_experiment(...)
def my_exp(...):
    db_connection = create_connection()  # Create inside function
    result = db_connection.query(...)
    db_connection.close()
    return {"result": result}

# Issue: Resource constraints
@custom_experiment(
    use_ray=True,
    ray_options={"num_gpus": 2}  # Requesting 2 GPUs
)
def my_exp(...):
    pass

# Fix: Reduce resource requirements or disable Ray
@custom_experiment(
    use_ray=False  # Run sequentially
)
def my_exp(...):
    pass
```

### 6. Type Inference Failures

**Symptom**: `ValueError: Unsupported annotation` or
`Unable to generate custom experiment`

**Cause**: ado cannot infer domain from type hint

**Debug**:

```python
# These types can be inferred:
def my_exp(
    x: float,  # CONTINUOUS_VARIABLE_TYPE
    y: int,    # DISCRETE_VARIABLE_TYPE
    z: bool,   # BINARY_VARIABLE_TYPE
    w: Literal["a", "b", "c"]  # CATEGORICAL_VARIABLE_TYPE
):
    pass

# These cannot be inferred:
def my_exp(
    x: str,    # Cannot infer domain
    y: list,   # Cannot infer domain
    z: dict    # Cannot infer domain
):
    pass
```

**Fix**: Use explicit property definitions

```python
from orchestrator.schema.property import ConstitutiveProperty
from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum

string_param = ConstitutiveProperty(
    identifier="string_param",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
        values=["option1", "option2", "option3"]
    )
)

@custom_experiment(
    required_properties=[string_param],
    output_property_identifiers=["result"]
)
def my_exp(string_param: str):
    pass
```

### 7. Output Serialization Errors

**Symptom**: `TypeError: Object of type X is not JSON serializable`

**Cause**: Returning non-serializable objects

**Debug**:

```python
import json

result = my_experiment(...)
try:
    json.dumps(result)
except TypeError as e:
    print(f"Serialization error: {e}")
    print(f"Result type: {type(result)}")
    for key, value in result.items():
        print(f"{key}: {type(value)}")
```

**Common issues**:

```python
import numpy as np

# Bad - NumPy types not serializable
return {
    "value": np.float64(1.5),  # Not JSON serializable
    "array": np.array([1, 2, 3])  # Not JSON serializable
}

# Good - Convert to Python types
return {
    "value": float(np.float64(1.5)),
    "array": np.array([1, 2, 3]).tolist()
}

# Bad - Complex objects
return {
    "model": trained_model,  # Object not serializable
    "result": some_custom_class_instance
}

# Good - Extract serializable data
return {
    "model_accuracy": float(trained_model.score()),
    "result_value": float(some_custom_class_instance.value)
}
```

### 8. Discovery Space Issues

**Symptom**: `ValueError: Entity does not have values for properties required`

**Cause**: Mismatch between discovery space and experiment requirements

**Debug**:

```bash
# Check space configuration
ado show details space SPACE_ID

# Check experiment requirements
ado get actuators --details | grep -A 20 "my_experiment"
```

**Fix**:

```yaml
# Ensure entitySpace matches experiment required properties
entitySpace:
  - identifier: param1 # Must match experiment parameter name
    propertyDomain:
      domainRange: [0, 10] # Must be compatible with experiment domain
  - identifier: param2
    propertyDomain:
      values: ["a", "b", "c"]

experiments:
  - actuatorIdentifier: custom_experiments
    experimentIdentifier: my_experiment # Must match decorated function name
```

## Debugging Workflow

### 1. Test Function Directly

```python
# Test without ado
from my_package.experiments import my_experiment

result = my_experiment(param1=5.0, param2="test")
print(result)
assert "expected_output" in result
```

### 2. Test with run_experiment

```bash
# Create test point
cat > test_point.yaml << EOF
param1: 5.0
param2: "test"
EOF

# Run experiment
run_experiment test_point.yaml --experiment custom_experiments.my_experiment
```

### 3. Test in Discovery Space

```bash
# Create minimal space
cat > test_space.yaml << EOF
entitySpace:
  - identifier: param1
    propertyDomain:
      domainRange: [1, 10]
experiments:
  - actuatorIdentifier: custom_experiments
    experimentIdentifier: my_experiment
EOF

# Create and run
ado create discoveryspace test_space.yaml
ado create operation --discoveryspace <space-id> \
    --operator randomwalk --samples 5
```

### 4. Check Logs

```bash
# ado logs
tail -f ~/.ado/logs/ado.log

# Ray logs (if using Ray)
tail -f /tmp/ray/session_latest/logs/worker-*.out
tail -f /tmp/ray/session_latest/logs/worker-*.err
```

## Logging Best Practices

```python
import logging

logger = logging.getLogger(__name__)

@custom_experiment(...)
def my_experiment(param1: float, param2: str) -> Dict[str, Any]:
    """Well-instrumented experiment."""

    # Log inputs
    logger.info(f"Starting experiment with param1={param1}, param2={param2}")

    try:
        # Log major steps
        logger.debug("Computing intermediate result")
        intermediate = compute_something(param1)
        logger.debug(f"Intermediate result: {intermediate}")

        logger.debug("Computing final result")
        final = process(intermediate, param2)

        # Log outputs
        result = {"metric": final}
        logger.info(f"Experiment completed successfully: {result}")
        return result

    except ValueError as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        return {}
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {}
```

## Performance Debugging

### Slow Experiments

```python
import time
import logging

logger = logging.getLogger(__name__)

@custom_experiment(...)
def my_experiment(...) -> Dict[str, Any]:
    """Experiment with timing."""

    start_time = time.time()

    # Time each major step
    step1_start = time.time()
    result1 = step1()
    logger.info(f"Step 1 took {time.time() - step1_start:.2f}s")

    step2_start = time.time()
    result2 = step2()
    logger.info(f"Step 2 took {time.time() - step2_start:.2f}s")

    total_time = time.time() - start_time
    logger.info(f"Total time: {total_time:.2f}s")

    return {
        "result": result2,
        "execution_time": total_time
    }
```

### Memory Issues

```python
import psutil
import os

def log_memory_usage(label=""):
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"Memory usage {label}: {mem_mb:.2f} MB")

@custom_experiment(...)
def my_experiment(...) -> Dict[str, Any]:
    log_memory_usage("start")

    # Your code
    result = compute_something()

    log_memory_usage("after computation")

    return {"result": result}
```

## Common Error Messages

- Error: `No valid output properties returned`
  - Likely Cause: Return dict missing keys from `output_property_identifiers`
  - Solution: Check return dictionary keys
- Error: `Unable to generate custom experiment`
  - Likely Cause: Type inference failed
  - Solution: Use explicit property definitions
- Error: `RayTaskError`
  - Likely Cause: Serialization or resource issue
  - Solution: Check Ray logs, try without using Ray
- Error: `Experiment X is not in catalog`
  - Likely Cause: Package not installed or entry point missing
  - Solution: Reinstall package, check pyproject.toml.
    Do not mix `uv pip install` with `pip install`.
- Error: `Arguments do not match`
  - Likely Cause: Parameter validation failed
  - Solution: Check parameter domains and types
- Error: `Object not JSON serializable`
  - Likely Cause: Returning non-serializable objects
  - Solution: Convert to Python primitives

## Useful Commands

```bash
# List all custom experiments
ado get actuators --details | grep -A 50 "custom_experiments"

# Test experiment directly
python -c "from my_package.experiments import my_exp; print(my_exp(param=5.0))"

# Check Python environment
python -c "import sys; print(sys.executable); print(sys.version)"

# List installed packages
pip list | grep -E "(ado|qiskit|tensorflow)"

# Check Ray cluster
ray status

# View experiment schema
python -c "from my_package.experiments import my_exp; print(my_exp._experiment)"
```

## References

- [ado Experiment Definition](../ado-experiment-definition/SKILL.md)
- [ado Library Integration](../ado-library-integration/SKILL.md)
- [Creating Custom Experiments](../../../website/docs/actuators/creating-custom-experiments.md)
- [Ray Documentation](https://docs.ray.io/)
