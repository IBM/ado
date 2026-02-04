---
name: ado-library-integration
description: Integrate external libraries (like Qiskit, TensorFlow, scikit-learn) with ado custom experiments. Use when wrapping third-party libraries or converting their outputs to ado format.
---

# ado Library Integration

This skill focuses on integrating third-party libraries with ado custom experiments.
It covers library-specific wrapping techniques, parameter mapping, output conversion,
and error handling patterns.

**Prerequisites:** Before using this skill, familiarize yourself with
[ado Custom Experiment Definition](../ado-custom-experiment-definition/SKILL.md)
for foundational concepts about `@custom_experiment`, PropertyDomain, and
return value requirements.

## When to Use This Skill

Use this skill when you need to:

- Wrap existing library functions as ado experiments
- Map library parameters to ado PropertyDomain definitions
- Convert library-specific outputs to ado-compatible formats
- Handle library-specific errors and exceptions
- Integrate domain-specific tools (Qiskit, TensorFlow, scikit-learn, etc.)

**Note:** For general experiment implementation without external libraries,
see [ado Custom Experiment Definition](../ado-custom-experiment-definition/SKILL.md).

## Core Integration Pattern

```python
from typing import Dict, Any
import logging
from orchestrator.modules.actuators.custom_experiments import custom_experiment
from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum
from orchestrator.schema.property import ConstitutiveProperty

# Import external library
import external_library

logger = logging.getLogger(__name__)

@custom_experiment(
    required_properties=[...],
    output_property_identifiers=[...],
)
def library_experiment(param1: type1, param2: type2) -> Dict[str, Any]:
    """Experiment using external library."""

    # 1. Map ado parameters to library parameters
    library_params = map_parameters(param1, param2)

    # 2. Call library function
    try:
        result = external_library.process(**library_params)
    except external_library.LibraryError as e:
        logger.error(f"Library error: {e}")
        return {}  # Fail the experiment

    # 3. Convert library output to ado format
    ado_output = convert_output(result)

    return ado_output
```

## Common Integration Scenarios

### 1. Qiskit (Quantum Computing)

```python
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, state_fidelity
import numpy as np

# Define properties
num_qubits_prop = ConstitutiveProperty(
    identifier="num_qubits",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        domainRange=[2, 6],
        interval=1
    )
)

gate_sequence_prop = ConstitutiveProperty(
    identifier="gate_sequence",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
        values=["H-CNOT", "X-H-CNOT", "RX-RY-CNOT"]
    )
)

@custom_experiment(
    required_properties=[num_qubits_prop, gate_sequence_prop],
    output_property_identifiers=["fidelity", "circuit_depth"],
)
def quantum_circuit_experiment(
    num_qubits: int,
    gate_sequence: str
) -> Dict[str, Any]:
    """Run quantum circuit simulation."""

    try:
        # Build circuit
        circuit = QuantumCircuit(num_qubits)

        # Apply gates based on sequence
        if gate_sequence == "H-CNOT":
            circuit.h(0)
            circuit.cx(0, 1)
        elif gate_sequence == "X-H-CNOT":
            circuit.x(0)
            circuit.h(1)
            circuit.cx(0, 1)
        # ... more patterns

        # Get statevector
        state = Statevector.from_instruction(circuit)
        target = Statevector.from_label('0' * num_qubits)

        # Calculate fidelity
        fidelity = float(state_fidelity(state, target))
        depth = circuit.depth()

        return {
            "fidelity": fidelity,
            "circuit_depth": depth
        }

    except Exception as e:
        logger.error(f"Qiskit error: {e}")
        return {}
```

### 2. TensorFlow/Keras (Machine Learning)

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

learning_rate_prop = ConstitutiveProperty(
    identifier="learning_rate",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[0.0001, 0.1]
    )
)

batch_size_prop = ConstitutiveProperty(
    identifier="batch_size",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        domainRange=[16, 129],
        interval=16
    )
)

@custom_experiment(
    required_properties=[learning_rate_prop, batch_size_prop],
    output_property_identifiers=["accuracy", "loss", "training_time"],
    use_ray=True,
    ray_options={"num_gpus": 1}
)
def train_neural_network(
    learning_rate: float,
    batch_size: int
) -> Dict[str, Any]:
    """Train a neural network."""

    import time
    start_time = time.time()

    try:
        # Load data (example)
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        # Build model
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=5,
            validation_split=0.1,
            verbose=0
        )

        # Evaluate
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

        training_time = time.time() - start_time

        return {
            "accuracy": float(test_accuracy),
            "loss": float(test_loss),
            "training_time": training_time
        }

    except Exception as e:
        logger.error(f"TensorFlow error: {e}")
        return {}
```

### 3. scikit-learn (Machine Learning)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
import numpy as np

n_estimators_prop = ConstitutiveProperty(
    identifier="n_estimators",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        domainRange=[10, 201],
        interval=10
    )
)

max_depth_prop = ConstitutiveProperty(
    identifier="max_depth",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        domainRange=[3, 21],
        interval=1
    )
)

@custom_experiment(
    required_properties=[n_estimators_prop, max_depth_prop],
    output_property_identifiers=["cv_score", "std_score"],
)
def random_forest_experiment(
    n_estimators: int,
    max_depth: int
) -> Dict[str, Any]:
    """Train and evaluate Random Forest."""

    try:
        # Generate synthetic data
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            random_state=42
        )

        # Create model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        # Cross-validation
        scores = cross_val_score(model, X, y, cv=5)

        return {
            "cv_score": float(scores.mean()),
            "std_score": float(scores.std())
        }

    except Exception as e:
        logger.error(f"scikit-learn error: {e}")
        return {}
```

### 4. Custom Simulation Library

```python
import simulation_library as simlib

temperature_prop = ConstitutiveProperty(
    identifier="temperature",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[273.15, 373.15]  # 0-100°C in Kelvin
    )
)

pressure_prop = ConstitutiveProperty(
    identifier="pressure",
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[1.0, 10.0]  # atmospheres
    )
)

@custom_experiment(
    required_properties=[temperature_prop, pressure_prop],
    output_property_identifiers=["energy", "stability", "convergence_time"],
)
def molecular_simulation(
    temperature: float,
    pressure: float
) -> Dict[str, Any]:
    """Run molecular dynamics simulation."""

    try:
        # Create simulation
        sim = simlib.Simulation()
        sim.set_temperature(temperature)
        sim.set_pressure(pressure)

        # Run simulation
        result = sim.run(steps=10000)

        # Extract metrics
        return {
            "energy": float(result.final_energy),
            "stability": float(result.stability_metric),
            "convergence_time": float(result.convergence_time)
        }

    except simlib.ConvergenceError as e:
        logger.warning(f"Simulation did not converge: {e}")
        return {}  # Fail if simulation doesn't converge
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        return {}
```

## Output Conversion Patterns

### Converting NumPy Arrays

```python
import numpy as np

def convert_numpy_output(result: np.ndarray) -> Dict[str, Any]:
    """Convert NumPy array to JSON-serializable format."""
    if result.ndim == 0:
        # Scalar
        return {"value": float(result)}
    elif result.ndim == 1:
        # Vector
        return {
            "mean": float(result.mean()),
            "std": float(result.std()),
            "min": float(result.min()),
            "max": float(result.max())
        }
    else:
        # Matrix or higher
        return {
            "shape": list(result.shape),
            "mean": float(result.mean()),
            "frobenius_norm": float(np.linalg.norm(result))
        }
```

### Converting Complex Objects

```python
def convert_model_output(model_result) -> Dict[str, Any]:
    """Convert complex model output to ado format."""
    output = {}

    # Extract numeric metrics
    if hasattr(model_result, 'accuracy'):
        output['accuracy'] = float(model_result.accuracy)

    if hasattr(model_result, 'loss'):
        output['loss'] = float(model_result.loss)

    # Convert arrays
    if hasattr(model_result, 'predictions'):
        preds = np.array(model_result.predictions)
        output['prediction_mean'] = float(preds.mean())
        output['prediction_std'] = float(preds.std())

    # Convert metadata to strings
    if hasattr(model_result, 'metadata'):
        output['metadata'] = str(model_result.metadata)

    return output
```

### Handling Time Series Data

```python
def convert_timeseries_output(timeseries: list) -> Dict[str, Any]:
    """Convert time series to summary statistics."""
    arr = np.array(timeseries)

    return {
        "final_value": float(arr[-1]),
        "mean_value": float(arr.mean()),
        "trend": float(np.polyfit(range(len(arr)), arr, 1)[0]),
        "volatility": float(arr.std())
    }
```

## Error Handling Strategies

### Graceful Degradation

```python
@custom_experiment(
    output_property_identifiers=["primary_metric", "secondary_metric"],
)
def robust_experiment(param: float) -> Dict[str, Any]:
    """Experiment with graceful degradation."""

    output = {}

    # Try primary computation
    try:
        primary = compute_primary(param)
        output["primary_metric"] = primary
    except Exception as e:
        logger.error(f"Primary computation failed: {e}")
        # Don't return yet, try secondary

    # Try secondary computation
    try:
        secondary = compute_secondary(param)
        output["secondary_metric"] = secondary
    except Exception as e:
        logger.warning(f"Secondary computation failed: {e}")

    # Return partial results if we got at least one metric
    if output:
        return output
    else:
        return {}  # Complete failure
```

## Parameter Mapping

### Simple Mapping

```python
def map_ado_to_library_params(
    ado_param1: float,
    ado_param2: str
) -> dict:
    """Map ado parameters to library parameters."""
    return {
        "library_param_a": ado_param1 * 100,  # Scale conversion
        "library_param_b": ado_param2.upper(),  # Format conversion
        "library_param_c": True  # Fixed parameter
    }
```

### Complex Mapping with Validation

```python
def map_and_validate_params(
    learning_rate: float,
    optimizer: str,
    batch_size: int
) -> dict:
    """Map and validate parameters."""

    # Map optimizer string to library object
    optimizer_map = {
        "adam": tf.keras.optimizers.Adam,
        "sgd": tf.keras.optimizers.SGD,
        "rmsprop": tf.keras.optimizers.RMSprop
    }

    if optimizer not in optimizer_map:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    # Create optimizer instance
    optimizer_instance = optimizer_map[optimizer](learning_rate=learning_rate)

    # Validate batch size
    if batch_size > 1024:
        logger.warning(f"Large batch size {batch_size}, may cause memory issues")

    return {
        "optimizer": optimizer_instance,
        "batch_size": batch_size
    }
```

## Best Practices

1. **Isolate library calls**: Wrap library-specific code in try-except blocks to
   handle library errors
2. **Convert types explicitly**: Libraries often return custom types - convert to
   Python primitives (`float()`, `int()`, `str()`)
3. **Handle library-specific errors**: Catch and handle exceptions specific to the
   library (e.g., `qiskit.QiskitError`, `tensorflow.errors.ResourceExhaustedError`)
4. **Document parameter mappings**: Clearly explain how ado parameters map to
   library parameters in docstrings
5. **Handle missing dependencies**: Provide clear error messages if required
   libraries are not installed
6. **Allocate appropriate resources**: Use Ray options to allocate GPUs/CPUs based
   on library requirements (e.g., TensorFlow needs GPUs)
7. **Cache expensive operations**: Reuse loaded models, datasets, or configurations
   when possible

## Checking Library Availability

Add version logging to help with debugging:

```python
import logging
logger = logging.getLogger(__name__)

try:
    import qiskit
    import tensorflow as tf
    logger.info(f"Qiskit version: {qiskit.__version__}")
    logger.info(f"TensorFlow version: {tf.__version__}")
except ImportError as e:
    logger.error(f"Required library not installed: {e}")
    raise
```

> **Note** that the above does not record the package versions to ado's database

## Handling File Outputs

### Important Constraints

**Key principle:** Experiments can generate files (plots, diagrams, data),
but these files are **not automatically persisted** in ado's database.

**Metric storage constraints:**

- Metrics are stored in a database
- Each metric should be at most tens of kilobytes
- Large files cannot be stored directly as metrics
- File paths are not reliably serializable across all contexts

Best Practices:

1. **Use temporary directories** for file generation
2. **Clean up files** after extracting metrics
3. **Encode small files** in metrics if needed (as base64 or text)
4. **Return computed metrics**, not file paths
5. **Document generated files** in docstrings

### Pattern 1: Temporary Directory (Recommended)

```python
import tempfile
import os
from typing import Dict, Any
import matplotlib.pyplot as plt

@custom_experiment(
    output_property_identifiers=["metric_value", "plot_summary"],
)
def experiment_with_plot(param: float) -> Dict[str, Any]:
    """
    Run experiment and generate visualization.

    Generated Files:
        Creates temporary plot file (automatically cleaned up)
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate plot
        plot_path = os.path.join(temp_dir, f"plot_{param}.png")

        plt.figure()
        plt.plot([1, 2, 3], [param, param*2, param*3])
        plt.savefig(plot_path)
        plt.close()

        # Extract metrics from plot or computation
        metric_value = compute_metric(param)

        # Optionally encode small plot as base64 (if < 50KB)
        with open(plot_path, 'rb') as f:
            plot_data = f.read()

        if len(plot_data) < 50000:  # 50KB limit
            import base64
            plot_base64 = base64.b64encode(plot_data).decode('utf-8')
        else:
            plot_base64 = None
            logger.warning(f"Plot too large ({len(plot_data)} bytes), "
                           "not including in metrics")

        return {
            "metric_value": metric_value,
            "plot_summary": f"Generated plot with {param} parameter",
            # Optionally include small encoded files
            # "plot_base64": plot_base64  # Only if needed and small
        }
    # temp_dir automatically cleaned up here
```

### Pattern 2: Persistent Output Directory

If you need to keep files for later inspection (not in database):

```python
import os
from pathlib import Path

@custom_experiment(
    output_property_identifiers=["accuracy", "file_count"],
)
def experiment_with_persistent_files(param: int) -> Dict[str, Any]:
    """
    Run experiment and save files to output directory.

    Generated Files:
        Saves results to: ./experiment_outputs/run_{param}/
        Note: File paths are stored in the ado database
              but there is no guarantee that the files will
              persist after the experiment terminates or
              that they will be accessible.
    """
    # Create output directory
    output_dir = Path("experiment_outputs") / f"run_{param}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate and save files
    result_file = output_dir / "results.txt"
    with open(result_file, 'w') as f:
        f.write(f"Results for param={param}\n")

    plot_file = output_dir / "plot.png"
    generate_plot(plot_file, param)

    # Return metrics only (not file paths)
    accuracy = compute_accuracy(param)

    return {
        "accuracy": accuracy,
        "file_count": len(list(output_dir.glob("*"))),
        "result_file": str(result_file)  # Only the path will be stored, there 
                                         # is no guarantee that you will be 
                                         # able to retrieve its contents later
    }
```

### Pattern 3: Encoding Small Files in Metrics

For small files that must be stored (< 10KB recommended):

```python
import base64
import json

@custom_experiment(
    output_property_identifiers=["metric", "config_data"],
)
def experiment_with_encoded_file(param: str) -> Dict[str, Any]:
    """
    Run experiment and encode small configuration file.

    Returns:
        Includes encoded configuration data (< 10KB)
    """
    # Generate small config file
    config = {
        "param": param,
        "settings": {"option1": True, "option2": False}
    }

    # Encode as JSON string (small)
    config_json = json.dumps(config)

    if len(config_json) < 10000:  # 10KB limit
        return {
            "metric": compute_metric(param),
            "config_data": config_json  # Store as string
        }
    else:
        logger.warning("Config too large, not including in metrics")
        return {
            "metric": compute_metric(param),
            "config_data_error": "Config too large to store"
        }
```

### Pattern 4: Qiskit Circuit Visualization

```python
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
import tempfile
import os

@custom_experiment(
    output_property_identifiers=["circuit_depth", "gate_count"],
)
def quantum_circuit_experiment(num_qubits: int) -> Dict[str, Any]:
    """
    Create quantum circuit and extract metrics.

    Generated Files:
        Creates temporary circuit diagram (automatically cleaned up)
    """
    # Build circuit
    circuit = QuantumCircuit(num_qubits)
    circuit.h(0)
    for i in range(num_qubits - 1):
        circuit.cx(i, i + 1)

    # Generate visualization in temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        diagram_path = os.path.join(temp_dir, "circuit.png")
        circuit_drawer(circuit, output='mpl', filename=diagram_path)

        # Extract metrics (not the file)
        return {
            "circuit_depth": circuit.depth(),
            "gate_count": len(circuit.data),
        }
```

### What NOT to Do

```python
# ❌ BAD: Returning file paths
@custom_experiment(...)
def bad_experiment(param: float) -> Dict[str, Any]:
    plot_path = "outputs/plot.png"
    plt.savefig(plot_path)

    return {
        "metric": 0.95,
        "plot_path": plot_path  # Don't do this!
    }

# ❌ BAD: Storing large files in metrics
@custom_experiment(...)
def bad_experiment_large_file(param: float) -> Dict[str, Any]:
    with open("large_data.bin", 'rb') as f:
        large_data = f.read()  # 10MB file

    return {
        "metric": 0.95,
        "data": large_data  # Too large for database!
    }

# ❌ BAD: Not cleaning up temporary files
@custom_experiment(...)
def bad_experiment_no_cleanup(param: float) -> Dict[str, Any]:
    temp_file = f"/tmp/temp_{param}.dat"
    with open(temp_file, 'w') as f:
        f.write("data")
    # File never cleaned up!

    return {"metric": 0.95}
```

### Summary: File Output Guidelines

- Temporary files: Use `with tempfile.TemporaryDirectory():`
- Small config/data (order of 10KB): Encode as strings in metrics
- Large files: Don't store in metrics; save externally if needed and
               persist the paths in metrics
- File paths: Only for files stored in persistent storage
- Cleanup: Always clean up temporary files

## References

- [ado Custom Experiment Definition](../ado-custom-experiment-definition/SKILL.md)
- [Quantum Circuit Example](../../../plugins/custom_experiments/quantum_circuit_exp/)
- [Creating Custom Experiments](../../../website/docs/actuators/creating-custom-experiments.md)
