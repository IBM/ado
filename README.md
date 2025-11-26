# AutoConf

This is a repository for the models used in automated configuration of resources for GenAI workloads. At the moment, there is only one model available called `min_gpu_recommender`

## min_gpu_recommender 

**min_gpu_recommender** recommends the minimum number of GPUs per worker and the number of workers required to run a tuning job without triggering a GPU Out Of Memory exception.

It combines rule-based logic with an [AutoGluon](https://auto.gluon.ai/stable/index.html) tabular model.

### Model Information

The classifier operates on the following features:

- `model_name`
- `method` (e.g., `lora`, `full`)
- `gpu_model`
- `tokens_per_sample`
- `batch_size`
- `is_valid`

and outputs 3 parameters:
- `can_recommend` with values [0,1]
- `workers` with an integer value
- `gpus` with an integer value

Please see [models README](min_gpu_recommender/AutoGluonModels/README.md) for information on model versions

### Installation

Install the package e.g. from the root directory of this git repository run `pip install ".[torch]"`. 
You may be required to install versions of ado-core that are not available on pypi or install optional dipendencies based on the model you want to run (please, refer to [the changelog](min_gpu_recommender/AutoGluonModels/changelog.md)).

### Usage

The min_gpu_recommender is exposed via an [`ado`](ibm.github.io/ado/) [custom experiment](https://ibm.github.io/ado/actuators/creating-custom-experiments/) and can be invoked in multiple ways:

#### 1. CLI
Via ado's `run_experiment` CLI command. Here's an example YAML file (which you can find under [examples/simple.yaml](examples/simple.yaml).

```yaml
entity:
  model_name: llama-7b
  method: lora
  gpu_model: NVIDIA-A100-80GB-PCIe
  tokens_per_sample: 8192
  batch_size: 16
  model_version: 1.1.0

experiments:
  - actuatorIdentifier: custom_experiments
    experimentIdentifier: min_gpu_recommender
```

To use it, from the root directory of this repository run `run_experiment examples/simple.yaml`:

After a few seconds you should see:

```bash
Point: {'model_name': 'llama-7b', 'method': 'lora', 'gpu_model': 'NVIDIA-A100-80GB-PCIe', 'tokens_per_sample': 8192, 'batch_size': 16, 'model_version': '1.1.0'}
2025-11-13 13:26:24,925 INFO worker.py:2003 -- Started a local Ray instance. View the dashboard at http://127.0.0.1:8265 
/Users/username/projects/orchestrator/autoconf/.venv/lib/python3.12/site-packages/ray/_private/worker.py:2051: FutureWarning: Tip: In future versions of Ray, Ray will no longer override accelerator visible devices env var if num_gpus=0 or num_gpus=None (default). To enable this behavior and turn off this error message, set RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
  warnings.warn(
{}
Validating entity ...
Executing: custom_experiments.min_gpu_recommender
(CustomExperiments pid=55466) Found 1 mismatches between original and current metadata:
(CustomExperiments pid=55466)   INFO: AutoGluon Python micro version mismatch (original=3.12.7, current=3.12.11)
Result:
[request_id                                                      2df09f
request_index                                                        0
entity_index                                                         0
result_index                                                         0
batch_size                                                          16
generatorid                                                        unk
gpu_model                                        NVIDIA-A100-80GB-PCIe
method                                                            lora
model_name                                                    llama-7b
model_version                                                    1.1.0
tokens_per_sample                                                 8192
identifier           model_name.llama-7b-method.lora-gpu_model.NVID...
experiment_id                   custom_experiments.min_gpu_recommender
valid                                                             True
can_recommend                                                        1
gpus                                                                 2
workers                                                              1
```

The output of the experiment are the lines:

```bash
gpus                                                                 2
workers                                                              1
can_recommend                                                        1
```

It reports that the recommender can make a suggestion (can_recommend=1). 
The suggestion comes in the form of number of workers and GPUs per worker. In the above example, you should use 1 worker with 2 GPUs.
 
#### 2. Example programmatic usage with validation
Calling decorated `min_gpu_recommender` custom experiment directly

```python
from orchestrator.schema.reference import (
    ExperimentReference,
)
from orchestrator.schema.point import SpacePoint
from orchestrator.modules.actuators.registry import ActuatorRegistry
from min_gpu_recommender.experiment_runner import (
    min_gpu_recommender,
)

configuration = {
    "model_name": "llama-7b",
    "method": "lora",
    "gpu_model": "NVIDIA-A100-80GB-PCIe",
    "tokens_per_sample": 8192,
    "batch_size": 16,
    "model_version": "1.1.0",
}

entity = SpacePoint.model_validate({"entity":configuration}).to_entity()
experiment = ActuatorRegistry().experimentForReference(
    ExperimentReference(
        actuatorIdentifier="custom_experiments",
        experimentIdentifier="min_gpu_recommender",
    )
)

assert experiment.validate_entity(entity, verbose=False) is True
measured_properties=min_gpu_recommender(entity=entity, experiment=experiment))
print(measured_properties)
```

This will print a similar text to:
```bash
2025-11-13 13:44:48,566 INFO worker.py:2003 -- Started a local Ray instance. View the dashboard at http://127.0.0.1:8265 
/Users/username/projects/orchestrator/autoconf/.venv/lib/python3.12/site-packages/ray/_private/worker.py:2051: FutureWarning: Tip: In future versions of Ray, Ray will no longer override accelerator visible devices env var if num_gpus=0 or num_gpus=None (default). To enable this behavior and turn off this error message, set RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
  warnings.warn(
(min_gpu_recommender pid=82241) Found 1 mismatches between original and current metadata:
(min_gpu_recommender pid=82241)         INFO: AutoGluon Python micro version mismatch (original=3.12.7, current=3.12.11)
[value-op-min_gpu_recommender-can_recommend:1, value-op-min_gpu_recommender-gpus:2, value-op-min_gpu_recommender-workers:1]
```

#### 3. Calling decorated `min_gpu_recommender` custom experiment fully via `ado` with validation. 
This will use ray, the `custom_experiment` actuator and return results in `ado` format (MeasurementRequest)
```python
from orchestrator.schema.reference import (
    ExperimentReference,
)
from orchestrator.schema.point import SpacePoint
from orchestrator.modules.actuators.registry import ActuatorRegistry
from orchestrator.utilities.run_experiment import local_execution_closure

configuration = {
    "model_name": "llama-7b",
    "method": "lora",
    "gpu_model": "NVIDIA-A100-80GB-PCIe",
    "tokens_per_sample": 8192,
    "batch_size": 16,
    "model_version": "1.1.0",
}

entity = SpacePoint.model_validate({"entity":configuration}).to_entity()
experiment = ActuatorRegistry().experimentForReference(
    ExperimentReference(
        actuatorIdentifier="custom_experiments",
        experimentIdentifier="min_gpu_recommender",
    )
)

assert experiment.validate_entity(entity, verbose=False) is True
request=local_execution_closure(registry=ActuatorRegistry())(reference=experiment.reference, entity=entity)
print(request.measurements[0].series_representation(output_format="target"))
```

#### 4. Example calling `min_gpu_recommender` function directly, no validation.
```python
from min_gpu_recommender.experiment_runner import (
    min_gpu_recommender,
)

configuration = {
    "model_name": "llama-7b",
    "method": "lora",
    "gpu_model": "NVIDIA-A100-80GB-PCIe",
    "tokens_per_sample": 8192,
    "batch_size": 16,
    "model_version": "1.1.0",
}

measured_properties = min_gpu_recommender._original_func(**configuration)
print(measured_properties)
```

### Example run a large sweep

You can also execute a large sweep. This example uses the space in [sweep/examples/space.yaml](sweep/examples/space.yaml) which applies the `min_gpu_recommender` experiment on 3960 configurations.

The space looks like this:

```yaml
experiments:
  - experimentIdentifier: min_gpu_recommender
    actuatorIdentifier: custom_experiments

entitySpace:
  - identifier: "model_name"
    propertyDomain:
      values:
        [
          "granite-3.1-2b",
          "granite-20b-v2",
          "granite-13b-v2",
          "granite-3-8b",
          "granite-3.1-3b-a800m-instruct",
          "granite-3.1-8b-instruct",
          "granite-34b-code-base",
          "granite-3b-code-base-128k",
          "granite-7b-base",
          "granite-8b-code-base",
          "granite-8b-japanese",
          "llama-13b",
          "llama-7b",
          "llama2-70b",
          "llama3-70b",
          "llama3-8b",
          "llama3.1-405b",
          "llama3.1-70b",
          "llama3.1-8b",
          "mistral-123b-v2",
          "mistral-7b-v0.1",
          "mixtral-8x7b-instruct-v0.1",
        ]
  - identifier: "tokens_per_sample"
    propertyDomain:
      values: [512, 1024, 2048, 4096, 8192]
  - identifier: "batch_size"
    propertyDomain:
      values: [1, 2, 4, 8, 32, 64]
  - identifier: "gpu_model"
    propertyDomain:
      values: ["NVIDIA-A100-SXM4-80GB", "NVIDIA-A100-80GB-PCIe", "L40S"]
  - identifier: method
    propertyDomain:
      values: ["full", "lora"]
  - identifier: model_version
    propertyDomain:
      values:
        - "1.1.0"
```

To execute this run:

```bash
ado create space -f examples/sweep/space.yaml
ado create operation -f examples/sweep/operation.yaml --use-latest space
: The above step will take a few minutes to sweep over the points
: This command will generate a CSV file with the results
ado show entities --use-latest space --output-format csv
open space-*.csv
```

Look for the `can_recommend`, `gpus`, and `workers` columns in the CSV file.

Learn more about exploring spaces in the ado documentation for taking a [RandomWalk on a space](https://ibm.github.io/ado/examples/random-walk/#exploring-the-discoveryspace).
