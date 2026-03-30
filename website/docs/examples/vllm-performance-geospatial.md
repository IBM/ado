# Benchmarking Geospatial Models with vLLM

<!-- markdownlint-disable no-blanks-blockquote -->

> [!NOTE] The scenario
>
> **In this example, the
> [_vllm_performance_ actuator](../actuators/vllm_performance.md) is used to
> benchmark geospatial models (IBM-NASA Prithvi) for Earth observation tasks.**
>
> Geospatial models process satellite imagery for tasks like flood detection,
> land use classification, and environmental monitoring. Unlike text-based LLMs,
> these models:
>
> - Accept base64-encoded satellite images as input
> - Output classification results rather than text tokens
> - Have different performance characteristics and optimization requirements
>
> In this example:
>
> - We will define a space of geospatial model deployment configurations to test
> - Use the `test-geospatial-deployment-v1` experiment to create and benchmark
>   vLLM deployments serving Prithvi models
> - Explore how deployment parameters affect inference latency for flood
>   detection tasks

> [!IMPORTANT] Prerequisites
>
> - Be logged-in to your Kubernetes/OpenShift cluster
> - Have access to a namespace where you can create vLLM deployments
> - Install the following Python packages locally:
>
> ```bash
> pip install ado-vllm-performance
> ```

> [!TIP] TL;DR
>
> Create the following files and execute:
>
> ```bash
> # Create resources and run operation
> ado create op -f geospatial_operation.yaml \
>    --with space=geospatial_space.yaml --with ac=vllm_actuator_configuration.yaml
> ```
>
> See
> [configuring the `vllm_performance` actuator](../actuators/vllm_performance.md#configuring-the-vllm_performance-actuator)
> for configuration options.

<!-- markdownlint-enable no-blanks-blockquote -->

## Verify the installation

Verify the installation with:

```commandline
ado get actuators --details
```

The actuator `vllm_performance` should appear in the list. To see the geospatial
experiments:

```commandline
ado get experiments --details
```

You should see experiments including `test-geospatial-deployment-v1` and
`test-geospatial-endpoint-v1`.

## Create an actuator configuration

The vllm-performance actuator needs information about the target cluster. This
is provided via an `actuatorconfiguration`.

First execute:

```commandline
ado template actuatorconfiguration --actuator-identifier vllm_performance -o vllm_actuator_configuration.yaml
```

Edit the file and set correct values for at least the `namespace` field:

```yaml
# you MUST set this to a namespace where you can create vLLM deployments
namespace: vllm-testing
# Required to access Prithvi models
hf_token: <your HuggingFace access token>
```

Then save this configuration:

```bash
ado create actuatorconfiguration -f vllm_actuator_configuration.yaml
```

## Define the geospatial configurations to test

For geospatial models, we focus on deployment parameters that affect inference
latency since these models output classification results rather than generating
tokens. Key parameters include:

- **GPU configuration**: Type and number of GPUs
- **Memory allocation**: CPU and GPU memory
- **Batch processing**: `max_num_seq` for concurrent requests
- **Workload pattern**: Request rate and concurrency

Save the following as `geospatial_space.yaml`:

```yaml
metadata:
  name: geospatial-flood-detection-space
  description:
    "Explore Prithvi geospatial model deployment configurations for flood
    detection"
entitySpace:
  - identifier: model
    propertyDomain:
      variableType: CATEGORICAL_VARIABLE_TYPE
      values:
        - "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"
  - identifier: n_gpus
    propertyDomain:
      variableType: DISCRETE_VARIABLE_TYPE
      values: [1]
  - identifier: gpu_type
    propertyDomain:
      variableType: CATEGORICAL_VARIABLE_TYPE
      values:
        - "NVIDIA-A100-80GB-PCIe"
  - identifier: memory
    propertyDomain:
      variableType: CATEGORICAL_VARIABLE_TYPE
      values:
        - "64Gi"
        - "128Gi"
  - identifier: max_num_seq
    propertyDomain:
      variableType: DISCRETE_VARIABLE_TYPE
      values: [32, 64, 128]
  - identifier: request_rate
    propertyDomain:
      variableType: DISCRETE_VARIABLE_TYPE
      values: [10, 50, 100]
  - identifier: dataset
    propertyDomain:
      variableType: CATEGORICAL_VARIABLE_TYPE
      values:
        - "india_url_in_b64_out"
        - "valencia_url_in_b64_out"
measurementSpace:
  - actuatorIdentifier: vllm_performance
    experimentIdentifier: test-geospatial-deployment-v1
```

Then run:

```bash
ado create space -f geospatial_space.yaml
```

This space explores:

- Two pre-packaged flood detection datasets (India and Valencia regions)
- Different memory allocations (64Gi vs 128Gi)
- Various batch sizes (32, 64, 128 concurrent requests)
- Multiple request rates (10, 50, 100 requests/second)

## Explore the space with random_walk

We'll use the `random_walk` operator with grouped sampling to efficiently
explore the space. Grouped sampling ensures we test all workload patterns for a
given deployment before creating a new one.

Save the following as `geospatial_operation.yaml`:

```yaml
metadata:
  name: geospatial-exploration
  description: "Explore geospatial model deployment configurations"
operatorIdentifier: random_walk
operatorConfiguration:
  sampler:
    samplerIdentifier: grouped_explicit_grid_sampler
    samplerConfiguration:
      grouping:
        - model
        - n_gpus
        - gpu_type
        - memory
        - max_num_seq
      numberOfSamples: all
```

Then, start the operation with:

```commandline
ado create operation -f geospatial_operation.yaml \
           --use-latest space --use-latest actuatorconfiguration
```

As it runs, a table of results is updated live in the terminal.

## Understanding the Results

Geospatial experiments measure end-to-end latency metrics:

- **duration**: Total benchmark duration
- **completed**: Number of successful requests
- **request_throughput**: Requests processed per second
- **mean_e2el_ms**: Mean end-to-end latency in milliseconds
- **p50_e2el_ms, p99_e2el_ms**: Latency percentiles

Unlike text LLMs, geospatial models don't generate tokens, so metrics like TTFT
(Time To First Token) and TPOT (Time Per Output Token) are not applicable.

### Monitor the deployment

While the operation is running you can monitor the deployment:

```bash
# In a separate terminal
oc get deployments --watch -n vllm-testing
```

You can also get the results table by executing (in another terminal):

```commandline
ado show entities operation --use-latest
```

### Check final results

When the experiment finishes, inspect all results with:

```commandline
ado show entities space --output-format csv --use-latest
```

## Pre-packaged Datasets

The actuator includes two pre-packaged datasets for flood detection:

- **india_url_in_b64_out**: Satellite imagery from India region with flood
  detection labels
- **valencia_url_in_b64_out**: Satellite imagery from Valencia region with flood
  detection labels

These datasets contain base64-encoded satellite images suitable for the
Prithvi-EO-2.0 flood detection models.

## Using Custom Datasets

To use your own geospatial datasets, use the
`test-geospatial-deployment-custom-dataset-v1` experiment. Your dataset should
be a JSONL (JSON Lines) file where each line is a JSON object with this
structure:

```jsonl
{"prompt": {"data": {"data": "https://example.com/path/to/image.tif",
"data_format": "url", "out_data_format": "b64_json",
"indices": [1, 2, 3, 8, 11, 12]}}}
{"prompt": {"data": {"data": "https://example.com/path/to/image2.tif",
"data_format": "url", "out_data_format": "b64_json",
"indices": [1, 2, 3, 8, 11, 12]}}}
```

> [!IMPORTANT] Model-Specific Payload Format
>
> The payload structure shown above is specific to the **IBM-NASA Prithvi
> geospatial models** (Prithvi-EO-2.0-300M and 600M). If you are using a
> different geospatial model, you must adapt the payload format to match your
> model's expected input structure. Consult your model's documentation for the
> correct payload format, including:
>
> - Required fields and their structure
> - Supported data formats (URL, base64, etc.)
> - Expected spectral band indices
> - Any model-specific parameters

Each line contains a `prompt` object with a `data` object containing:

- **data**: URL or base64-encoded string of the satellite image
- **data_format**: Format of the input data (`"url"` or `"b64"`)
- **out_data_format**: Format for output data (`"b64_json"`)
- **indices**: List of spectral band indices to use (e.g.,
  `[1, 2, 3, 8, 11, 12]` for Sentinel-2 bands used by Prithvi models)

Update your space definition to use the custom dataset experiment:

```yaml
measurementSpace:
  - actuatorIdentifier: vllm_performance
    experimentIdentifier: test-geospatial-deployment-custom-dataset-v1
```

And add the dataset path to your entity space:

```yaml
entitySpace:
  - identifier: dataset
    propertyDomain:
      variableType: CATEGORICAL_VARIABLE_TYPE
      values:
        - "/path/to/your/dataset.jsonl"
```

## Next steps

<!-- markdownlint-disable MD028 -->

- Try the **600M parameter Prithvi model** by changing the model identifier to
  `ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL-Sen1Floods11`
- Explore different **GPU types** if your cluster has multiple options
- Test **endpoint benchmarking** with `test-geospatial-endpoint-v1` if you have
  an existing deployment
- Use the [**RayTune** operator](../operators/optimisation-with-ray-tune.md) to
  find optimal configurations for your latency requirements
- Run
  [the exploration on the OpenShift/Kubernetes cluster](../actuators/vllm_performance.md#the-in_cluster-configuration-option)
  to avoid keeping your laptop open
- Check the
  [`vllm_performance` actuator documentation](../actuators/vllm_performance.md)
  for more configuration options

<!-- markdownlint-enable MD028 -->
