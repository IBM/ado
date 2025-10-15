# Exploring vLLM deployment configurations

> [!NOTE]
>
> This example illustrates using the vllm-performance actuator to discover
> how best to deploy vLLM for a given use-case
<!-- markdownlint-disable-next-line MD028 -->

> [!IMPORTANT]
>
> **Prerequisites**
>
> - Access to a k8s namespace where you can deploy vLLM

## The scenario

When deploying vLLM, you need to choose values for parameters like GPU type,
batch size, and memory limits.
These choices directly affect performance, cost, and scalability.
To find the best configuration for your workload, whether you're optimizing for
latency, throughput, or cost—you need to explore the deployment parameter space.

In this example,

- we will define a space of vLLM deployment configurations to test with
the `vllm_performance` actuator's `performance_testing_full` experiment
  - This experiment can create and characterize a vLLM deployment on Kubernetes
- use the `random_walk` operator to explore the space

## Install the actuator

[//]: # (If you haven't already:)

[//]: # ()
[//]: # (```commandline)

[//]: # (pip install ado-vllm-performance)

[//]: # (```)

[//]: # ()
[//]: # (If you have cloned the `ado` source repository you can also do:)

[//]: # ()
[//]: # (```commandline)

[//]: # (# From the root of this repository )

[//]: # (pip install -e plugins/actuators/vllm_performance)

[//]: # (```)

Execute:

```commandline
pip install -e plugins/actuators/vllm_performance
```

in the root of the `ado` source repository.
You can clone the repository with

```commandline
git clone https://github.com/IBM/ado.git
```

Verify the installation with:

```commandline
ado get actuators --details 
```

The actuator `vllm_performance` will appear in the list of available actuators.

## Create an actuator configuration

The vllm-performance actuator needs some information the target cluster to
deploy on. This is provided via an `actuatorconfiguration`.

First execute,

```commandline
# Generate the template file
ado template actuatorconfiguration --actuator-identifier vllm_performance
```

This will create a file called:

Edit the file and set correct values for the following fields:

<!-- markdownlint-disable line-length -->
```yaml
hf_token: <your HuggingFace access token>
namespace: vllm-testing # OpenShift namespace you have write access to
node_selector: '{"kubernetes.io/hostname":"<host-with-gpu>"}' # JSON string selecting a node that owns GPU
```
<!-- markdownlint-enable line-length -->

Then save this configuration as an `actuatorconfiguration` resource:

```bash
ado create actuatorconfiguration -f $CONFIG_FILE
```

Record the identifier of the created `actuatorconfiguration` as it
will be used later.

> [!TIP]
>
> You can create multiple actuator configurations corresponding
> to different clusters/target environments.
> You choose the one to use when you launch an operation requiring the actuator

## Define the configurations to test

When exploring vLLM deployments there are two sets of
parameters that can be varied:

- the deployment creation parameters (number GPUs, memory allocated etc)
- the benchmark test parameters (request per second to send, tokens per request etc.)

In this case we define a space where we look at the impact of a few vLLM
deployment parameters, including `max_num_seq` and `max_batch_tokens`, for a
scenario where requests arrive between 1 and 10 per second with sizes
around 2000 tokens .

```yaml
sampleStoreIdentifier: 2963a5
entitySpace:
  - identifier: model
    propertyDomain:
      values:
        - ibm-granite/granite-3.3-8b-instruct
  - identifier: image
    propertyDomain:
      values:
        - quay.io/dataprep1/data-prep-kit/vllm_image:0.1
  - identifier: "number_input_tokens"
    propertyDomain:
      values: [1024, 2048, 4096]
  - identifier: "request_rate"
    propertyDomain:
      domainRange: [1,10]
      interval: 1
  - identifier: n_cpus
    propertyDomain:
      domainRange: [2,16]
      interval: 2
  - identifier: memory
    propertyDomain:
      values: ["128Gi", "256Gi"]
  - identifier: "max_batch_tokens"
    propertyDomain:
      values: [1024, 2048, 4096, 8192, 16384, 32768]
  - identifier: "max_num_seq"
    propertyDomain:
      values: [16,32,64]
  - identifier: "n_gpus"
    propertyDomain:
      values: [1]
  - identifier: "gpu_type"
    propertyDomain:
      values: ["NVIDIA-A100-80GB-PCIe"]
experiments:
  - actuatorIdentifier: vllm_performance
    experimentIdentifier: performance-testing-full
metadata:
  description: A space of vllm deployment configurations
  name: vllm_deployments
```

Save the above as `vllm_discoveryspace.yaml`.
Then, if you have an existing `samplestore`, run

```bash
ado create space -f vllm_discoveryspace.yaml --set sampleStoreIdentifier=$SAMPLE_STORE_ID
```

otherwise create a new one:

```bash
ado create space -f vllm_discoveryspace.yaml --new-sample-store
```

Record the identifier of the created `discoveryspace` as it
will be used in next section.

## Explore the space with random_walk

Next we'll scan this space sequentially using a `grouped` sampler
to increase efficiency.
The `grouped` sampler ensures we explore all the different
benchmark configurations for a given vLLM deployment before
creating a new deployment - minimising the need the number
of deployment creations.

```yaml
metadata:
  name: randomwalk-grouped-vllm-performance-full
spaces:
  - space-230d24-03b22d
actuatorConfigurationIdentifiers:
  - actuatorconfiguration-vllm_performance-09fcdf30
operation:
  module:
    moduleClass: RandomWalk
  parameters:
    numberEntities: all
    batchSize: 1
    samplerConfig:
      mode: 'sequentialgrouped'
      samplerType: 'generator'
      grouping: #A unique combination of these properties is a new vLLM deployment
        - model
        - image
        - memory
        - max_batch_tokens
        - max_num_seq
        - n_gpus
        - gpu_type
        - n_cpus
```

Save the above as `random_walk.yaml`. Then execute the operation:

<!-- markdownlint-disable line-length -->
```commandline
ado create operation -f random_walk.yaml --set "spaces[0]=$DISCOVERY_SPACE_ID" --set actuatorConfigurationIdentifier[0]=$ACTUATOR_CONFIGURATION_IDENTIFIER
```
<!-- markdownlint-enable line-length -->

where `$DISCOVERY_SPACE_ID` is the identifier of the `discoveryspace`
you created in the previous step, and `ACTUATOR_CONFIGURATION_IDENTIFIER` is
the identifier of the `actuatorconfiguration` created earlier.

### Monitor the optimization

While the operation is running you can watch the deployment:

```bash
# In a separate terminal
oc get deployments --watch -n vllm-testing
```

You can see the measurement requests as the operation runs
by executing (in another terminal):

```commandline
ado show requests operation $OPERATION_ID
```

and the results (this outputs the entities in sampled order):

```commandline
ado show entities operation $OPERATION_ID
```

If the `operation` is running the $OPERATION_ID will have been output
just before the sampling started.
Assuming no other operation was started it will also be
the last id output by

```commandline
ado get operations
```

### Check final results

When the output indicates that the experiment has finished, you
can inspect the results of all operations run so far on the space with:

```commandline
ado show entities space $DISCOVERY_SPACE_ID --output-format csv
```

> [!NOTE]
> At any time after an operation, $OPERATION_ID, is finished you can run
> `ado show entities operation $OPERATION_ID`
> to see the sampling time-series of that operation.

## Next steps

- Try varying **`max_batch_tokens`** or **`gpu_memory_utilization`** to
explore the impact on throughput.
- Try creating a difference `actuatorconfiguration` with more
`max_environments` and running the random walk with a non-grouped sampler
- Replace the model with a different HF checkpoint to compare performance.
- Use **RayTune**
(see the [vLLM endpoint performance](vllm-performance-endpoint.md) example)
to optimise the hyper‑parameters of the benchmark.
