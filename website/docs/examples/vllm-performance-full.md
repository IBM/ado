# Testing the throughput of an inference endpoint

> [!NOTE]
>
> This example illustrates:
>
> 1. Using the vllm-performance actuator to discover how best to deploy vllm
> for a given use-case

## The scenario

TBA

> [!IMPORTANT]
>
> **Prerequisites**
>
> - An endpoint serving a LLM via the OpenAI API

## 1. Install the actuator

```bash
# From the root of this repository
pip install -e plugins/actuators/vllm_performance
# Verify installation
ado get actuators --details
```

The actuator will appear in the list of available actuators.

## 2. Create an actuator configuration

The vllm-performance actuator needs some information, for example about the target
cluster to deploy on.

```bash
# Generate the template file
ado template actuatorconfiguration --actuator-identifier vllm_performance
```

This will create a file called

Edit the file (open in your editor) and set correct values for the following fields:

<!-- markdownlint-disable line-length -->
```yaml
hf_token: <your HuggingFace access token>
namespace: vllm-testing # OpenShift namespace you have write access to
node_selector: '{"kubernetes.io/hostname":"<host-with-gpu>"}' # JSON string selecting a node that owns GPU
```
<!-- markdownlint-enable line-length -->

Then create the actuator configuration resource

```bash
ado create actuatorconfiguration -f $CONFIG_FILE
```

> [!TIP]
>
> You can create multiple actuator configurations corresponding
> to different clusters/target environments.
> You choose the one to use when you launch an operation requiring the actuator

## 3. Prepare a discovery space (the configuration space to explore)

The discovery space is defined in a YAML file. An example is:

```yaml
# Example discovery space for vLLM performance
sampleStoreIdentifier: <sample_store_id>
entitySpace:
  - property: model
    type: string
    values:
      - ibm-granite/granite-3.3-8b-instruct
  - property: gpu_type
    type: string
    values:
      - NVIDIA-A100-80GB-PCIe
  - property: n_gpus
    type: integer
    min: 1
    max: 1
  - property: n_cpus
    type: integer
    min: 2
    max: 8
  - property: memory
    type: string
    values:
      - 128Gi
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

## 4. Create an operation to run the experiment

```yaml
operation:
  module:
    operatorName: "random_walk"
    operationType: "search"
  parameters:
    numberEntities: all
    singleMeasurement: true
    mode: sequential
    samplerType: generator
    spaces:
      - <spaceid>
    actuatorConfigurationIdentifiers:
      - <actuatorconfiguration-identifier-from-step-2>
```

Save the above as `random_walk.yaml`. Then execute the operation:
<!-- markdownlint-disable line-length -->
```commandline
ado create operation -f random_walk.yaml --set "spaces[0]=$DISCOVERY_SPACE_ID" --set "actuatorConfigurationIdentifiers[0]=$ACTUATOR_CONF_ID"
```
<!-- markdownlint-enable line-length -->

> [!TIP]
>
> If you prefer you can also edit the file and change the fields

## 5. Monitor the run

While the operation is running you can watch the deployment:

```bash
# In a separate terminal
oc get deployments --watch -n vllm-testing
```

you can also see the measurement requests as the operation runs

```commandline
ado show requests operation $OPERATION_ID
```

and the results

```commandline
ado show entities operation $OPERATION_ID
```

When the output indicates that the experiment has finished,
you can inspect the results of all operation  run so far on the space with

```bash
ado show entities space $afo --output-format csv
```

## 6. Clean up

Delete the Kubernetes resources if you no longer need them:

```bash
# From the actuator directory (if you used the default templates)
ad
auto-delete -f deployment.yaml
```

And optionally delete the ADO space and store:

```bash
ado delete space $afo
ado delete samplestore <your_sample_store_id>
```

## Next steps

- Try varying **`max_batch_tokens`** or **`gpu_memory_utilization`** to
explore the impact on throughput.
- Replace the model with a different HF checkpoint to compare performance.
- Use **RayTune** (see `best-configuration-search.md`) to optimise the
hyper‑parameters of the benchmark.

---

This example demonstrates the full workflow from installation to result retrieval,
mirroring the style of the other examples.
