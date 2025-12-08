# Measure throughput of finetuning locally

> [!NOTE]
>
> This example illustrates:
>
> 1. Setting up a local environment for running finetuning performance benchmarks
>    using SFTTrainer
>
> 2. Benchmarking a set of finetuning configurations for a small model using a
>    local context and only the CPU

## The scenario

When you run a finetuning workload, you can choose values for parameters like
the model name, batch size, and number of GPUs. To understand how these choices
affect performance, a common strategy is to measure changes in system behavior
by **exploring the workload parameter space**.

This approach applies to many machine learning workloads where performance
depends on configuration.

**In this example, `ado` is used to explore the parameter space for finetuning a
small language model on your laptop without using GPUs.**

To explore this space, you will:

- define the parameters to test - such as the batch size and the model max
  length
- define what to test them with - In this case we will use SFTTrainer's
  `finetune_full_benchmark-v1.0.0` experiment
- explore the parameter space - the sampling method

Here, you'll use the `finetune_full_benchmark-v1.0.0` experiment that the
SFTTrainer actuator provides to run four measurements on your laptop without
using GPUs. Each measurement records metrics like `dataset tokens per second`
and stores the results in `ado`'s database.

## Pre-requisites

### Set Active Context

You should use the `local` context for the example.

```commandline
ado context local
```

### Install the SFTTrainer actuator

> [!WARNING]
>
> The SFTTrainer actuator currently **supports only Python 3.10, 3.11, 3.12**.

=== "Install the SFTTrainer Actuator plugin from PyPi"

    <!-- markdownlint-disable code-block-style -->
    ```commandline
    pip install ado-sfttrainer
    ```
    <!-- markdownlint-enable code-block-style -->

=== "Install SFTTrainer from the `ado` sources"

    <!-- markdownlint-disable code-block-style -->
    !!! info

        This step assumes you are in the root directory of the ado source repository.
    <!-- markdownlint-enable code-block-style -->

    If you haven't already installed the `SFTTrainer` actuator, run
    (assumes you are in the root directory of ado):

    ```commandline
    pip install plugins/actuators/sfttrainer
    ```

     then executing

     ```commandline
     ado get actuators
     ```

     should show an entry for `SFTTrainer` like below

     ```
                ACTUATOR ID
     0   custom_experiments
     1                 mock
     2               replay
     3           SFTTrainer
     ```

### Configure the parameters of the SFTTrainer actuator

SFTTrainer includes parameters that control its behavior. For example, it pushes
any training metrics it collects, like system profiling metadata, to an
[AIM](https://github.com/aimhubio/aim) server by default. It also features
parameters that specify important paths, such as the location of the Hugging Face
cache and the directory where the actuator expects to find files like the test
dataset.

In this section you will configure the actuator for running experiments locally
and storing data under the path `/tmp/ado-sft-trainer-hello-world/`.

Create a file called `actuator_configuration.yaml` with the following contents:

```yaml
actuatorIdentifier: SFTTrainer
parameters:
  match_exact_dependencies: False
  data_directory: /tmp/ado-sft-trainer-hello-world/data-files
  cache: /tmp/ado-sft-trainer-hello-world/cache
  hf_home: ~/.cache/huggingface
```

To create the `actuatorconfiguration` resource run:

```commandline
ado create actuatorconfiguration -f actuator_configuration.yaml
```

See the full list of parameters you can set in an `actuatorconfiguration`
resource for the SFTTrainer actuator in its
[reference docs](../../actuators/sft-trainer#actuator-parameters).

## Environment setup

### Create the Dataset

The finetuning measurements require a synthetic dataset which is a file named
`news-tokens-16384plus-entries-4096.jsonl` in the directory
`/tmp/ado-sft-trainer-hello-world/data-files` which is under the path specified
by the `data_directory` actuator parameter.

You can reuse this Dataset for any future measurements you run on this device.

To generate the dataset run the following command:

```commandline
sfttrainer_generate_dataset_text \
  -o /tmp/ado-sft-trainer-hello-world/data-files/news-tokens-16384plus-entries-4096.jsonl
```

### Download model weights

Next download the weights of the model we use in this example
([`smollm2-135m`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)) in the
appropriate path under the directory specified by the `hf_home` parameter of the
SFTTrainer actuator.

First, store the below YAML to the file `models.yaml` inside your working
directory:

```yaml
smollm2-135m:
  Vanilla: HuggingFaceTB/SmolLM2-135M
```

Then, run the command:

```commandline
sfttrainer_download_hf_weights -i models.yaml -o ~/.cache/huggingface
```

## Run the example

This section explains the process of using `ado` to define and launch a set of
finetuning measurements which store their results in the `local` context.

### Define the finetuning workload configurations to test and how to test them

A `discoveryspace` defines what you want to measure (Entity Space) and how you
want to measure it (Measurement Space). It also links to the `samplestore` which
is where Entities and their measured properties are stored in.

In this example, we create a `discoveryspace` that runs the
[finetune_full_benchmark-v1.0.0](../../actuators/sft-trainer/#finetune_full_benchmark-v100)
experiment to finetune the
[`smollm2-135m`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) model
without using any GPUs.

The `entitySpace` defined below includes four dimensions:

- `model_name` and `number_gpus` each contain a single value.
- `model_max_length` and `batch_size` each contain two values.

The total number of entities in the `entitySpace` is the number of unique
combinations of values across all dimensions. In this case, the configuration
contains 4 entities.

You can find the complete list of the entity space properties in the
documentation of the
[finetune_full_benchmark-v1.0.0](../../actuators/sft-trainer/#finetune_full_benchmark-v100)
experiment.

To create the Discovery Space:

1. Create the file `space.yaml` with the following content

    <!-- markdownlint-disable line-length -->
    ```yaml
    experiments:
      - experimentIdentifier: finetune_full_benchmark-v1.0.0
        actuatorIdentifier: SFTTrainer
        parameterization:
          - property:
              identifier: fms_hf_tuning_version
            value: "2.8.2"
          - property:
              identifier: stop_after_seconds
            value: 30
          - property:
              identifier: flash_attn
            value: False

    entitySpace:
      - identifier: "model_name"
        propertyDomain:
          values: ["smollm2-135m"]
      - identifier: "number_gpus"
        propertyDomain:
          values: [0]
      - identifier: "model_max_length"
        propertyDomain:
          values: [512, 1024]
      - identifier: "batch_size"
        propertyDomain:
          values: [1, 2]
    ```
    <!-- markdownlint-enable line-length -->

2. Create the space:

    ```commandline
    ado create space -f space.yaml
    ```

   The space will use the `default` sample store.

### Create a random walk `operation` to explore the space

1. Create the file `operation.yaml` with the following content:

    ```yaml
    spaces:
      - <will be set by ado>
    actuatorConfigurationIdentifiers:
      - <will be set by ado>

    operation:
      module:
        operatorName: "random_walk"
        operationType: "search"
      parameters:
        numberEntities: all
        singleMeasurement: True
        samplerConfig:
          mode: sequential
          samplerType: generator
    ```

2. Create the operation

    ```commandline
    ado create operation -f operation.yaml \
                --use-latest space --use-latest actuatorconfiguration
    ```

The operation will execute the measurements (i.e. apply the experiment
**finetune_full_benchmark-v1.0.0** on the 4 entities) based on the definition of
your `discoveryspace`. The remaining three measurements will reuse both the
cached model weights and the cached data, making them faster to complete.

!!! info end
    <!-- markdownlint-disable-next-line code-block-style -->
    Each measurement takes about two minutes to complete, with a total of four
    measurements. Ray may take a few minutes to build the Ray runtime
    environment on participating ray workers, so expect the operation to take around
    10 minutes to complete.

### Examine the results of the exploration

After the operation completes, you can retrieve the results of your
measurements:

<!-- markdownlint-disable line-length -->
```commandline
ado show entities space --output-format csv --use-latest
```
<!-- markdownlint-enable line-length -->

The command will generate a CSV file. Open it to explore the data that your
operation has collected!

It should look similar to this:

```csv
,identifier,generatorid,experiment_id,model_name,number_gpus,model_max_length,batch_size,gpu_compute_utilization_min,gpu_compute_utilization_avg,gpu_compute_utilization_max,gpu_memory_utilization_min,gpu_memory_utilization_avg,gpu_memory_utilization_max,gpu_power_watts_min,gpu_power_watts_avg,gpu_power_watts_max,gpu_power_percent_min,gpu_power_percent_avg,gpu_power_percent_max,cpu_compute_utilization,cpu_memory_utilization,train_runtime,train_samples_per_second,train_steps_per_second,dataset_tokens_per_second,dataset_tokens_per_second_per_gpu,is_valid
0,model_name.smollm2-135m-number_gpus.0-model_max_length.512-batch_size.1,explicit_grid_sample_generator,SFTTrainer.finetune_full_benchmark-v1.0.0-fms_hf_tuning_version.2.8.2-stop_after_seconds.30-flash_attn.0,smollm2-135m,0,512,1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,79.55,49.11366699999999,30.4385,134.566,33.642,2624.0452059069926,2624.0452059069926,1
1,model_name.smollm2-135m-number_gpus.0-model_max_length.512-batch_size.2,explicit_grid_sample_generator,SFTTrainer.finetune_full_benchmark-v1.0.0-fms_hf_tuning_version.2.8.2-stop_after_seconds.30-flash_attn.0,smollm2-135m,0,512,2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,76.3,49.163925750000004,30.095,136.103,17.013,4355.274962618375,4355.274962618375,1
2,model_name.smollm2-135m-number_gpus.0-model_max_length.1024-batch_size.1,explicit_grid_sample_generator,SFTTrainer.finetune_full_benchmark-v1.0.0-fms_hf_tuning_version.2.8.2-stop_after_seconds.30-flash_attn.0,smollm2-135m,0,1024,1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,68.775,49.0008355,30.3635,134.899,33.725,3912.0654733479346,3912.0654733479346,1
3,model_name.smollm2-135m-number_gpus.0-model_max_length.1024-batch_size.2,explicit_grid_sample_generator,SFTTrainer.finetune_full_benchmark-v1.0.0-fms_hf_tuning_version.2.8.2-stop_after_seconds.30-flash_attn.0,smollm2-135m,0,1024,2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,46.85,49.00481125,30.1101,136.034,17.004,4353.090823344991,4353.090823344991,1
```

In the above CSV file you will find 1 column per:

- entity space property (input to the experiment) such as `batch_size` and
  `model_max_length`
- measured property (output to the experiment) such as
  `dataset_tokens_per_second_per_gpu` and `gpu_memory_utilization_peak`

For a complete list of the entity space properties check out the documentation
for the
[finetune_full_benchmark-v1.0.0](../../actuators/sft-trainer/#finetune_full_benchmark-v100)
experiment in the SFTTrainer docs. The complete list of measured properties is
[available there too](../../actuators/sft-trainer/#measured-properties).

## Next steps

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line no-inline-html -->
<div class="grid cards" markdown>

- 🔬️ **Find out more about the SFTTrainer actuator**

    ---

    The actuator supports several experiments, each with a set of configurable parameters.

    [Reference docs for the SFTTrainer actuator](../../actuators/sft-trainer/)

- ⚙️ **Configure your RayCluster for SFTTrainer measurements**

    ---

    Learn how to configure your RayCluster for SFTTrainer measurements.

    [Configure the RayCluster for SFTTrainer](../actuators/sft-trainer.md#configure-your-raycluster)

- :octicons-rocket-24:{ .lg .middle } **Scale it up!**

    ---

    Take it to the next level by running an experiment on your remote RayCluster.

    [run Operations remotely](./finetune-remotely.md)

</div>
<!-- markdownlint-enable line-length -->