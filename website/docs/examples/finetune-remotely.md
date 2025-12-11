# Measure throughput of finetuning on a Remote RayCluster

> [!NOTE]
>
> This example illustrates:
>
> 1. Setting up a remote RayCluster environment for running finetuning performance
>    benchmarks with SFTTrainer
>
> 2. Benchmarking a set of finetuning configurations using GPUs on a remote
>    RayCluster

## The scenario

When you run a finetuning workload, you can choose values for parameters like
the model name, batch size, and number of GPUs. To understand how these choices
affect performance, a common strategy is to measure changes in system behavior
by **exploring the workload parameter space**.

This approach applies to many machine learning workloads where performance
depends on configuration.

**In this example, `ado` is used to explore LLM fine-tuning throughput across a
fine-tuning workload parameter space on a remote RayCluster.**

To explore this space, you will:

- define the parameters to test - such as the batch size and the model max
  length
- define what to test them with - In this case we will use SFTTrainer's
  `finetune_full_benchmark-v1.0.0` experiment
- explore the parameter space - the sampling method

> [!NOTE]
>
> This example assumes you have already followed the
> [Measure throughput of finetuning locally](./finetune-locally.md) example.

## Prerequisites

1. A remote shared context is available (see
   [shared contexts](../../resources/metastore/) for more information). Here we
   call it `finetuning` but it can have any name.

2. A [remote RayCluster](../../actuators/sft-trainer/#configure-your-raycluster)
   with a GPU worker with at least one `NVIDIA-A100-SXM4-80GB` GPU. The `RayCluster`
   should also include the NVIDIA development and runtime packages. We recommend
   deploying the RayCluster following our
   [documentation](../../actuators/sft-trainer/#configure-your-raycluster).
   Ensure that the base virtual environment on your Ray GPU workers meets the
   requirements of `fms-hf-tuning==3.0.0`: a) Python 3.11 and b) `torch==2.6.0`
   pre-installed.
   If you are using a RayCluster on Kubernetes, we recommend using the image:
   `quay.io/ado/ado:c6ba952ad79a2d86d1174fd9aaebddd8953c78cf-py311-cu121-ofed2410v1140`.

3. If you host your RayCluster on Kubernetes or OpenShift, make sure you're
   logged in to the Kubernetes or Openshift cluster.

4. Activate the `finetuning` shared context for the example.

    <!-- markdownlint-disable-next-line code-block-style -->
    ```commandline
    ado context finetuning
    ```

## Install and Configure the SFTTrainer actuator

> [!WARNING]
>
> The SFTTrainer actuator currently **supports only Python 3.10, 3.11, 3.12**.

### Install the SFTTrainer actuator

<!-- markdownlint-disable code-block-style -->
=== "Install the SFTTrainer Actuator plugin from PyPi"

    ```commandline
    pip install ado-sfttrainer
    ```

=== "Install SFTTrainer from the `ado` sources"

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
 <!-- markdownlint-enable code-block-style -->

### Configure the SFTTrainer Actuator

SFTTrainer includes parameters that control its behavior. For example, it pushes
any training metrics it collects, like system profiling metadata, to an
[AIM](https://github.com/aimhubio/aim) server by default. It also features
parameters that define important paths, such as the location of the Hugging Face
cache and the directory where the actuator expects to find files like the test
Dataset.

In this section you will configure the actuator for experiments on your remote
RayCluster.

<!-- markdownlint-disable code-block-style -->
=== "If you do not have an AIM server"

     Create the file `actuator_configuration.yaml` with the following contents:

     ```yaml
     actuatorIdentifier: SFTTrainer
     parameters:
       hf_home: /hf-models-pvc/huggingface_home
       data_directory: /data/fms-hf-tuning/artificial-dataset/
     ```

=== "If you have an AIM server"

    Create the file `actuator_configuration.yaml` with the following contents:

    ```yaml
    actuatorIdentifier: SFTTrainer
    parameters:
      aim_db: aim://$the-aim-server-domain-or-ip:port
      aim_dashboard_url: https://$the-aim-dashboard-domain-or-ip:port
      hf_home: /hf-models-pvc/huggingface_home
      data_directory: /data/fms-hf-tuning/artificial-dataset/
    ```
<!-- markdownlint-enable code-block-style -->

>[!IMPORTANT]
>
> If you have deployed a custom RayCluster then make sure that the `hf_home` and
> `data_directory` parameters point to paths that can be created by your remote
> RayCluster workers. We recommend deploying a remote RayCluster following our
> [instructions](../../actuators/sft-trainer/#configure-your-raycluster).

Next, create the `actuatorconfiguration` resource:

```commandline
ado create actuatorconfiguration -f actuator_configuration.yaml
```

The command will print the ID of the resource. Make a note of it, you will need
it in a later step.

See the full list of the actuator parameters you can set in the
[SFTTrainer reference docs](../../actuators/sft-trainer#actuator-parameters).

## Prepare the remote RayCluster

>[!NOTE]
>
> This section assumes you have
> [configured your RayCluster for use with SFTTrainer](../../actuators/sft-trainer/#configure-your-raycluster)
> and that you have configured your SFTTrainer actuator with the values we
> provided above for the `hf_home` and `data_directory` parameters.

### For RayClusters on Kubernetes/OpenShift - create a port-forward

>[!TIP]
>
> If your remote RayCluster is not hosted on Kubernetes or OpenShift, you can
> skip this step.

In a terminal, start a `kubectl port-forward` process to the service that
connects to the head of your RayCluster. Keep this process running until your
experiments finish.

For example, if the name of your RayCluster is `ray-disorch`, run:

<!-- markdownlint-disable-next-line code-block-style -->
```commandline
kubectl port-forward svc/ray-disorch-head-svc 8265
```

Verify that the port forward is active by visiting <http://localhost:8265> you
should see the landing page of the Ray web dashboard.

### Prepare files for the Ray jobs you will run later

Create a directory named `my-remote-measurements` and `cd` into it. You will
keep all the files for this example in there.

Similar to how you installed `ado` and `SFTTrainer` on your laptop, it's
important to ensure these Python packages are also available on your remote
RayCluster.

You have two options for installing the required packages:

1. **Pre-install the packages** in the virtual environment of your RayCluster
   before deployment.
2. Use a
   [Ray Runtime environment YAML](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#api-reference),
   which instructs Ray to dynamically install Python packages during runtime.

In this section, we’ll focus on the second approach.

<!-- markdownlint-disable code-block-style -->
=== "Use the SFTTrainer plugin wheel from PyPi"

    Create the `ray_runtime_env.yaml` file under the directory
    `my-remote-measurements` with the following contents:

    <!-- markdownlint-disable line-length -->
    ```yaml
    pip:
       - ado-sfttrainer
    env_vars:
      env_vars:
        AIM_UI_TELEMETRY_ENABLED: "0"
        # We set HOME to /tmp because "import aim.utils.tracking" tries to write under $HOME/.aim_profile.
        # However, the process lacks permissions to do so and that leads to an ImportError exception.
        HOME: "/tmp/"
        OMP_NUM_THREADS: "1"
        OPENBLAS_NUM_THREADS: "1"
        RAY_AIR_NEW_PERSISTENCE_MODE: "0"
        PYTHONUNBUFFERED: "x"
    ```
    <!-- markdownlint-enable line-length -->

    If your RayCluster doesn't already have `ado` installed in its virtual
    environment then include the `ado-core` package too.

=== "Build the python wheel yourself"

    Build the python wheel for the Actuator plugin `SFTTrainer`.

    Briefly, if you are in the top level of the `ado` repository execute:

    ```bash
    python -m build -w plugins/actuators/sfttrainer -o plugins/actuators/sfttrainer/dist
    mv plugins/actuators/sfttrainer/dist/*.whl ${path to my-remote-measurements}
    ```

    Then create a `ray_runtime_env.yaml` file under `my-remote-measurements` with
    the following contents (update the wheel name accordingly):

    <!-- markdownlint-disable line-length -->
    ```yaml
    pip:
       - ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/ado_sfttrainer-1.1.0.dev152+g23c7ba34e-py3-none-any.whl
    env_vars:
       env_vars:
       AIM_UI_TELEMETRY_ENABLED: "0"
       # We set HOME to /tmp because "import aim.utils.tracking" tries to write under $HOME/.aim_profile.
       # However, the process lacks permissions to do so and that leads to an ImportError exception.
       HOME: "/tmp/"
       OMP_NUM_THREADS: "1"
       OPENBLAS_NUM_THREADS: "1"
       RAY_AIR_NEW_PERSISTENCE_MODE: "0"
       PYTHONUNBUFFERED: "x"
    ```
    <!-- markdownlint-enable line-length -->

    If your RayCluster doesn't already have `ado` installed in its virtual
    environment then build the wheel for `ado-core` by repeating the above in the
    root directory of `ado`. Then add an entry under `pip` pointing to the the
    resulting `ado-core` wheel file.

    !!! info end

        Your wheel filenames may vary.

    For convenience, you can run the script below from inside the
    `my-remote-measurements` directory. It will build the wheels of both `ado` and
    `sfttrainer` and then automatically generate the `ray_runtime_env.yaml` file
    under your working directory. The script builds the wheel for `ado` too.

    ```bash
    $path_to_ado_root/plugins/actuators/sfttrainer/examples/build_wheels.sh
    ```

    [Reference docs on using ado with remote RayClusters](../../getting-started/remote_run/#getting-ready).
<!-- markdownlint-enable code-block-style -->

You will use the files you created during this step in later steps when
submitting jobs to your remote RayCluster.

### Create the test Dataset on the remote RayCluster

Use the `.whl` and `ray_runtime_env.yaml` files with `ray job submit` to launch
a job on your remote RayCluster. This job will create the synthetic dataset and
place it in the correct location under the directory specified by the
`data_directory` parameter of the SFTTrainer actuator.

>[!INFO]
>
> You can find instructions for generating the `.whl` and `ray_runtime_env.yaml`
> files in the
> [Prepare files for the Ray jobs you will run later](#prepare-files-for-the-ray-jobs-you-will-run-later)
> section.

To submit the job to your remote RayCluster run the command:

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line code-block-style -->
```commandline
ray job submit --address http://localhost:8265 --runtime-env ray_runtime_env.yaml \
--working-dir $PWD -v -- sfttrainer_generate_dataset_text \
-o /data/fms-hf-tuning/artificial-dataset/news-tokens-16384plus-entries-4096.jsonl
```
<!-- markdownlint-enable line-length -->

[Reference docs on creating the datasets](../../actuators/sft-trainer/#creating-the-datasets)

### Download model weights on the remote RayCluster

Next, submit a ray job that downloads the model weights for
[`granite-3.1-2b`](https://huggingface.co/ibm-granite/granite-3.1-2b-base) in
the appropriate path under the directory specified by the `hf_home` parameter of
the SFTTrainer actuator.

First, save the following YAML to a file `models.yaml` inside your working
directory (`my-remote-measurements`):

<!-- markdownlint-disable-next-line code-block-style -->
```yaml
granite-3.1-2b:
  Vanilla: ibm-granite/granite-3.1-2b-base
```

To submit the Ray job run:

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line code-block-style -->
```commandline
ray job submit --address http://localhost:8265 --runtime-env ray_runtime_env.yaml \
--working-dir $PWD -v -- \
sfttrainer_download_hf_weights -i models.yaml -o /hf-models-pvc/huggingface_home
```
<!-- markdownlint-enable line-length -->

[Reference docs on pre-fetching weights](../../actuators/sft-trainer/#model-weights)

## Run the example

### Define the finetuning workload configurations to test and how to test them

In this example, we create a `discoveryspace` that runs the
[finetune_full_benchmark-v1.0.0](../../actuators/sft-trainer/#finetune_full_benchmark-v100)
experiment to finetune the
[`granite-3.1-2b`](https://huggingface.co/ibm-granite/granite-3.1-2b-base) using
1 GPU.

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

1. Create the file `space.yaml` with the contents:

    <!-- markdownlint-disable-next-line code-block-style -->
    ```yaml
    experiments:
      - experimentIdentifier: finetune_full_benchmark-v1.0.0
        actuatorIdentifier: SFTTrainer
        parameterization:
          - property:
              identifier: fms_hf_tuning_version
            value: "3.0.0"
          - property:
              identifier: stop_after_seconds
            # Set training duration to at least 30 seconds.
            # For meaningful system metrics, we recommend a minimum of 300 seconds.
            value: 30

    entitySpace:
      - identifier: "model_name"
        propertyDomain:
          values: ["granite-3.1-2b"]
      - identifier: "number_gpus"
        propertyDomain:
          values: [1]
      - identifier: "gpu_model"
        propertyDomain:
          values: ["NVIDIA-A100-SXM4-80GB"]
      - identifier: "model_max_length"
        propertyDomain:
          values: [512, 1024]
      - identifier: "batch_size"
        propertyDomain:
          values: [1, 2]
    ```

2. Create the space:

    <!-- markdownlint-disable-next-line code-block-style -->
    ```commandline
    ado create space -f space.yaml
    ```

   The space will use the `default` sample store.

### Create a random walk `operation` to explore the space

1. Create the file `operation.yaml` with the following contents:

    <!-- markdownlint-disable-next-line code-block-style -->
    ```yaml
    spaces:
      - The identifier of the DiscoverySpace resource
    actuatorConfigurationIdentifiers:
      - The identifier of the Actuator Configuration resource

    operation:
      module:
        operatorName: "random_walk"
        operationType: "search"
      parameters:
        numberEntities: all
        singleMeasurement: True
        batchSize: 1 # you may increase this number if you have more than 1 GPU
        samplerConfig:
          mode: sequential
          samplerType: generator
    ```

2. Replace the placeholders with your `discoveryspace` ID and
    `actuatorconfiguration` ID and save it in a file with the name
    `operation.yaml`.

3. Export the `finetuning` context so you can supply it to the remote
    operation.

    <!-- markdownlint-disable-next-line code-block-style -->
    ```commandline
    ado get context --output yaml finetuning > context.yaml
    ```

4. For the next step your `my-remote-measurements` directory needs the
    following files, although the wheels may have different ids.

    <!-- markdownlint-disable-next-line code-block-style -->
    ```terminaloutput
    my-remote-measurements
    ├── ado_core-1.1.0.dev133+f4b639c1.dirty-py3-none-any.whl
    ├── context.yaml
    ├── operation.yaml
    ├── ray_runtime_env.yaml
    └── ado_sfttrainer-1.1.0.dev133+gf4b639c10.d20250812-py3-none-any.whl
    ```

5. Create the operation on the remote RayCluster

Use the `.whl` and `ray_runtime_env.yaml` files to submit a job to your
remote RayCluster which creates the `operation` that runs your finetuning
measurements.

>[!NOTE]
>
> You can find instructions for generating the `.whl` and `ray_runtime_env.yaml`
> files in the
> [Prepare files for the Ray jobs you will run later](#prepare-files-for-the-ray-jobs-you-will-run-later)
> section.

Run the command:

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line code-block-style -->
```commandline
ray job submit --no-wait --address http://localhost:8265  --working-dir . \
--runtime-env ray_runtime_env.yaml -v -- \
ado -c context.yaml create operation -f operation.yaml
```
<!-- markdownlint-enable line-length -->

The operation will execute the measurements (i.e. apply the experiment
**finetune_full_benchmark-v1.0.0** on the 4 entities) as defined in your
`discoveryspace`.

> [!NOTE]
>
> Each measurement finetunes the
> [`granite-3.1-2b`](https://huggingface.co/ibm-granite/granite-3.1-2b-base)
> model and takes about two minutes to complete.
> There is a total of four measurements.
> It will also take a couple of minutes for Ray to create the ray environment
> on participating GPU worker nodes, so expect the `operation`
> to take around 10 minutes to complete.

[Reference docs for submitting ado operations to remote RayClusters](../../getting-started/remote_run).

### Examine the results of the exploration

After the operation completes, you can retrieve the results of your
measurements:

<!-- markdownlint-disable-next-line code-block-style -->
```commandline
ado show entities --output-format csv --property-format=target space --use-latest
```

> [!NOTE]
>
> Notice that because the context we are using refers to a remote project we can
> access the data created by the operation on the remote ray cluster. Anyone that
> has access to the `finetuning` context can also retrieve the results of your
> measurements!

The command will generate a CSV file. Open it to explore the data that your
operation has collected!

It should look similar to this:

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line code-block-style -->
```csv
,identifier,generatorid,experiment_id,model_name,number_gpus,gpu_model,model_max_length,batch_size,gpu_compute_utilization_min,gpu_compute_utilization_avg,gpu_compute_utilization_max,gpu_memory_utilization_min,gpu_memory_utilization_avg,gpu_memory_utilization_max,gpu_memory_utilization_peak,gpu_power_watts_min,gpu_power_watts_avg,gpu_power_watts_max,gpu_power_percent_min,gpu_power_percent_avg,gpu_power_percent_max,cpu_compute_utilization,cpu_memory_utilization,train_runtime,train_samples_per_second,train_steps_per_second,dataset_tokens_per_second,dataset_tokens_per_second_per_gpu,is_valid
0,model_name.granite-3.1-2b-number_gpus.1-gpu_model.NVIDIA-A100-SXM4-80GB-model_max_length.512-batch_size.2,explicit_grid_sample_generator,SFTTrainer.finetune_full_benchmark-v1.0.0-fms_hf_tuning_version.3.0.0-stop_after_seconds.30,granite-3.1-2b,1,NVIDIA-A100-SXM4-80GB,512,2,40.0,40.0,40.0,25.49774175,25.49774175,25.49774175,31.498108,169.3855,169.3855,169.3855,42.346375,42.346375,42.346375,74.075,2.6414139999999997,31.3457,130.672,16.334,2744.108442306281,2744.108442306281,1
1,model_name.granite-3.1-2b-number_gpus.1-gpu_model.NVIDIA-A100-SXM4-80GB-model_max_length.512-batch_size.1,explicit_grid_sample_generator,SFTTrainer.finetune_full_benchmark-v1.0.0-fms_hf_tuning_version.3.0.0-stop_after_seconds.30,granite-3.1-2b,1,NVIDIA-A100-SXM4-80GB,512,1,27.25,27.25,27.25,25.71380625,25.71380625,25.71380625,31.786194,141.78924999999998,141.78924999999998,141.78924999999998,35.447312499999995,35.447312499999995,35.447312499999995,74.325,2.6420105,30.5903,133.899,33.475,1405.9358685596414,1405.9358685596414,1
2,model_name.granite-3.1-2b-number_gpus.1-gpu_model.NVIDIA-A100-SXM4-80GB-model_max_length.1024-batch_size.1,explicit_grid_sample_generator,SFTTrainer.finetune_full_benchmark-v1.0.0-fms_hf_tuning_version.3.0.0-stop_after_seconds.30,granite-3.1-2b,1,NVIDIA-A100-SXM4-80GB,1024,1,43.25,43.25,43.25,25.43853775,25.43853775,25.43853775,31.498108,181.79625,181.79625,181.79625,45.4490625,45.4490625,45.4490625,74.32499999999999,2.64201475,30.6802,133.506,33.377,2670.126009608803,2670.126009608803,1
3,model_name.granite-3.1-2b-number_gpus.1-gpu_model.NVIDIA-A100-SXM4-80GB-model_max_length.1024-batch_size.2,explicit_grid_sample_generator,SFTTrainer.finetune_full_benchmark-v1.0.0-fms_hf_tuning_version.3.0.0-stop_after_seconds.30,granite-3.1-2b,1,NVIDIA-A100-SXM4-80GB,1024,2,63.75,63.75,63.75,25.67718525,25.67718525,25.67718525,31.737366,238.53825,238.53825,238.53825,59.6345625,59.6345625,59.6345625,74.1,2.6399939999999997,30.1566,135.824,16.978,5161.324552502603,5161.324552502603,1
```
<!-- markdownlint-enable line-length -->

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

</div>
<!-- markdownlint-enable line-length -->