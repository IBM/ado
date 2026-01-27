# The `vllm_performance` actuator

> [!TIP] Overview
>
> The `vllm_performance` actuator **can automatically create and
> benchmark [vLLM](https://github.com/vllm-project/vllm) inference
> deployments on Kubernetes and OpenShift clusters**.
>
> It is designed for robust, repeatable, and configurable experiment execution.
> It is suitable for both simple one-off benchmarks and large parameter sweeps.
> This actuator supports benchmarking vLLM deployments via
> [vLLM's built-in benchmarking tool](https://docs.vllm.ai/en/stable/cli/bench/serve/)
> and [GuideLLM](https://github.com/vllm-project/guidellm).

<!-- markdownlint-disable-next-line MD028 -->

> [!NOTE] Installing
>
> Run:
>
> ```commandline
> pip install ado-vllm-performance
> ```
>
> This will automatically install both vLLM and GuideLLM benchmarking
> tools, enabling all experiments:
>
> - `test-deployment-v1` and `test-endpoint-v1` (vLLM benchmarks)
> - `test-deployment-guidellm-v1` and `test-endpoint-guidellm-v1`
>   (GuideLLM benchmarks)

## Key Capabilities

- **Automated LLM benchmarking:** Deploys vLLM serving endpoints
  on NVIDIA GPU-enabled OpenShift/Kubernetes clusters and runs
  standardized serving benchmarks.
- **Cluster integration:** Handles deployments and clean-up of vLLM inference
  pods on OpenShift/Kubernetes, with configurable resource selection via namespace,
  node selector, and PVC/service templates.
- **Scenario configurability:** Supports customizing models, NVIDIA GPU types,
  node selection, retry behavior, concurrent deployments, and more
- **Efficient sampling:** Supports grouped sampling which maximises reuse
  of vLLM deployments, hence minimising time spent creating such deployments
- **Endpoint benchmarking:** Can also be used to benchmark existing OpenAI
  compatible endpoints

### Available experiments

The `vllm_performance` actuator implements four experiments:

- `test-deployment-v1`: This experiment can test the full vLLM workload
  configuration, including resource requests and server deployment
  configuration. It deploys servers with given configuration on kubernetes and
  runs vLLM's built-in benchmarking tool
  (`vllm bench serve`) on them with the given parameters.
- `test-endpoint-v1`: This experiment is equivalent to running
  `vllm bench serve` against an endpoint.
- `test-deployment-guidellm-v1`: Similar to `test-deployment-v1`, but uses
  GuideLLM (`guidellm benchmark run`)
  for benchmarking instead of vLLM's built-in benchmarking tool.
- `test-endpoint-guidellm-v1`: Similar to `test-endpoint-v1`, but uses
  GuideLLM (`guidellm benchmark run`) for benchmarking instead of vLLM's
  built-in benchmarking tool.

---

## Running single experiments: Quick endpoint and deployment tests

For rapid testing and debugging, you can use the [`run_experiment`](run_experiment.md)
tool to execute individual experiments on a single point (entity).
This is ideal when you want to:

- Quickly check if your actuator installation and configuration works
- Debug a deployment scenario or endpoint using the vllm_performance actuator

### Running an endpoint test

To test the throughput or limits of an existing vLLM-compatible endpoint, create
a `point.yaml`file like this:

```yaml
entity:
  model: openai/gpt-oss-20b
  endpoint: http://localhost:8000
  request_rate: 50
experiments:
  - actuatorIdentifier: vllm_performance
    experimentIdentifier: test-endpoint-v1
```

Then run:

```shell
run_experiment point.yaml
```

This will assess how many requests per second the endpoint can handle for the given
model and configuration.

> [!TIP] Inference endpoint testing example
>
> See [the detailed endpoint scenario](../examples/vllm-performance-endpoint.md)
> for a production-style workflow exploring inference endpoint throughput.

### Running a deployment test

To launch and benchmark a temporary vLLM deployment
(including provisioning on Kubernetes/OpenShift), you must provide both:

<!-- markdownlint-disable MD007 -->

- An entity definition (as before)
- The identifier of a valid `actuatorconfiguration` resource -
This contains information necessary for accessing and creating
deployments on the Kubernetes/OpenShift cluster -
See [configuring the vllm_performance actuator](#configuring-the-vllm_performance-actuator)
for details.
<!-- markdownlint-enable MD007 -->

Example `point.yaml`:

```yaml
entity:
  model: ibm-granite/granite-3.3-2b-instruct
  n_cpus: 8
  memory: 128Gi
  gpu_type: NVIDIA-A100-80GB-PCIe
  max_batch_tokens: 8192
  max_num_seq: 32
  n_gpus: 1
  request_rate: 10
experiments:
  - actuatorIdentifier: vllm_performance
    experimentIdentifier: test-deployment-v1
```

Then run:

```shell
run_experiment point.yaml  --actuator-config-id my-vllm-performance-config
```

Here `my-vllm-performance-config` is the ID of an `actuatorconfiguration`
resource containing the details for accessing and running on your target
cluster. See
[configuring the vllm_performance actuator](#configuring-the-vllm_performance-actuator)
for more.

This command will provision the deployment for the specified entity, using your
indicated actuator configuration, run the benchmark, and print results.

> [!TIP] vLLM deployment example
>
> See [the vLLM deployment exploration example](../examples/vllm-performance-full.md)
> for details on how to explore many deployment configurations.

---

## Configuring the vllm_performance actuator

You can configure how the `vllm_performance` actuator creates, manage, and
monitor vLLM deployments on a Kubernetes/OpenShift cluster. This configuration
covers several needs:

- **Cluster targeting and permissions**: Specify the OpenShift/Kubernetes
  namespace and optionally node selectors, secrets, and templates to match your
  cluster resources.
- **Secure access**: Pass required HuggingFace tokens, set up image pull
  secrets, control in-cluster or remote execution, and toggle SSL verification.
- **Experiment protocol and retries**: Choose how benchmarks are run, including
  interpreter, retry logic, and YAML templates for deployments/services used.
- **Deployment resource management**: Limit the number of concurrent deployments
  and control automated clean-up.

You supply this configuration information as an `ado`
[`actuatorconfiguration` resource](../resources/actuatorconfig.md),
which is a YAML file with the configuration options.
An example is:

<!-- markdownlint-disable line-length -->

```yaml
actuatorIdentifier: vllm_performance #The actuator the configuration is for
metadata:
  description: "Actuator config for vLLM LLM benchmarking"
  name: demo-vllm-perf
parameters:
  benchmark_retries: 3                  # Number of benchmark attempts (see Failure Handling)
  hf_token: "<YOUR_HUGGINGFACE_TOKEN>"  # Required for pulling some models
  image_secret: ""                      # Optional image pull secret
  in_cluster: false                     # Set to true if running from within the cluster
  interpreter: python3                  # Language for test drivers/benchmarks
  max_environments: 1                   # Max concurrent vLLM deployments
  namespace: "mynamespace"              # OpenShift/K8s namespace to deploy into
  node_selector:                        # A dictionary of Kubernetes node_selector key:value pairs
    "kubernetes.io/hostname":"gpunode01"
  pvc_name: null                        # Name of existing PVC to use. If null/omitted a temporary PVC is created
  retries_timeout: 5                    # Seconds between retries (exponential backoff)
  verify_ssl: false                     # Whether to verify HTTPS endpoints
```

<!-- markdownlint-enable line-length -->

If the above YAML was saved to a file called `vllm_config.yaml` you would create
the configuration using

```commandline
ado create actuatorconfiguration -f vllm_config.yaml
```

> [!WARNING] namespace
>
> The critical parameter you must set in the configuration is `namespace`

<!-- markdownlint-disable-next-line MD028 -->

> [!WARNING] GPU type
>
> The GPU type to use in an experiment is set via the experiment itself (test-deployment-v1).
> **Do not** set this via the `node_selector` parameter of the configuration.

<!-- markdownlint-disable-next-line MD028 -->

> [!TIP] Further details
>
> For further details on specific options and advanced behavior see:
>
> - [Maximum number of deployments](#maximum-number-of-deployments)
>   (details on `max_environments`)
> - [Handling benchmark failures](#handling-benchmark-failures) and
>   [Deployment Clean-Up](#deployment-clean-up)
> - [Grouped sampling for efficient deployment usage](#grouped-sampling-for-efficient-deployment-usage)

### Multiple configurations

You can create multiple `actuatorconfiguration`s for the `vllm_performance`
actuator. Each configuration captures the cluster-specific, security-sensitive,
and experiment-relevant settings necessary for the actuator to operate in a
given environment. Each configuration will have a different id and you can
choose the one to use when submitting an operation or single experiment that
uses the `vllm_performance` actuator.

> [!TIP] Getting a default configuration
>
> You can generate a default configuration via the ado CLI:
>
> ```shell
> ado template actuatorconfiguration --actuator-identifier vllm_performance -o actuatorconfiguration.yaml
> ```

---

## vLLM deployment management

### The `in_cluster` configuration option

The `in_cluster` option in your `actuatorconfiguration` tells the
`vllm_performance` actuator how to communicate with the target Kubernetes or
OpenShift cluster when running `test-deployment-v1`.

If running `ado` from outside the Kubernetes/OpenShift cluster where the
deployments will be created, leave `in_cluster: false` (the default).

Set `in_cluster: true` if your `ado` operation will be run on a **remote Ray
cluster that is in the same Kubernetes/OpenShift cluster** as your vLLM
deployments. This configuration maximizes efficiency for large-scale,
distributed benchmarking. For a detailed guide on running `ado` remotely on a
Ray cluster, including environment and package setup, see
[Running ado remotely](../getting-started/remote_run.md).

> [!IMPORTANT] RayCluster permissions
>
> If running with `in_cluster=True`, your RayCluster **must** be configured so that
> jobs launched by `ado` have permissions to create and manage Kubernetes deployments,
> pods, and services.
> For configuring the necessary ServiceAccount, roles, and permissions,
> see our [documentation on deploying RayClusters for `ado`](../getting-started/installing-backend-services.md).

<!-- markdownlint-disable-next-line MD028 -->

> [!TIP] Installing the `vllm_performance` actuator on a remote RayCluster
>
> If the `ado-vllm-performance` actuator is not installed in the
> image used by the RayCluster you can have [ray install it following
> this guide](../getting-started/remote_run.md).
>
> In particular, if a compatible version of vLLM and GuideLLM is not installed
> in the image this step will require installing vLLM
> (so `vllm bench serve` is available) and GuideLLM on each
> RayCluster node.
> This can take some time so you may see the `ado` `operation` output "hang"
> while this is happening.

### Maximum number of deployments

The actuator configuration parameter `max_environments` controls how many
concurrent vLLM deployments will be created. The default is 1.

When experiments are requested, if an existing deployment cannot be used a new
environment is created as long as `max_environments` has not been reached. If it
has been reached, then the actuator waits for an existing environment to become
idle, at which point it is deleted and the new environment is created.

Some notes:

<!-- markdownlint-disable MD007 -->

- `max_environments` deployments are always created before any are deleted
  - This means idle environments will remain until
    there is a need to delete them
  - This is to increases chances they can be
    reused/minimise cost of redeploying
- Environment creation is serialized - If `max_environments`
is reached and all are active, the first experiment
that requires a new environment will block. Subsequent experiment
requests will queue behind it in FIFO order until it can proceed (i.e. delete
an existing environment and create the one it needs)
<!-- markdownlint-enable MD007 -->

### Handling benchmark failures

Once deployments are created and the vLLM health endpoint is responding to
requests (pod running, container ready), or 20 mins has elapsed, the actuator
runs `vllm bench serve` or GuideLLM against it. The 20min timeout is so the
wait won't pend forever in a case where something goes wrong in K8s that means
the health check will never pass.

When running the benchmark the actuator will try `benchmark_retries` times
backing off exponentially based on `retries_timeout` to run the benchmark
successfully. The retries may be required as it can happen for large models that
20 minutes is not sufficient for model download and load for serving. Since vLLM
bench itself waits 10 minutes for the endpoint to come up this means with
`benchmark_retries=3` (the default) there is roughly 50mins-1hr timeout for the
endpoint to become available.

### PVCs

#### `pvc_name` not given

If no `pvc_name` is set in the `actuatorconfiguration`, when an actuator
instance is created with this configuration, e.g., via `create operation` or
`run_experiment`, it creates a PVC called `vllm-support-$UUID` that is shared by
all deployments it creates. The `$UUID` is a randomly generated string that will
vary each time the actuator is created. When the `operation` or `run_experiment`
exits this PVC will be deleted.

#### `pvc_name` given

If a `pvc_name` is set in the `actuatorconfiguration`, when an actuator instance
is created with this configuration, e.g., via `create operation` or
`run_experiment`, it will look for an existing PVC with the given name. If the
PVC exists it will be used for all deployments the actuator instance creates.
When the `operation` or `run_experiment` exits this PVC will NOT be deleted. If
the PVC does not exist the actuator will exit with an error.

### Deployment Clean-Up

The `vllm_performance` actuator will automatically clean up all Kubernetes
resources associated with the vLLM deployments as it proceeds leaving at most
`max_environments` active at a time. On a graceful shutdown of the `ado` process
running the operation (CTRL-C, SIGTERM, SIGINT) active deployments will be
deleted before exit. On an uncontrolled shutdown (SIGKILL) you will need to
manually clean up any K8s deployments that were running at the time.

> [!IMPORTANT] PVC Deletion
>
> If the actuator created a PVC (i.e. `vllm-support-$UUID`) it will be deleted.
>
> If the actuator used an existing PVC it will not be deleted.

### Kubernetes resource templates

The `vllm_performance` actuator creates Kubernetes resources based on a set of
template YAML files that are distributed with the actuator. The templates are
for:

- vLLM deployment
- PVC used by deployment pod
- vLLM service

You can use your own templates, by creating a vllm_performance
`actuatorconfiguration` resource with the following fields set to the path to
your templates.

```yaml
deployment_template: $PATH_RELATIVE_TO_WORKING_DIR
service_template: $PATH_RELATIVE_TO_WORKING_DIR
pvc_template: $PATH_RELATIVE_TO_WORKING_DIR
```

Then use this `actuatorconfiguration` resource when running operations with the
actuator.

The paths given are always interpreted relative to the working directory of
process using the actuator (where `ado create operation` or `run_experiment` is
executed).

> [!IMPORTANT] Custom templates and executing on remote RayClusters
>
> The template path must be accessible where the actuator is running.
> This is important to consider when running operation using
> `vllm_performance` on a remote RayCluster.
> To handle this we recommend:
>
> - Put custom templates in the working directory (or a subdirectory of it)
>   that you will
>   [send to the RayCluster](../getting-started/remote_run.md#other-options)
> - Create an `actuatorconfiguration` with the relative paths to the
>   templates from this working directory

### Grouped sampling for efficient deployment usage

Creating and deleting vLLM deployments takes time. If you have limited number of
vLLM deployments that can be created concurrently, say one, then this can add
significant overhead if consecutive points being sampled require different
deployments. The
[grouped sampling](../operators/random-walk.md#enabling-grouping) feature of the
`random_walk` operator can be useful in this case. This allows configuring the
sampling so points that require a given vLLM deployment are submitted in a
batch.
