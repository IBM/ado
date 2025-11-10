# The `vllm_performance` actuator

<!-- markdownlint-disable line-length -->

> [!TIP] Overview
> The `vllm_performance` actuator **can
> automatically create and benchmark [vLLM](https://github.com/vllm-project/vllm) inference deployments on Kubernetes and OpenShift clusters**.
>
> It is designed for robust, repeatable, and configurable experiment execution.
> It is suitable for both simple one-off benchmarks and large parameter sweeps.
<!-- markdownlint-enable line-length -->

## Key Capabilities

- **Automated LLM benchmarking:** Deploys vLLM serving endpoints
on GPU-enabled Open/Shift Kubernetes clusters and runs
standardized serving benchmarks.
- **Cluster integration:** Handles deployments and clean-up of vLLM inference
pods on  OpenShift/Kubernetes, with configurable resource selection via namespace,
node selector,  and PVC/service templates.
- **Scenario configurability:** Supports customizing models, GPU types, node selection,
retry behavior, concurrent deployments, and more
- **Efficient sampling:** Supports grouped sampling which maximises reuse
of vLLM deployments, hence minimising time spent creating such deployments
- **Endpoint benchmarking:** Can also be used to benchmark existing OpenAI
compatible endpoints

### Available experiments

The `vllm_performance` actuator implements two experiments

- `performance-testing-full`: This experiment can test the full vLLM workload configuration,
including resource requests and server deployment configuration. It deploys
servers with given configuration on kubernetes and runs `vllm bench serve` on them
with the given parameters
- `performance-testing-endpoint`: This experiment is equivalent to running
`vllm bench serve` against an endpoint.

---

## Running single experiments: Quick endpoint and deployment tests

For rapid testing and debugging, you can use the [`run_experiment`](run_experiment.md)
tool to execute individual  experiments on a single point (entity).
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
  experimentIdentifier: performance-testing-endpoint
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
- The identifier of a valid `actuatorconfiguration` resource
    - This contains information necessary for accessing and creating
     deployments on the Kubernetes/OpenShift cluster
    - See [configuring the vllm_performance actuator](#configuring-the-vllm_performance-actuator)
      for details.
<!-- markdownlint-enable MD007 -->

Example `point.yaml`:

```yaml
entity:
  model: ibm-granite/granite-3.3-8b-instruct
  n_cpus: 8
  memory: 128Gi
  gpu_type: NVIDIA-A100-80GB-PCIe
  max_batch_tokens: 8192
  max_num_seq: 32
  n_gpus: 1
experiments:
- actuatorIdentifier: vllm_performance
  experimentIdentifier: performance-testing-full
```

Then run:

```shell
run_experiment point.yaml  --actuator-config-id my-vllm-performance-config
```

Here `my-vllm-performance-config` is the ID of an `actuatorconfiguration` resource
containing the details for accessing and running on your target cluster.
See [configuring the vllm_performance actuator](#configuring-the-vllm_performance-actuator)
for more.

This command will provision the deployment for the specified entity, using your indicated
actuator configuration, run the benchmark, and print results.

> [!TIP] vLLM deployment example
>
> See [the vLLM deployment exploration example](../examples/vllm-performance-full.md)
> for details on how to explore many deployment configurations.

---

## Configuring the vllm_performance actuator

In order for the `vllm_performance` actuator to create,
manage, and monitor vLLM deployments on a Kubernetes/OpenShift
cluster, you must provide some configuration information
This configuration covers several needs:

- **Cluster targeting and permissions**: Specify the OpenShift/Kubernetes namespace
and optionally node selectors, secrets, and templates to match your cluster resources.
- **Secure access**: Pass required HuggingFace tokens, set up image pull secrets,
control in-cluster or remote execution, and toggle SSL verification.
- **Experiment protocol and retries**: Choose how benchmarks are run, including interpreter,
retry logic, and YAML templates for deployments/services used.
- **Deployment resource management**: Limit the number of concurrent deployments
and control automated clean-up.

You supply this configuration information as an `ado`
[`actuatorconfiguration` resource](../resources/actuatorconfig.md),
which is a YAML file with the configuration options.
An example is:

```yaml
actuatorIdentifier: vllm_performance #The actuator the configuration is for
actuatorIdentifier: vllm_performance #The actuator the configuration is for
metadata:
  description: "Actuator config for vLLM LLM benchmarking"
  name: demo-vllm-perf
parameters:
  benchmark_retries: 3              # Number of benchmark attempts (see Failure Handling)
  deployment_template: deployment.yaml  # k8s deployment spec template
  hf_token: "<YOUR_HUGGINGFACE_TOKEN>" # Required for pulling some models
  image_secret: ""                 # Optional image pull secret
  in_cluster: true                  # Run from within the cluster
  interpreter: python3              # Language for test drivers/benchmarks
  max_environments: 1               # Max concurrent vLLM deployments
  namespace: "mynamespace"          # OpenShift/K8s namespace to deploy into
  node_selector: '{"kubernetes.io/hostname":"gpunode01"}' # Restricts GPU node
  pvc_template: pvc.yaml            # Persistent volume claim template
  retries_timeout: 5                # Seconds between retries (exponential backoff)
  service_template: service.yaml    # k8s service spec template
  verify_ssl: false                 # Whether to verify HTTPS endpoints
```

If the above YAML was saved to a file called `vllm_config.yaml` you would create
the configuration using

```commandline
ado create actuatorconfiguration -f vllm_config.yaml
```

> [!TIP] Getting a default configuration
>
> You can generate a default configuration via the ado CLI:
>
> ```shell
> ado template actuatorconfiguration --actuator-identifier vllm_performance -o actuatorconfiguration.yaml
> ```

### Configuration option details
<!-- markdownlint-disable MD007 -->
- `actuatorIdentifier`: Always set to `vllm_performance` for this actuator.
- `metadata`: Descriptive metadata for organization or tracking.
- **parameters:**
    - `benchmark_retries`: Number of times a benchmark can be retried if it fails
    (see Handling benchmark failures)
    - `deployment_template`, `service_template`, `pvc_template`: YAML templates for
        k8s resources created by the actuator
    - `hf_token`: [HuggingFace token](https://huggingface.co/settings/tokens)
        for protected model downloads
    - `image_secret`: Kubernetes secret name for private registry images
    - `in_cluster`: Whether to execute inside the cluster for better network access
    - `interpreter`: Python interpreter or path
    - `max_environments`: Maximum number of deployments to create concurrently
        (see Maximum number of deployments)
    - `namespace`: Namespace to use for deployments
    - `node_selector`: Kubernetes node label for targeting e.g. GPU nodes
    - `retries_timeout`: Timeout in seconds for exponential backoff between retries
    - `verify_ssl`: Toggle SSL certificate verification for endpoints
<!-- markdownlint-enable MD007 -->
> [!IMPORTANT] Further details
>
> For further details on specific options and advanced behavior see:
>
> - [Maximum number of deployments](#maximum-number-of-deployments) (details on `max_environments`)
> - [Handling benchmark failures](#handling-benchmark-failures) and [Deployment Clean-Up](#deployment-clean-up)
> - [Grouped sampling for efficient deployment usage](#grouped-sampling-for-efficient-deployment-usage)

### Multiple configurations

You can create multiple `actuatorconfiguration`s for the `vllm_performance` actuator.
Each configuration captures
the cluster-specific, security-sensitive, and experiment-relevant settings necessary
for the actuator to operate in a given environment.
Each configuration will have a different id and you can choose the one to use
when submitting an operation or single experiment that uses the `vllm_performance`
actuator.

---

## vLLM deployment management

### The `in_cluster` configuration option

The `in_cluster` option in your `actuatorconfiguration` tells the `vllm_performance`
actuator how to communicate with the target Kubernetes or OpenShift cluster when
running `performance-testing-full`.

If running `ado` from outside the Kubernetes/OpenShift cluster where
the deployments will be created, leave `in_cluster: false` (the default).

Set `in_cluster: true` if your `ado` operation will be run on a
**remote Ray cluster that is in the same Kubernetes/OpenShift cluster** as your
vLLM deployments.
This configuration maximizes efficiency for large-scale, distributed benchmarking.
For a detailed guide on running `ado` remotely on a Ray cluster, including environment
and package setup, see [Running ado remotely](../getting-started/remote_run.md).

> [!IMPORTANT] RayCluster permissions
>
> If running with `in_cluset=True`, your RayCluster **must** be configured so that
> jobs launched  by `ado` have permissions to create and manage Kubernetes deployments,
> pods, and services.
> For configuring the necessary ServiceAccount, roles, and permissions,
> see our [documentation on deploying RayClusters for `ado`](../getting-started/installing-backend-services.md).

### Maximum number of deployments

The actuator configuration parameter `max_environments` controls how
many concurrent vLLM deployments will be created. The default is 1.

When experiments are requested, if an existing deployment cannot
be used a new environment is created as long as `max_environments` has
not been reached.
If it has been reached, then the actuator waits for an existing
environment to become idle, at which point it is deleted and
the new environment is created.

Some notes:

- `max_environments` deployments are always created before any are deleted
  - This means idle environments will remain until there is a need to delete them
  - This is to increases chances they can be reused/minimise cost of redeploying
- Environment creation is serialized
  - If `max_environments` is reached and all are active, the first experiment
      that requires a new environment will block. Subsequent experiment
      requests will queue behind it in FIFO order until it can proceed (i.e. delete
      an existing environment and create the one it needs)

### Handling benchmark failures

Once deployments are created and the vLLM health endpoint is responding to requests
(pod running, container ready), or 20 mins has elapsed, the actuator runs
`vllm bench serve` against it.
The 20min timeout is so the wait won't pend forever in a case where something
goes wrong
in k8s that means the health check will never pass.

When running the benchmark the actuator will try `benchmark_retries` times
backing off exponentially based  on `retries_timeout` to run the benchmark successfully.
The retries may be required as it can happen for large models that 20minutes is
not sufficient for model download and load for serving.
Since vLLM bench itself waits 10mins for the endpoint to come up this means with
`benchmark_retries=3` (the default) there is roughly 50mins-1hr timeout for the
endpoint to become available.

### Deployment Clean-Up

The `vllm_performance` actuator will automatically clean up
vLLM deployments as it proceeds leaving at most `max_environments`
active at a time.
On a graceful shutdown of the `ado` process running the operation
(CTRL-C, SIGTERM, SIGINT) active deployments will be deleted
before exit.
On an uncontrolled shutdown (SIGKILL) you will need to manually
clean up any k8s deployments that were running  at the time

### Grouped sampling for efficient deployment usage

Creating and deleting vLLM deployments takes time.
If you have limited number of vLLM deployments that can be
created concurrently, say one, then this can add significant
overhead if consecutive points being sampled require
different deployments.
The [grouped sampling](../operators/random-walk.md#enabling-grouping)
feature of the `random_walk` operator can be useful in this case.
This allows configuring the sampling so points that
require a given vLLM deployment are submitted in a batch.
