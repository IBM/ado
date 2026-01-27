<!-- markdownlint-disable code-block-style -->
<!-- markdownlint-disable-next-line first-line-h1 -->
This repository contains the vLLM `ado` actuator for benchmarking LLM inference
performance with vLLM. (For more about Actuators, what they represent, how to
create them etc., see the `ado`
[docs](https://ibm.github.io/ado/actuators/working-with-actuators/)).

The actuator implements a set of functionalities to deploy and run serving
benchmarks for different LLMs for vLLM. This actuator deploys
[vLLM](https://github.com/vllm-project/vllm) on to an
[OpenShift](https://www.redhat.com/en/technologies/cloud-computing/openshift)
cluster to serve
[IBM Granite-3.3-8b](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct)
and runs an experiment that utilises the
[vLLM serving benchmark](https://docs.vllm.ai/en/stable/api/vllm/benchmarks/serve.html).
The the actuator is named `vllm_performance` and features two experiments:
`performance-testing-full` and `performance-testing-endpoint`.

# Getting Started

This guide has two parts:

- [Getting Started](#getting-started)
  - [Installing and configuring the vLLM actuator](#installing-and-configuring-the-vllm-actuator)
    - [Installation](#installation)
    - [Configuring the actuator](#configuring-the-actuator)
  - [A Simple Benchmarking Exercise](#a-simple-benchmarking-exercise)
    - [Creating a Discovery Space to describe the vLLM configurations to test](#creating-a-discovery-space-to-describe-the-vllm-configurations-to-test)
      - [Activating the local context](#activating-the-local-context)
      - [Defining a Discovery Space of vLLM configurations](#defining-a-discovery-space-of-vllm-configurations)
      - [Querying the Discovery Space](#querying-the-discovery-space)
    - [Exploring the vLLM workload configuration space](#exploring-the-vllm-workload-configuration-space)
- [Exploring Further](#exploring-further)
  - [vLLM testing approach](#vllm-testing-approach)
  - [The Actuator Package: Key Files](#the-actuator-package-key-files)
    - [Customising Actuator Configurations](#customising-actuator-configurations)
    - [Customising Experiment Protocol](#customising-experiment-protocol)
    - [Notes on the Random walk operation](#notes-on-the-random-walk-operation)
  - [A few ideas for further exploration](#a-few-ideas-for-further-exploration)

After running the exercise, please feel free to
[explore further](#exploring-further) and
[try a larger experiment](#a-few-ideas-for-further-exploration).

> [!NOTE]
>
> These prerequisites must be fulfilled before you start with this actuator
>
> 1. Access to an OpenShift cluster with at least 1 node with 1 available
>     NVIDIA GPU. You will need access to a namespace with permissions for
>     GPU-based deployments
> 2. You will need to have downloaded and installed `ado` according to
>     [this guide](https://ibm.github.io/ado/getting-started/install/).

## Installing and configuring the vLLM actuator

### Installation

Ensure the virtual environment you installed `ado` into is active. Then, run:

```commandline
pip install ado-vllm-performance
```

This will automatically install both vLLM and GuideLLM benchmarking
tools, enabling all experiments:

- `test-deployment-v1` and `test-endpoint-v1` (vLLM benchmarks)
- `test-deployment-guidellm-v1` and `test-endpoint-guidellm-v1`
  (GuideLLM benchmarks)

**For development from source:**

```commandline
pip install -e plugins/actuators/vllm_performance
```

from the root of the `ado` source repository. You can clone the repository with:

```commandline
git clone https://github.com/IBM/ado.git
```

Confirm that the actuator is installed:

```commandline
ado get actuators --details
```

You should see an output like below:

```commandline
        ACTUATOR ID        CATALOG ID                 EXPERIMENT ID  SUPPORTED
0              mock              mock               test-experiment       True
1              mock              mock           test-experiment-two       True
2  vllm_performance  vllm_performance            test-deployment-v1       True
3  vllm_performance  vllm_performance              test-endpoint-v1       True
```

On the last two lines you can see the new actuator and the experiments. You can
understand the
[constitutive properties required for the experiment](https://ibm.github.io/ado/core-concepts/actuators/#experiments)
and the
[target and observed properties](https://ibm.github.io/ado/core-concepts/actuators/#target-and-observed-properties)
measured by an experiment by running:

```commandline
ado describe experiment test-deployment-v1
```

The experiment protocol for the vLLM actuator is defined in
[this YAML file](https://github.com/IBM/ado/blob/main/plugins/actuators/vllm_performance/ado_actuators/vllm_performance/experiments.yaml).
You will need to update this if you want to modify the values that can be
accepted as valid for the input properties.

### Configuring the actuator

Before using the vLLM actuator to execute experiments, you must
configure its parameters. First, get the template for the configuration:

```commandline
ado template actuatorconfiguration --actuator-identifier vllm_performance \
                                   -o actuatorconfiguration.yaml
```

This will create the `vllm_performance_actuatorconfiguration.yaml` file, which
will look like:

```yaml
actuatorIdentifier: vllm_performance
metadata:
  description: null
  labels: null
  name: null
parameters:
  benchmark_retries: 3
  hf_token: ""
  image_secret: ""
  in_cluster: true
  interpreter: python3
  max_environments: 1
  namespace: null
  node_selector: ""
  retries_timeout: 5
  verify_ssl: false
```

The three key parameters we have to set here are `hf_token`, `namespace`, and
`node_selector`.

- `hf_token`: Access token from
  [HuggingFace](https://huggingface.co/settings/tokens).
- `namespace`: The namespace you have access to in your OpenShift cluster.
- `node_selector`: JSON dictionary representing a Kubernetes selector for a node
  with available GPUs. Make sure it is formatted correctly, for example:

    ```text
    node_selector: '{"kubernetes.io/hostname":"cpu16"}'
    ```

We will discuss the other parameters later. Once you have put in the parameters,
create the actuator configuration with:

```commandline
ado create actuatorconfiguration -f `vllm_performance_actuatorconfiguration.yaml`
```

Note: You can have multiple configurations for an actuator.

## A Simple Benchmarking Exercise

To get started, we have provided an exercise to run a benchmarking experiment
for a single vLLM deployment configuration. The instructions for this exercise
assume you are running `ado` from a machine outside of the target
Kubernetes/OpenShift cluster.

### Creating a Discovery Space to describe the vLLM configurations to test

> [!NOTE]
>
> Since this is an example exercise, we will use the `local` context and the
> `default` sample store.

#### Activating the local context

To ensure the `local` context is active, run:

```commandline
ado context local
```

#### Defining a Discovery Space of vLLM configurations

`ado` uses the concept of
[Discovery Spaces](https://ibm.github.io/ado/core-concepts/concepts/) to
describe what to test (in this case vLLM workload configurations) and how to
test them (the vLLM benchmark(s) to run).

The set of configurations to test is defined by the
[entity space](https://ibm.github.io/ado/core-concepts/entity-spaces/), and the
set of experiments to perform by the
[measurement space](https://ibm.github.io/ado/core-concepts/actuators#measurementspace/).

An example `discoveryspace` for vLLM inference benchmarking can be found in
[`yamls/discoveryspace_override_defaults.yaml`](yamls/discoveryspace_override_defaults.yaml).
This defines a simple discovery space with a **single** entity.

Our sample space will benchmark vLLM serving the LLM specified by `model_name`,
on a node (determined through `node_selector`) with a specific GPU
(`NVIDIA-A100-80GB-PCIe`) specified in `gpu_type`.

> [!NOTE]
>
> Ensure that the GPU specified in `gpu_type` is present on the node. To find
> out the gpu model of your selected node, try the following command:
>
> ```commandline
> oc describe node <node name> | grep "nvidia.com/gpu.product"
> ```
>
> If this returns a different GPU model, then you must
> [update the experiment protocol](#customising-experiment-protocol).

Create the `discoveryspace`:

```commandline
ado create space -f yamls/discoveryspace_override_defaults.yaml \
                 --use-default-sample-store
```

#### Querying the Discovery Space

Before we run any experiment, we can see that the `discoveryspace` is empty:

```commandline
ado show entities space --use-latest
```

Will output:

<!-- markdownlint-disable line-length -->
```text
Nothing was returned for entity type matching and property format observed in space space-c81773-df57a3.
```
<!-- markdownlint-enable line-length -->

To see all the entities (parameter combinations) that are waiting to be
measured, try executing:

```commandline
ado show entities space --include missing --use-latest
```

The output will look like:

<!-- markdownlint-disable line-length -->
```terminaloutput
   model                             image                                           n_cpus  memory dtype  num_prompts  request_rate  max_concurrency  gpu_memory_utilization  cpu_offload  max_batch_tokens  max_num_seq  n_gpus  gpu_type
0  ibm-granite/granite-3.3-8b-instruct  quay.io/dataprep1/data-prep-kit/vllm_image:0.1  8.0     128Gi  auto   500.0        -1.0          -1.0             0.9                     0.0          16384.0           256.0        1.0     NVIDIA-A100-80GB-PCIe
```
<!-- markdownlint-enable line-length -->

Which is the entity we want to measure.

### Exploring the vLLM workload configuration space

First, log in to your OpenShift cluster and select your assigned namespace

```commandline
oc login <your OpenShift API endpoint>
oc project <your assigned namespace>
```

Next, we'll set up the operation to measure our entity defined above.

In `ado` parlance, measurements are executed through `operations` which
represent the executions of `experiments` on `entities`.

An example of an operation can be found in
[`yamls/random_walk_operation.yaml`](https://github.com/IBM/ado/blob/main/plugins/actuators/vllm_performance/yamls/random_walk_operation.yaml).
You can run the operation using the actuator configuration and space that we
have created earlier with:

<!-- markdownlint-disable line-length -->
```commandline
ado create operation -f yamls/random_walk_operation.yaml \
                     --use-latest space --use-latest actuatorconfiguration
```
<!-- markdownlint-enable line-length -->

`ado` will initialise a local Ray cluster and starts the measurement at the
point where these lines appear:

<!-- markdownlint-disable line-length -->
```terminaloutput
...
=========== Starting Discovery Operation ===========

(RandomWalk pid=2780) 'all' specified for number of entities to sample. This is 1 entities - the size of the entity space
...
```
<!-- markdownlint-enable line-length -->

The actuator uses the entity to create a vLLM deployment, followed by execution
of the benchmark script. This process will take some time as it involves
downloading the container image from [Quay](quay.io) and the model from
HuggingFace, both of which are network-intensive. You can monitor if the
deployment is ready by executing the following in another shell:

```commandline
oc get deployments --watch
```

The experiment is successfully completed if the `ado` output is similar to the
following:

<!-- markdownlint-disable line-length -->
```text
(RandomWalk pid=46852) Continuous Batching: EXPERIMENT COMPLETION. Received finished notification for experiment in measurement request in group 0: request-4332aa-experiment-performance-testing-entities-model.ibm-granite/granite-3.3-8b-instruct-image.quay.io/dataprep1/data-prep-kit/vllm_image:0.1-n_cpus.8-memory.128Gi-dtype.auto-num_prompts.500-request_rate.-1-max_concurrency.-1-gpu_memory_utilization.0.9-cpu_offload.0-max_batch_tokens.16384-max_num_seq.256-n_gpus.1-gpu_type.NVIDIA-A100-80GB-PCIe (explicit_grid_sample_generator)-requester-randomwalk-0.9.7.dev10+b7a010dd.dirty-42ad60-time-2025-08-11 15:53:54.137571+01:00
(RandomWalk pid=46852) Continuous batching: GET EXPERIMENT. No new experiments in queue. Requests made: 1. Experiments Completed: 1
```
<!-- markdownlint-enable line-length -->

If the output contains `EXPERIMENT FAILURE`, then something has gone wrong.

Verify that the entity has been measured by running:

```commandline
ado show entities space --use-latest --output-format csv
```

The csv file will have one line representing the entity featuring values for all
its measured properties
(`performance-testing-output_throughput`,`performance-testing-total_token_throughput`,`performance-testing-mean_ttft_ms`,
etc.)

Congratulations! You have successfully executed the vLLM benchmark on a vLLM
workload configuration using `ado`!

# Exploring Further

## vLLM testing approach

vLLM testing implementation is based on this
[guide](https://github.com/vllm-project/vllm/discussions/7181) which is using
[benchmark_serving.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py)
to implement the actual benchmarking. The benchmarking is done using HTTP
requests using `vLLM OpenAI API server`.

To use this approach it is necessary to:
<!-- markdownlint-disable descriptive-link-text -->
- Create a docker image: Existing docker images for VLLM project are not
  directly suitable for this purpose, as they are hard to use on Openshift
  clusters and not directly extensible. We have provided a Docker image to get
  started but if you want to customize it for your installation, then you will
  need to rebuild it. We provide a slightly different
  [build](docker_image), described [here](docker_image/README.md)
- Create automation for vLLM deployment for running experiments. A simple
  implementation of such an automation is presented
  [here](ado_actuators/vllm_performance/k8)
- Create a vLLM performance test. Here we are directly reusing
  [performance test](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py)
  provided by the vLLM project. The required code is
  [here](ado_actuators/vllm_performance/vllm_performance_test)
<!-- markdownlint-enable descriptive-link-text -->

This figure shows the outline of the components and the parameters available for
configuring each of them

![vLLM_testing](vllm_testing.png)

The test results in the figure are the measurements recorded for the entity. The
deployment parameters form the configuration space. Test parameters are
partially inferred from the configuration space and partially from the context
(Kubernetes endpoints, etc.)

## The Actuator Package: Key Files

The actuator package is under `ado_actuators/vllm_performance`. Note all
actuator packages should be placed under a directory called `ado_actuators` as
this is the name of package that contains all `ado` plugins.

The key files are:

- actuator_definitions.yaml
  - This defines which classes in which modules of your package contain
    Actuators.
- actuators.py
  - Implementation of the actuator logic.
  - It just needs to be the same name as in `actuator_definitions.yaml`
- experiments.yaml
  - This file contains the definitions of the experiments the actuator defines
    as YAML
- experiment_executor.py (OPTIONAL)
  - This file contains the code that
    - determines the values for the experiment parameters from the passed Entity
      and Experiment
    - execute the experiment and get measured property values
    - sends the measured property values back to the orchestrator

### Customising Actuator Configurations

The actuator is configured using
[VLLMPerformanceTestParameters class](ado_actuators/vllm_performance/actuator_parameters.py)

You can customise `deployment_template`, `service_template` and `pvc_template`
for your OpenShift/K8s cluster. Refer to the
[default yamls](ado_actuators/vllm_performance/k8/yaml_support) for the
templates referred to in [Configuring the actuator](#configuring-the-actuator)
and modify them appropriately

If you create a custom Docker image and upload it to a repository, please do not
forget to create a corresponding Image pull secret in your assigned namespace.
You must also update the value of the `image_secret` parameter of the actuator
configuration.

### Customising Experiment Protocol

The values for the parameters in the entity space must be a subset of the
acceptable values defined for the experiment (_the experiment protocol_).
Therefore, depending on your environment and use case, you may need to update
the set of values to expand the configuration space being studied.

For example, you may want to benchmark a different LLM or you may want to change
the GPU type to the one installed in your cluster. In the former case, you will
add values to `model_name` and in the latter case, you will have to modify the
domain of the `gpu_type` parameter to avoid validation errors.

To do this, open the
[experiment definition YAML file](ado_actuators/vllm_performance/experiments.yaml)
in a text editor, and add your GPU model to the list of values of `gpu_type`.

Then, reinstall this actuator by running:

```commandline
pip install .
```

After that, you can use the new value of `gpu_type` in your experiments. For
example, in
[the sample space definition file](yamls/discoveryspace_override_defaults.yaml),
the location to update will be:

```yaml
- identifier: "gpu_type"
  propertyDomain:
    values: ["NVIDIA-A100-80GB-PCIe"]
```

### Notes on the Random walk operation

VLLM testing is using external environment (deployment + service) to run tests.
Creating such an environment is resource-intensive. To speed up experiments
execution it is recommended to use group samplers for running VLLM testing. This
allows to create an environment once and use it for all experiments that can be
used for it. In this case the group definition looks as follows:

```yaml
grouping:
  - model
  - image
  - n_gpus
  - gpu_type
  - n_cpus
  - memory
  - max_batch_tokens
  - gpu_memory_utilization
  - dtype
  - cpu_offload
  - max_num_seq
```

<!-- markdownlint-disable descriptive-link-text -->
For the complete example of configuring random walk operation for the group
samplers, look [here](yamls/random_walk_operation_grouped.yaml)
<!-- markdownlint-enable descriptive-link-text -->

## A few ideas for further exploration

Try:

- Testing throughput for different sequence length for multiple models using
  this actuator (See
  [discoveryspace_override_defaults_small.yaml](yamls/discoveryspace_override_defaults_small.yaml)
  for an example with multiple values for `max_batch_tokens`)
