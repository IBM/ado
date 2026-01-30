# Exploring vLLM deployment configurations

> [!NOTE] The scenario
>
> **In this example, the
> [_vllm_performance_ actuator](../actuators/vllm_performance.md) is used to
> evaluate different vLLM server deployment configurations on
> Kubernetes/OpenShift.**
>
> When deploying vLLM, you must choose values for parameters like GPU type,
> batch size, and memory limits. These choices directly affect performance,
> cost, and scalability. To find the best configuration for your workload,
> whether you are optimizing for latency, throughput, or cost, you need to
> explore the deployment parameter space. In this example:
>
> - We will define a space of vLLM deployment configurations to test with
> the `vllm_performance` actuator's `test-deployment-v1` experiment
>       - This experiment can create and characterize a vLLM deployment on Kubernetes
> - Use the [`random_walk` operator](../operators/random-walk.md) to
>   explore the space
<!-- markdownlint-disable-next-line MD028 -->

> [!IMPORTANT] Prerequisites
>
> - Be logged-in to your Kubernetes/OpenShift cluster
> - Have access to a namespace where you can create vLLM deployments
> - Install the following Python packages locally:
>
> ```bash
> pip install ado-vllm-performance
> ```

<!-- markdownlint-disable-next-line MD028 -->

> [!TIP] TL;DR
>
> Get the files `vllm_deployment_space.yaml`, `vllm_actuator_configuration.yaml`
> and `operation_random_walk.yaml` from
> [our repository.](https://github.com/IBM/ado/tree/main/plugins/actuators/vllm_performance/yamls)
>
> **You must edit `vllm_actuator_configuration.yaml` with your details.** In
> particular the following two fields are important:
>
> <!-- markdownlint-disable line-length -->
> ```yaml
> hf_token: <your HuggingFace access token> # Required to access gated models
> namespace: vllm-testing # you MUST set this to a namespace where you can create vLLM deployments
> ```
>
> Then, in a directory with these files, execute:
>
> ```bash
> : # Note: this will create space and actuator conf resources you can reuse subsequently
> ado create op -f random_walk_operation_grouped.yaml \
>    --with space=vllm_deployment_space.yaml --with ac=vllm_actuator_configuration.yaml
> ```
> <!-- markdownlint-enable line-length -->
>
> See
> [configuring the `vllm_performance` actuator](../actuators/vllm_performance.md#configuring-the-vllm_performance-actuator)
> for more configuration options.

## Verify the installation

Verify the installation with:

```commandline
ado get actuators --details
```

The actuator `vllm_performance` should appear in the list of available actuators
if installation completed successfully.

## Create an actuator configuration

The vllm-performance actuator needs some information about the target cluster to
deploy on. This is provided via an `actuatorconfiguration`.

First execute:

```commandline
ado template actuatorconfiguration --actuator-identifier vllm_performance -o vllm_actuator_configuration.yaml
```

This will create a file called `vllm_actuator_configuration.yaml`

Edit the file and set correct values for at least the `namespace` field. Also
consider if you need to supply a value for `hf_token` :

<!-- markdownlint-disable line-length -->

```yaml
hf_token: <your HuggingFace access token> # Required to access gated models
namespace: vllm-testing # you MUST set this to a namespace where you can create vLLM deployments
```

<!-- markdownlint-enable line-length -->

Then save this configuration as an `actuatorconfiguration` resource:

```bash
ado create actuatorconfiguration -f vllm_actuator_configuration.yaml
```

> [!TIP]
>
> You can create multiple actuator configurations corresponding to different
> target environments. You choose the one to use when you launch an operation
> requiring the actuator.

## Define the configurations to test

When exploring vLLM deployments there are two sets of parameters that can be
changed:

- the deployment creation parameters (number GPUs, memory allocated etc)
- the benchmark test parameters (request per second to send, tokens per request
  etc.)

In this case we define a space where we look at the impact of a few vLLM
deployment parameters, including `max_num_seq` and `max_batch_tokens`, for a
scenario where requests arrive between 1 and 10 per second with sizes around
2000 tokens.

```yaml
{%
  include "./example_yamls/vllm_deployment_space.yaml"
%}
```

Save the above as `vllm_deployment_space.yaml`. Then run:

```bash
ado create space -f vllm_deployment_space.yaml
```

## Explore the space with random_walk

Next, we'll scan this space sequentially using a `grouped` sampler to increase
efficiency. The `grouped` sampler ensures we explore all the different benchmark
configurations for a given vLLM deployment before creating a new deployment -
minimizing the number of deployment creations.

Save the following as `random_walk_operation_grouped.yaml`:

```yaml
{%
  include "./example_yamls/random_walk_operation_grouped.yaml"
%}
```

Then, start the operation with:

```commandline
ado create operation -f random_walk_operation_grouped.yaml \
           --use-latest space --use-latest actuatorconfiguration
```

As it runs a table of the results is updated live in the terminal as they come
in.

### Monitor the optimization

While the operation is running you can monitor the deployment:

```bash
# In a separate terminal
oc get deployments --watch -n vllm-testing
```

You can also get the results table by executing (in another terminal)

```commandline
ado show entities operation --use-latest
```

### Check final results

When the output indicates that the experiment has finished, you can inspect the
results of all operations run so far on the space with:

```commandline
ado show entities space --output-format csv --use-latest
```

> [!NOTE]
>
> At any time after an operation, $OPERATION_ID, is finished you can run
> `ado show entities operation $OPERATION_ID` to see the sampling time-series of
> that operation.

## Next steps

<!-- markdownlint-disable MD028 -->
- Try running the same operation with the
  [GuideLLM](https://github.com/vllm-project/guidellm) benchmarking tool
  by setting the `experimentIdentifier` field in the entity space definition
  to `test-deployment-guidellm-v1`.
- Try varying **`max_batch_tokens`** or **`gpu_memory_utilization`** to explore
  the impact on throughput.
- Try creating a different `actuatorconfiguration` with more `max_environments`
  and running the random walk with a non-grouped sampler
- Replace the model with a different HF checkpoint to compare performance.
- Use the [**RayTune** operator](../operators/optimisation-with-ray-tune.md)
  (see the [vLLM endpoint performance](vllm-performance-endpoint.md) example)
  to find "best" configurations.
- Run a  [multi-objective optimization](../operators/optimisation-with-ray-tune.md#multi-objective-optimization)
  to explore e.g. latency v throughput tradeoffs.
- Run
  [the exploration on the OpenShift/Kubernetes cluster](../actuators/vllm_performance.md#the-in_cluster-configuration-option)
  you create the deployments on, so you don't have to keep your laptop open.
- Check the
[`vllm_performance` actuator documentation](../actuators/vllm_performance.md)
<!-- markdownlint-enable MD028 -->
