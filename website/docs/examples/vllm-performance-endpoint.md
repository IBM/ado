# Testing the throughput of an inference endpoint

> [!NOTE] The scenario
>
> **In this example,
> the [_vllm_performance_ actuator](../actuators/vllm_performance.md)
> is used to find
> the maximum requests per second a server can handle while maintaining
> stable maximum throughput.**
>
> A model deployed for inference will have a certain max stable throughput in
> terms of the requests it can serve per second.
> Sending more requests than this maximum will often lead to a drop in throughput.
> Hence, it can be useful to know what this maximum is so the maximum throughput
> is reliably maintained e.g. by limiting
> the max number of concurrent requests.
>
> To explore this space, you will:
>
> - define an endpoint, model and range of requests per second to test
> - use [an optimizer](../operators/optimisation-with-ray-tune.md)
> to efficiently find the maximum requests per second
<!-- markdownlint-disable-next-line MD028 -->

> [!IMPORTANT] Prerequisites
>
> - An endpoint serving an LLM via an OpenAI API-compatible API  
> - Install the following Python packages:
>
> ```bash
> pip install hyperopt
> pip install ado-ray-tune
> pip install ado-vllm-performance
> ```
<!-- markdownlint-disable-next-line MD028 -->

> [!TIP] TL;DR
>
> Get the files `vllm_request_rate_space.yaml` and `operation_hyperopt.yaml`
> from [our repository](https://github.com/IBM/ado/tree/main/plugins/actuators/vllm_performance/yamls).
>
> - `vllm_request_rate_space.yaml`: this file defines the _endpoint_, _model_,
> and _request_ _range_ to explore.
> <!-- markdownlint-disable MD007 -->
> <!-- markdownlint-disable MD046 -->
>     - **You must edit the _model_ and _endpoint_ fields in this file
>        to match your own.**
> <!-- markdownlint-enable MD046 -->
> <!-- markdownlint-enable MD007  -->
> - `operation_hyperopt.yaml`: this file contains the optimization parameters.
>    You do not need to edit it.
>
> Then, in a directory with these files, execute:
>
> ```bash
> : # Note: this will create a discoveryspace resource you can reuse subsequently
> ado create operation -f operation_hyperopt.yaml --with space=vllm_request_rate_space.yaml
> ```

## Verify the installation

Execute:

```commandline
ado get actuators --details 
```

If the prerequisites (see above) have been installed correctly the actuator
`vllm_performance` will appear in the list of available actuators

## Define the request rates to test

The file
[`vllm_request_rate_space.yaml`](https://github.com/IBM/ado/tree/main/plugins/actuators/vllm_performance/yamls/])
defines a space with all request rates from 10 to 100 for an endpoint serving `gpt-oss-20b`.
It's contents are:

```yaml
# Example discovery space for vLLM performance
entitySpace:
  - identifier: model
    propertyDomain:
      values:
        - openai/gpt-oss-20b
  - identifier: endpoint
    propertyDomain:
      values:
        - http://localhost:8000
  - identifier: request_rate
    propertyDomain:
      domainRange: [10,100]
      interval: 1
experiments:
- actuatorIdentifier: vllm_performance
  experimentIdentifier: test-endpoint-v1
```

Create the space with:

```bash
ado create space -f vllm_request_rate_space.yaml
```

> [!NOTE]
>
> More complex `discoveryspace`s can be created,
> for example also including the number of input tokens.
> See [Next Steps](#next-steps).

## Use hyperopt to find the best input request rate

[Hyperopt](http://hyperopt.github.io/hyperopt/) uses
[Tree-Parzen Estimators (TPE)](https://proceedings.neurips.cc/paper_files/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf)
which is a bayesian approach that is expected to be good for discrete dimensions
and noisy metrics, which we have here i.e. `request_throughput`.

<!-- markdownlint-disable-next-line MD013 -->
The file [operation_hyperopt.yaml](https://github.com/IBM/ado/tree/main/plugins/actuators/vllm_performance/yamls/) defines an optimization that
will look for points (in this case `request_rate`s)that result in a `request_throughput`
within the top 20th percentile.
The files contents look like:

```yaml
spaces:
  - <will be filled by ado>
operation:
  module:
    operatorName: "ray_tune"
    operationType: "search"
  parameters:
    tuneConfig:
      metric: "request_throughput" # The metric to optimize
      mode: 'max'
      num_samples: 16
      search_alg:
        name: hyperopt
        n_initial_points: 8 #Number of points to sample before optimizing 
        gamma: 0.25 #The top gamma fraction of measured values are considered "good"
```

Create the operation with:

```commandline
ado create operation -f hyperopt.yaml --use-latest space
```

Results will appear as they are measured.

> [!NOTE]
>
> Hyperopt samples with replacement so you may see the same points
> sampled twice.
> The likelihood increases as the number of points in the space decreases
> The likelihood increase as number of points in the space decreases

### Monitor the optimization

You can see the measurement requests as the operation runs
by executing (in another terminal):

```commandline
ado show requests operation --use-latest 
```

and the results (this outputs the entities in sampled order):

```commandline
ado show entities operation --use-latest 
```

Instead of `--use-latest` you can also supply the operation id directly.

### Check final results

When the output indicates that the experiment has finished, you
can inspect the results of all operations run so far on the space with:

```commandline
ado show entities space --output-format csv --use-latest
```

> [!NOTE]
>
> At any time after an operation, $OPERATION_ID, is finished you can run
> `ado show entities operation $OPERATION_ID`
> to see the sampling time-series of that operation.

## Some notes on hyperopt and TPE

What you should observe is that as the search proceeds **hyperopt**
will begin to prefer sampling points in the region with stable maximum,
even if it has seen better values in "unstable" regions.

> [!IMPORTANT]
>
> Do not just take the best point found by hyperopt but look at where it was
> focusing its attention

TPE builds models of where the "good" regions and "bad" regions of the
discovery space are i.e. `P(x|good)`, `P(x|bad)`, where x is an input point.
It then chooses new points to test by maximizing  `P(x|good)/P(x|bad)`

This makes TPE robust to noise in request_throughput as it is not trying to find
where the maximum is but is trying to find the request_rates that are most likely
to give high throughput (above defined as throughput in top 20 percentile).
This also makes it robust to outliers.

Issues may arise if the optimal region is not sampled in the initial points
and this region  is disjoint from other regions with "good" performance.
As the search runs it will be directed towards where it has already seen good values
and the best region is unlikely to be visited.

> [!TIP]
>
> The number of samples hyperopt will use for first guess of good region
> is (n_initial_points)*gamma -> in above case 2, the other will be used
> for the "bad" region

## Next steps

<!-- markdownlint-disable MD007 -->
- Try running the same operation with the
  [GuideLLM](https://github.com/vllm-project/guidellm) benchmarking tool
  by setting the `experimentIdentifier` field in the entity space definition
  to `test-endpoint-guidellm-v1`.
- Use `ado describe experiment vllm_performance_endpoint` to see what
other parameters can be explored
- Try varying **`burstiness`** or **`number_input_tokens`**, or adding
them as dimensions of the `entityspace`, to explore their impact on throughput
- Try varying `num_samples`, `gamma` and `n_initial_points` parameters of hyperopt
    - You can keep running the optimization on the same `discoveryspace`.
       The previous runs will not influence new runs, but their results will
        be reused, speeding experimentation up
- Measure the [performance of vLLM deployment configurations](vllm-performance-full.md)
- Check the [`vllm_performance` actuator documentation](../actuators/vllm_performance.md)

<!-- markdownlint-enable MD007 -->
