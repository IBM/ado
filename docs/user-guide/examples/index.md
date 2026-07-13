<!-- markdownlint-disable-next-line first-line-h1 -->
## Introductory examples

- **[Your first ado experiment](density-example.md)** — Write a Python
  function as a custom experiment, define a discovery space, and run an
  operation end-to-end. Start here if you are new to `ado`.

- **[Taking a random walk](random-walk.md)** — Walk through the core `ado`
  workflow — defining a `discoveryspace`, running an `operation` to sample and
  measure points, and retrieving the results — using a real cloud workload
  dataset.

## Search

These examples show how to use `ado` to search a space for the best
configuration. They build on the tutorial and introduce the `ray_tune` operator,
which gives access to the RayTune optimisation framework.

- **[Search a space with an optimizer](best-configuration-search.md)** — Use
  `ray_tune` with a pluggable optimiser to find the minimum of a test function.
  Also covers creating custom experiments and using parameterisable experiments.
- **[Search based on a custom objective](search-custom-objective.md)** — Define
  a dependent experiment that derives a new metric (e.g. cost) from the output
  of another experiment, then search the combined space.

## Analysis

These examples show how to use `ado` to analyse and model a configuration space
after, or instead of, exhaustive measurement.

- **[Identify the important dimensions of a space](lhu.md)** — Use the
  Latin-Hypercube sampler and the `InformationGain` stopper from `ray_tune` to
  rank which entity-space dimensions most influence a target metric, stopping
  automatically once the ranking stabilises.
- **[Quickly building a predictive model](trim.md)** — Use the TRIM operator to
  intelligently sample just enough points to train an accurate `AutoGluon`
  surrogate model, stopping once the model quality plateaus. The resulting model
  can be used for prediction at unmeasured points.

## Fine-tuning

These examples use the [SFTTrainer](../actuators/sft-trainer.md) actuator to
benchmark LLM fine-tuning throughput across a workload parameter space (model
name, batch size, max sequence length, etc.). Start with the local example and
then scale up to a remote cluster.

- **[Measure fine-tuning throughput locally](finetune-locally.md)** — Explore a
  fine-tuning parameter space on a laptop without GPUs using the
  `finetune_full_benchmark-v1.0.0` experiment and the `random_walk` operator.
- **[Measure fine-tuning throughput on a RayCluster](finetune-remotely.md)** —
  Scale the same exploration to a remote RayCluster with GPUs. Assumes
  completion of the local example.

## vLLM Performance

These examples use the [vllm_performance](../actuators/vllm-performance.md)
actuator to benchmark foundation model inference — from a single live endpoint
to full Kubernetes/OpenShift deployments and specialised geospatial models.

- **[Testing the throughput of an inference endpoint](vllm-performance-endpoint.md)**
  — Find the maximum stable request rate for a running OpenAI API-compatible
  endpoint by using an optimiser to efficiently probe the request-rate
  dimension.
- **[Exploring vLLM deployment configurations](vllm-performance-full.md)** —
  Evaluate different vLLM server deployment configurations (GPU type, batch
  size, memory limits) on Kubernetes/OpenShift by combining the
  `test-deployment-v1` experiment with the `random_walk` operator.
- **[Benchmarking geospatial models with vLLM](vllm-performance-geospatial.md)**
  — Benchmark IBM-NASA Prithvi geospatial models for Earth observation tasks
  (flood detection, land-use classification) using the
  `test-geospatial-deployment-v1` experiment.
