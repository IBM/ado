<!-- markdownlint-disable-next-line first-line-h1 -->
## Tutorial

Our short tutorial, [Taking a random walk](random-walk.md), introduces core
`ado` concepts and is the recommended place to start.

## General Examples

The following examples illustrate general features of `ado`. They build on the
concepts learned in the tutorial and leverage pre-existing data and/or toy
measurements allowing them to run quickly.

- [Search a space with an optimizer](best-configuration-search.md)
- [Search a space based on a custom objective function](search-custom-objective.md)
- [Identify the important dimensions of a space](lhu.md)

After following these examples you can also try applying capabilities learned in
one example to another.

## Foundation Models Characterization

The following examples illustrate using the
[vllm_performance](../actuators/vllm_performance.md) and
[SFTTrainer](../actuators/sft-trainer.md)
actuators which offer benchmarking experiments for foundation model inference
and fine-tuning respectively.

- [Measure throughput of fine-tuning locally](finetune-locally.md)
- [Measure throughput of fine-tuning on a RayCluster with GPUs](finetune-remotely.md)
- [Find the request rate giving the highest stable throughput for an inference server](vllm-performance-endpoint.md)
- [Evaluate different vLLM server deployment configurations on Kubernetes/OpenShift](vllm-performance-full.md)

## Adding experiments or analysis tools to `ado`

The
[search a space based on a custom objective function](search-custom-objective.md)
example, combines with the
[creating a custom experiment](../actuators/creating-custom-experiments.md)
documentation to illustrate a simple method for adding your own experiments to
`ado`.

For adding actuators, we provide an
[example template actuator repository](https://github.com/IBM/ado/tree/main/plugins/actuators/example_actuator)
which can be used with our
[documentation on writing actuators](../actuators/creating-actuator-classes.md).

For adding operators, we have an
[example template operator repository](https://github.com/IBM/ado/tree/main/plugins/actuators/example_actuator)
which can be used with our
[documentation on writing operators](../operators/creating-operators.md).

## What's next

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line no-inline-html -->
<div class="grid cards" markdown>

- :octicons-workflow-24:{ .lg .middle } __Learn about Core Concepts__

      ---

      Find out more about the [core concepts](../core-concepts/concepts.md) underpinning ado.

      [Core concepts :octicons-arrow-right-24:](../core-concepts/concepts.md)

- :octicons-rocket-24:{ .lg .middle } __Extend ado with new Actuators__

    ---

    Learn about how ado can be extended with custom [Actuators](../actuators/working-with-actuators.md) that provide ability to run experiments in new domains.

    [Creating new Actuators :octicons-arrow-right-24:](../actuators/working-with-actuators.md)

</div>
<!-- markdownlint-enable line-length -->