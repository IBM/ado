# Introduction

This is the webpage for the **a**ccelerated **d**iscovery **o**rchestrator
(**`ado`**).

**`ado`** is a unified platform for **executing computational experiments at
scale** and **analysing their results**. It can be extended with new experiments
or new analysis tools. It allows distributed teams of researchers and engineers
to collaborate on projects, execute experiments, and share data.

You can run the experiments and analysis tools already available in **`ado`** in
a distributed, shared, environment with your team. You can also use **`ado`** to
get features like data-tracking, data-sharing, tool integration and a CLI, for
your analysis method or experiment for free.

🧑‍💻 Using **`ado`** assumes familiarity with command line tools.

🛠️ Developing **`ado`** requires knowledge of python.

## Key Features

- :computer: _CLI_: Our human-centric CLI follows
  [best practices](https://clig.dev)
- :handshake: _Projects_: Allow distributed groups of users to
  [collaborate and share data](resources/metastore.md)
- :electric_plug: _Extendable_: Easily
  [add new experiments](actuators/creating-custom-experiments.md),
  [optimizers or other tools.](operators/creating-operators.md)
- :gear: _Scalable_: We use [ray](https://ray.io) as our execution engine
  allowing experiments and tools to easily scale
- :recycle: _Automatic data-reuse_: Avoid repeating work with
  [transparent reuse of experiment results](core-concepts/data-sharing.md).
  `ado` internal protocols ensure this happens only when it makes sense
- :link: _Provenance_: As you work, the relationship between the data you create
  and operations you perform are
  [automatically tracked](getting-started/ado.md#ado-show-related)
- :mag: _Optimization and sampling_: Out-of-the-box, leverage powerful
  optimization methods [via `raytune`](operators/optimisation-with-ray-tune.md)
  or use our [flexible in built sampler](operators/random-walk.md)

### Foundation Model Experimentation

We have developed `ado` plugins providing advanced experiments for testing
foundation-models:

- :stopwatch: [fine-tuning performance benchmarking](actuators/sft-trainer.md)
- :stopwatch:
[inference performance benchmarking](examples/vllm-performance-endpoint.md)
(using the
  [vLLM performance benchmark](https://docs.vllm.ai/en/latest/cli/bench/serve.html))
- **COMING SOON** :crystal_ball: inference and fine-tuning prediction

## Requirements

A basic installation of `ado` only requires a recent Python version (3.10 to 3.13).
This will allow you to run [many of our examples](examples/examples.md) and
explore ado features.

### Additional Requirements

Some advanced features have additional requirements:

<!-- markdownlint-disable descriptive-link-text -->
- **Distributed Projects** **_(Optional)_**: To support projects with multiple
  users you will need a remote, accessible, MySQL database. See
  [here](getting-started/installing-backend-services.md#using-the-distributed-mysql-backend-for-ado)
  for more
- **Multi-Node Execution** **_(Optional)_**: To support multi-node or scaling
  execution you may need a multi-node RayCluster. See
  [here](getting-started/installing-backend-services.md#deploying-kuberay-and-creating-a-raycluster)
  for more details
<!-- markdownlint-enable descriptive-link-text -->

In addition `ado` plugins may have additional requirements for executing
**_realistic_** experiments. For example,

- **_Fine-Tuning Benchmarking_**: Requires a
  [RayCluster with GPUs](actuators/sft-trainer.md#configure-your-raycluster)
- **_vLLM Performance Benchmarking_**: Requires an OpenShift cluster with GPUs

## Try it out

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line no-inline-html -->
<div class="grid cards" markdown>

- :material-clock-fast:{ .lg .middle } **Set up in 1 minute**

    ---

    You can install **ado** by:

    ```shell
    pip install ado-core
    ```

    Now try:

    ```commandline
    ado get contexts
    ```

    You will see a **context**, `local`, is listed.

    A context is like a project.
    The `local` context links to a local database you can use as a sandbox for testing.

    Try:

    ```commandline
    ado get operators
    ```

    to see a list of the in-built operators.  

    Next, we recommend you try our short [tutorial](examples/random-walk.md) which will give an idea of how `ado` works.

</div>
<!-- markdownlint-enable line-length -->

## Example

This video shows listing [actuators](actuators/working-with-actuators.md) and
getting the details of an experiment. Check [demo](getting-started/demo.md) for
more videos.

<!-- markdownlint-disable no-inline-html -->
<video controls preload="auto" poster="getting-started/videos/step1_trimmed_thumbnail.png">
<source src="getting-started/videos/step1_trimmed.mp4" type="video/mp4">
</video>
<!-- markdownlint-enable no-inline-html -->

## Acknowledgement

This project is partially funded by the European Union through the Smart
Networks and Services Joint Undertaking (SNS JU) under grant agreement No.
101192750 (Project 6G-DALI).

## What's next

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line no-inline-html -->
<div class="grid cards" markdown>

- :octicons-rocket-24:{ .lg .middle } **Let's get started!**

    ---

    Jump into our tutorial

    [Taking a random walk :octicons-arrow-right-24:](examples/random-walk.md)

- :octicons-terminal-24:{ .lg .middle } **Check out the ADO cli**

    ---

    Get familiar with the capabilities of the `ado` command-line interface.

    [Dive into the CLI reference docs :octicons-arrow-right-24:](getting-started/ado.md)

</div>
