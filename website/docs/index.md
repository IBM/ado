# Introduction

![PyPI Version](https://img.shields.io/pypi/v/ado-core)
![PyPI Python Version](https://img.shields.io/pypi/pyversions/ado-core)
![GitHub License](https://img.shields.io/github/license/ibm/ado)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.10304/status.svg)](https://doi.org/10.21105/joss.10304)

**`ado`** is a Python platform for **designing experiment campaigns and
executing them at scale**. It enables distributed teams of researchers and
engineers to collaborate, execute experiments, and share data.

You can extend ado across different domains through its **plugin model** — often
as simple as decorating a Python function. By integrating your methodology, you
gain cross-cutting capabilities — such as **parallel execution**, **data
provenance**, and a **unified CLI** — alongside a structured foundation that
allows AI coding agents to **autonomously formulate and run your experiments**.

- 🧑‍💻 **Using `ado`** assumes familiarity with command line tools.
- 🛠️ **Developing `ado`** requires knowledge of Python.

## Key Features

- :computer: _CLI_: Our human-centric CLI follows [best practices](https://clig.dev)
- :handshake: _Projects_: Allow distributed groups of users to
  [collaborate and share data](resources/metastore.md)
- 🔌 _Extendable_: Easily
  [add new experiments](actuators/creating-custom-experiments.md)
  or [optimizers and other tools](operators/creating-operators.md)
- :gear: _Scalable_: We use [Ray](https://ray.io) as our execution engine,
  allowing experiments and tools to scale easily
- :recycle: _Automatic data-reuse_: Avoid repeating work with
  [transparent reuse of experiment results](core-concepts/data-sharing.md);
  `ado`'s internal protocols ensure this happens only when it makes sense
- :link: _Provenance_: Relationships between data and operations are
  [automatically tracked](getting-started/ado.md#ado-show-related).
  The versions of `ado-core` and every plugin used to create a resource are also
  recorded, keeping results reproducible and debuggable
- :mag: _Optimization and sampling_: Out-of-the-box, leverage powerful optimization
  methods [via Ray Tune](operators/optimisation-with-ray-tune.md)
  or use our [flexible built-in sampler](operators/random-walk.md)
- :material-robot-outline: _Coding agents_: Supercharge your workflow. `ado`'s
  typed resources and bundled skills enable AI assistants to autonomously
  formulate, validate, and run experiments. [Learn more](how-to/index.md)

### Foundation Model Experimentation

We have developed `ado` plugins providing advanced capabilities for performance
testing of foundation models:

- :stopwatch: [Fine-tuning performance benchmarking](actuators/sft-trainer.md)
- :stopwatch:
  [Inference performance benchmarking](examples/vllm-performance-endpoint.md)
  (using [vLLM bench](https://docs.vllm.ai/en/latest/cli/bench/serve.html) or
  [guidellm](https://github.com/vllm-project/guidellm))
- :crystal_ball: [Predictive performance model creation](operators/trim.md)

## Requirements

A basic installation of `ado` only requires a recent Python version (3.10 to
3.14). This will allow you to run [many of our examples](examples/examples.md)
and explore `ado` features.

### Additional Requirements

Some advanced features have additional requirements:

<!-- markdownlint-disable descriptive-link-text -->

- **Distributed Projects** **_(Optional)_**: To support projects with multiple
  users you will need a remote, accessible MySQL database. See
  [here](getting-started/installing-backend-services.md#using-the-distributed-mysql-backend-for-ado)
  for more details
- **Multi-Node Execution** **_(Optional)_**: To support multi-node or scaled
  execution you may need a multi-node RayCluster. See
  [here](getting-started/installing-backend-services.md#deploying-kuberay-and-creating-a-raycluster)
  for more details
<!-- markdownlint-enable descriptive-link-text -->

In addition, `ado` plugins may have additional requirements for executing
**_realistic_** experiments. For example:

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
