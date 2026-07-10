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
  [add new experiments](developer-guide/creating-custom-experiments.md)
  or [optimizers and other tools](developer-guide/creating-operators.md)
- :gear: _Scalable_: We use [Ray](https://ray.io) as our execution engine,
  allowing experiments and tools to scale easily
- :recycle: _Automatic data-reuse_: Avoid repeating work with
  [transparent reuse of experiment results](concepts/data-sharing.md);
  `ado`'s internal protocols ensure this happens only when it makes sense
- :link: _Provenance_: Relationships between data and operations are
  [automatically tracked](cli-reference/index.md#ado-show-related).
  The versions of `ado-core` and every plugin used to create a resource are also
  recorded, keeping results reproducible and debuggable
- :mag: _Optimization and sampling_: Out-of-the-box, leverage powerful optimization
  methods [via Ray Tune](user-guide/operators/ray-tune.md)
  or use our [flexible built-in sampler](user-guide/operators/random-walk.md)
- :material-robot-outline: _Coding agents_: Supercharge your workflow. `ado`'s
  typed resources and bundled skills enable AI assistants to autonomously
  formulate, validate, and run experiments. [Learn more](user-guide/index.md)

### Foundation Model Experimentation

We have developed `ado` plugins providing advanced capabilities for performance
testing of foundation models:

- :stopwatch: [Fine-tuning performance benchmarking](user-guide/actuators/sft-trainer.md)
- :stopwatch:
  [Inference performance benchmarking](user-guide/examples/vllm-performance-endpoint.md)
  (using [vLLM bench](https://docs.vllm.ai/en/latest/cli/bench/serve.html) or
  [guidellm](https://github.com/vllm-project/guidellm))
- :crystal_ball: [Predictive performance model creation](user-guide/operators/trim.md)

## `ado` and Coding Agents

`ado` is designed from the ground up to partner with coding agents, creating a
powerful automated research assistant. This isn't just about agent skills; `ado`'s
core design allows an agent to reason about and execute a complete research
workflow.

<!-- markdownlint-disable line-length -->

| Step                    | Capability                                   | Benefit for Agent-Driven Research                                                                                                                                                                                          |
| :---------------------- | -------------------------------------------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **1. Discover**         | **Self-Describing Experiments and Operators**| Before acting, an agent must understand what's possible. `ado` enables agents to discover exactly what capabilities are available, and what's required to use them, in a structured manner rather than parsing code.       |
| **2. Model**            | **Clear Separation of Concerns**             | Once possibilities are known, the agent structures the problem. `ado` provides clear separation between the _what_, the _how_, and the _action_, allowing the agent to reason about each part of the problem independently.|
| **3. Act & Verify**     | **Pre-Run Validation**                       | With a model in mind, the agent can safely execute. Using `ado template` and `ado create --dry-run`, it can create a tight **generate → validate → fix → run** loop.                                                       |
| **4. Analyze & Refine** | **Structured & Queryable Data**              | All data and metadata created via `ado` is stored in a structured database. This allows the agent to analyze, compare, link, and synthesize data to decide on the next course of action.                                   |

<!-- markdownlint-enable line-length -->

Together, these properties enable a **closed research loop**: an agent can
describe a problem, run experiments, read the measurements, and refine its
approach, all while operating at a high level of abstraction — manipulating spaces
and operators rather than writing bespoke glue code.

See [Core Concepts](concepts/index.md) and [The ado CLI](cli-reference/index.md)
for more.

## Requirements

A basic installation of `ado` only requires a recent Python version (3.10 to
3.14). This will allow you to run [many of our examples](user-guide/examples/index.md)
and explore `ado` features.

### Additional Requirements

Some advanced features have additional requirements:

<!-- markdownlint-disable descriptive-link-text -->

- **Distributed Projects** **_(Optional)_**: To support projects with multiple
  users you will need a remote, accessible MySQL database. See
  [here](user-guide/backend-services.md#using-the-distributed-mysql-backend-for-ado)
  for more details
- **Multi-Node Execution** **_(Optional)_**: To support multi-node or scaled
  execution you may need a multi-node RayCluster. See
  [here](user-guide/backend-services.md#deploying-kuberay-and-creating-a-raycluster)
  for more details
<!-- markdownlint-enable descriptive-link-text -->

In addition, `ado` plugins may have additional requirements for executing
**_realistic_** experiments. For example:

- **_Fine-Tuning Benchmarking_**: Requires a
  [RayCluster with GPUs](user-guide/actuators/sft-trainer.md#configure-your-raycluster)
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

    Next, we recommend you try our short [tutorial](user-guide/examples/random-walk.md) which will give an idea of how `ado` works.

</div>
<!-- markdownlint-enable line-length -->

## Example

This video shows listing [actuators](user-guide/actuators/index.md) and
getting the details of an experiment. Check [demo](demo.md) for
more videos.

<!-- markdownlint-disable no-inline-html -->
<video controls preload="auto" poster="videos/step1_trimmed_thumbnail.png">
<source src="videos/step1_trimmed.mp4" type="video/mp4">
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

    [Taking a random walk :octicons-arrow-right-24:](user-guide/examples/random-walk.md)

- :octicons-terminal-24:{ .lg .middle } **Check out the ADO cli**

    ---

    Get familiar with the capabilities of the `ado` command-line interface.

    [Dive into the CLI reference docs :octicons-arrow-right-24:](cli-reference/index.md)

</div>
