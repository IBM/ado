# Introduction

![PyPI Version](https://img.shields.io/pypi/v/ado-core)
![PyPI Python Version](https://img.shields.io/pypi/pyversions/ado-core)
![GitHub License](https://img.shields.io/github/license/ibm/ado)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.10304/status.svg)](https://doi.org/10.21105/joss.10304)

**`ado`** is a Python platform for **designing computational experiment
campaigns and executing them at scale**. It enables distributed teams of
researchers and engineers to collaborate, execute experiments, and share data.

You can extend `ado` across different domains through its **plugin model** —
often as simple as decorating a Python function. By integrating your
methodology, you gain cross-cutting capabilities — such as **parallel
execution**, **data provenance**, and a **unified CLI** — alongside a structured
foundation that allows AI coding agents to **autonomously formulate and run your
experiments**.

## At its _core_

ado is built around four key concepts that power these features (explore them
all in the [concepts](concepts/index.md) section):

| Concept             | Role                                                                                                                 |
| ------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Discovery Space** | Defines _what_ to measure (Entity Space), _how_ to measure it (Experiments), and _where_ to store results.           |
| **Experiments**     | Pluggable measurement functions — take entity properties as input, produce new properties as output.                 |
| **Operation**       | Defines _which_ operator to use (e.g. Ray Tune) and _how_ to parameterise it to explore or analyse the entity space. |
| **Sample Store**    | Stores measurements and transparently reuses prior results across Discovery Spaces and team members.                 |

## ado ❤️ agents

ado's typed resources, expressive CLI, and bundled agent skills make it a
natural fit for agentic research workflows. Once prompted with a research
problem, an agent can design the Discovery Space, write new experiments or reuse
existing ones, and run the full exploration loop. See
[Getting Started with an agent](user-guide/getting-started.md#how-do-you-want-to-use-ado).

## Try It Out

The following example runs a small experiment campaign that samples combinations
of `mass` and `volume`, computes `density` at each point, and stores the
results.

Install `ado-core` (a virtual environment is recommended). For complete
instructions see [Getting Started](user-guide/getting-started.md#installing):

```shell
pip install ado-core
```

Clone the repository and install the density example package:

```shell
git clone https://github.com/IBM/ado.git
cd ado
pip install -e examples/density_example/
```

Run the experiment campaign:

```shell
ado create operation -f examples/density_example/operation.yaml --with space=examples/density_example/space.yaml
```

Once the operation finishes, inspect the collected measurements:

```shell
ado show measurements operation --use-latest
```

For a deeper walkthrough, see the
[density example tutorial](user-guide/examples/tutorials/density-example.md).

## Use Cases

Here are some examples of what the team has built with `ado`:

- 🧠
  [Fine-tuning performance benchmarking](user-guide/examples/finetune-remotely.md)
- 📈
  [Inference performance benchmarking](user-guide/examples/vllm-performance-endpoint.md)
  (using [vLLM bench](https://docs.vllm.ai/en/stable/cli/bench/serve/) or
  [guidellm](https://github.com/vllm-project/guidellm))
- 🔮 [Predictive performance model creation](user-guide/examples/trim.md)

## Acknowledgement

This project is partially funded by the European Union through the Smart
Networks and Services Joint Undertaking (SNS JU) under grant agreement No.
101192750 (Project 6G-DALI).

## What's next

<!-- prettier-ignore-start -->
<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line no-inline-html -->
<div class="grid cards" markdown>

- :octicons-rocket-24:{ .lg .middle } **Let's get started!**

    ---

    Jump into our tutorial

    [The basics: your first `ado` experiment :octicons-arrow-right-24:](user-guide/examples/tutorials/density-example.md)

- :octicons-terminal-24:{ .lg .middle } **Check out the ADO cli**

    ---

    Get familiar with the capabilities of the `ado` command-line interface.

    [Dive into the CLI reference docs :octicons-arrow-right-24:](cli-reference/index.md)

</div>

<!-- prettier-ignore-end -->
