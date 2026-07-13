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

ado is built around four key concepts (explore them all in the
[concepts](concepts/index.md) section):

| Concept             | Role                                                                                                                     |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Discovery Space** | Defines _what_ to measure (Entity Space), _how_ to measure it (Experiments), and _where_ to store results.               |
| **Experiments**     | Pluggable measurement functions — take entity properties as input, produce new properties as output.                     |
| **Operation**       | Defines _which_ operator to use (e.g. Ray Tune) and _how_ to parameterise it to drive experiments over the entity space. |
| **Sample Store**    | Stores measurements and transparently reuses prior results across Discovery Spaces and team members.                     |

## Quick Start

Install `ado-core` (a virtual environment is recommended). For complete
instructions see [Getting Started](user-guide/index.md#choose-your-workflow):

```shell
pip install ado-core
```

Clone the repository and install the density example package:

```shell
git clone https://github.com/IBM/ado.git
cd ado
pip install -e examples/density_example/
```

Run an operation over a density discovery space — `ado` resolves the space
reference automatically via `--with`:

```shell
ado create operation -f examples/density_example/operation.yaml --with space=examples/density_example/space.yaml
```

Once the operation finishes, inspect the collected measurements:

```shell
ado show measurements operation --use-latest
```

For a deeper walkthrough, see the
[density example tutorial](user-guide/examples/density-example.md).

## ado ❤️ agents

`ado` is designed from the ground up to partner with coding agents, creating a
powerful automated research assistant. This isn't just about agent skills;
`ado`'s core design allows an agent to reason about and execute a complete
research workflow.

<!-- markdownlint-disable line-length -->

| Step                    | Capability                                    | Benefit for Agent-Driven Research                                                                                                                                                                                           |
| :---------------------- | --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Discover**         | **Self-Describing Experiments and Operators** | Before acting, an agent must understand what's possible. `ado` enables agents to discover exactly what capabilities are available, and what's required to use them, in a structured manner rather than parsing code.        |
| **2. Model**            | **Clear Separation of Concerns**              | Once possibilities are known, the agent structures the problem. `ado` provides clear separation between the _what_, the _how_, and the _action_, allowing the agent to reason about each part of the problem independently. |
| **3. Act & Verify**     | **Pre-Run Validation**                        | With a model in mind, the agent can safely execute. Using `ado template` and `ado create --dry-run`, it can create a tight **generate → validate → fix → run** loop.                                                        |
| **4. Analyze & Refine** | **Structured & Queryable Data**               | All data and metadata created via `ado` is stored in a structured database. This allows the agent to analyze, compare, link, and synthesize data to decide on the next course of action.                                    |

<!-- markdownlint-enable line-length -->

Together, these properties enable a **closed research loop**: an agent can
describe a problem, run experiments, read the measurements, and refine its
approach — all while operating at a high level of abstraction, manipulating
spaces and operators rather than writing bespoke glue code.

See the [concepts](concepts/index.md) section and the ado
[CLI reference](cli-reference/index.md) for more.

## Use Cases

Here are some examples of what the team has built with `ado`:

- 🧠 [Fine-tuning performance benchmarking](user-guide/actuators/sft-trainer.md)
- 📈
  [Inference performance benchmarking](user-guide/examples/vllm-performance-endpoint.md)
  (using [vLLM bench](https://docs.vllm.ai/en/stable/cli/bench/serve/) or
  [guidellm](https://github.com/vllm-project/guidellm))
- 🔮 [Predictive performance model creation](user-guide/operators/trim.md)

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

    [Beyond the basics: `ado` with real data :octicons-arrow-right-24:](user-guide/examples/random-walk.md)

- :octicons-terminal-24:{ .lg .middle } **Check out the ADO cli**

    ---

    Get familiar with the capabilities of the `ado` command-line interface.

    [Dive into the CLI reference docs :octicons-arrow-right-24:](cli-reference/index.md)

</div>
