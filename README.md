# ado — accelerated discovery orchestrator

![PyPI Version](https://img.shields.io/pypi/v/ado-core)
![PyPI Python Version](https://img.shields.io/pypi/pyversions/ado-core)
![GitHub License](https://img.shields.io/github/license/ibm/ado)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.10304/status.svg)](https://doi.org/10.21105/joss.10304)
[![Give a Star!](https://img.shields.io/badge/⭐-Give%20a%20Star!-yellow)](https://github.com/ibm/ado)

**`ado`** provides tools for **executing computational experiment
campaigns**. Coding Agents can use `ado` to **autonomously formulate, run, and
analyze your experiments**.

## Why ado?

- **Declarative Experiment Campaigns:** Provides flexible,
  cross-domain schemas for defining experiment campaigns. A valid campaign
  definition has strong execution guarantees.
- **Simplifies Execution** : Handles the complex setup and
  distributed plumbing of scale-out campaign execution
- **Persistent Storage:** Automatically captures campaign definitions and
  measurements in local or shared SQL databases
- **Adapt to Any Domain:** Its flexible plugin model allows extending these core
  capabilities to your specific research field.
- **Empowers Agent-Driven Research**: The verifiable schemas, executable APIs,
  and persistent storage enable coding agents to treat experimentation as a coding
  problem.

## At its _core_

**ado** is built on three concepts:

| Concept             | Role                                                                                                                                         |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **Discovery Space** | Defines _what_ to measure (Entities), _how_ to measure them (Experiments) and _where_ to store results.                                      |
| **Operation**       | You explore or analyse a Discovery Space using Operations. You can select from different Operators to perform different types of Operations. |
| **Sample Store**    | Stores the results of measurements, and enables Operations to transparently reuse existing results (memoization).                            |

In **ado** the research loop involves defining a Discovery Space, exploring it
with an Operation, analyzing the results with additional Operations, and
repeating.

You can create your own Experiments to use in a Discovery Space. `ado`
provides many advanced exploration and analysis Operators you can
use, or you can define your own.

## Try It Out

The following toy example runs a small experiment campaign that samples
combinations of mass and volume, computes density at each point, and stores the
results.

Install `ado-core` (a virtual environment is recommended). For complete
instructions see the
[install guide](https://ibm.github.io/ado/latest/user-guide/getting-started/#installing):

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
ado create space -f examples/density_example/space.yaml
ado create operation -f examples/density_example/operation.yaml --use-latest space
```

Once the operation finishes, inspect the collected measurements:

```shell
ado show measurements operation --use-latest
```

![Terminal recording of ado listing the installed experiments, describing
calculate_density, viewing the discovery space definition, running an operation
across it and printing the resulting
measurements](https://github.com/user-attachments/assets/876ee1fc-1b59-45cc-ab23-9042ea2b6f8f)

For a deeper walkthrough, see the
[density example tutorial](https://ibm.github.io/ado/latest/user-guide/examples/tutorials/density-example/).

## ado 🤝 agents

- 🔍 _Self-describing resources_: experiments and operators declare their
  required properties, so an agent can discover what's available and what's
  needed without parsing code.
- ✅ _Campaigns as Verifiable Code_: `ado`'s structured, verifiable schemas
  for expressing research intent enable a tight
  **generate → validate → fix → run** cycle. This reduces hallucinations and the
  need for free-form code.
- 📦 _Durable Long-Term Memory_: `ado`s structured databases enable agents
  to understand what has been done, and to access the relevant data, beyond
  their context window.
- 🔗 _Full provenance_: every result is annotated with resource relationships
  and plugin versions, so an agent always knows where data came from and how to
  reproduce it.
- 🤖 _Bundled agent skills_: skills guide agents through
  [end-to-end discovery workflows](https://ibm.github.io/ado/latest/user-guide/ado-and-agents/#what-you-can-ask-your-agent-to-do)
  — from formulating a problem to analysing results.

## Use Cases

Here are some examples of what the team has built with `ado`:

- 🧠
  [Fine-tuning performance benchmarking](https://ibm.github.io/ado/latest/user-guide/examples/finetune-remotely/)
- 📈
  [Inference performance benchmarking](https://ibm.github.io/ado/latest/user-guide/examples/vllm-performance-endpoint/)
  (using [vLLM bench](https://docs.vllm.ai/en/stable/cli/bench/serve/) or
  [guidellm](https://github.com/vllm-project/guidellm))
- 🔮
  [Predictive performance model creation](https://ibm.github.io/ado/latest/user-guide/examples/trim/)

## Contributing

Contributions are welcome — new actuators, operators, bug fixes, and
documentation improvements. To set up a development environment, run the test
suite, or understand code style and commit conventions, see
[CONTRIBUTING.md](CONTRIBUTING.md), [DEVELOPING.md](DEVELOPING.md) and
[tests/README.md](tests/README.md).

## Citation

For an overview of the design and architecture of `ado`, see
[our Journal of Open Source Software paper](https://doi.org/10.21105/joss.10304).

If `ado` has been useful in your research, please cite us using:

```bibtex
@article{Johnston_ado_a_Python_2026,
author = {Johnston, Michael A. and Pomponio, Alessandro},
doi = {10.21105/joss.10304},
journal = {Journal of Open Source Software},
month = may,
number = {121},
pages = {10304},
title = {{ado: a Python framework for computational experimentation and benchmarking}},
url = {https://joss.theoj.org/papers/10.21105/joss.10304},
volume = {11},
year = {2026}
}
```

You can also click **"Cite this repository"** in the GitHub sidebar for
alternative formats such as APA.

## Acknowledgement

This project is partially funded by the European Union through the Smart
Networks and Services Joint Undertaking (SNS JU) under grant agreement No.
101192750 (Project 6G-DALI).
