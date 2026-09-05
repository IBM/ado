# ado — accelerated discovery orchestrator

![PyPI Version](https://img.shields.io/pypi/v/ado-core)
![PyPI Python Version](https://img.shields.io/pypi/pyversions/ado-core)
![GitHub License](https://img.shields.io/github/license/ibm/ado)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.10304/status.svg)](https://doi.org/10.21105/joss.10304)
[![Give a Star!](https://img.shields.io/badge/⭐-Give%20a%20Star!-yellow)](https://github.com/ibm/ado)

**`ado`** provides tools for **designing and executing computational experiment
campaigns**. AI Coding Agents can use `ado` to **autonomously formulate, run and
analyze your experiments**.

## Why ado?

- **Defines Campaigns as Verifiable Code:** Provides rich, strictly typed,
  objects for defining experiment campaigns, giving flexibility in design while
  guaranteeing they can be executed.
- **Simplifies Execution** : Seamlessly executes campaigns on remote clusters,
  hiding complex setup and handling the distributed plumbing.
- **Durable Structured Memory:** Automatically captures designs, executions, and
  measurements in local or shared SQL databases
- **Adapt to Any Domain:** Its flexible plugin model allows extending these core
  capabilities to your specific research field.
- **Empowers Agent-Driven Research**: The verifiable objects, executable APIs,
  and durable memory enable coding agents to treat experimentation as a coding
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

You can create your own Experiments to use in a Discovery Space, and your own
Operators to explore and analyse it. You can also leverage Experiments and
Operators others have created.

## Try It Out

The following toy example runs a small experiment campaign that samples
combinations of mass and volume, computes density at each point, and stores the
results. It uses an experiment added to ado by decorating a Python function:

```python
from typing import Any

from ado.modules.actuators.custom_experiments import custom_experiment


@custom_experiment(output_property_identifiers=["density"])
def calculate_density(mass: float, volume: float) -> dict[str, Any]:
    density_value = mass / volume if volume else None
    return {"density": density_value}
```

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
measurements](docs/videos/readme_try_it_out.gif)

For a deeper walkthrough, see the
[density example tutorial](https://ibm.github.io/ado/latest/user-guide/examples/tutorials/density-example/).

## ado 🤝 agents

- 🧱 _Validated schemas_: research intent is expressed as structured, validated
  configurations — constraining the agent to well-defined inputs rather than
  free-form code generation, reducing hallucinations and keeping experiments
  repeatable
- ✅ _Safe execution loop_: `ado template` and `--dry-run` support a tight
  **generate → validate → fix → run** cycle before any work is committed
- 🔍 _Self-describing resources_: experiments and operators declare their
  required properties, so an agent can discover what's available and what's
  needed without parsing code
- 📦 _Structured & queryable results_: all measurements and metadata are stored
  in a structured database, giving agents clean access to data for analysis and
  refinement
- 🔗 _Full provenance_: every result is annotated with resource relationships
  and plugin versions, so an agent always knows where data came from and how to
  reproduce it
- 🤖 _Bundled agent skills_: skills guide agents through
  [end-to-end discovery workflows](https://ibm.github.io/ado/latest/user-guide/ado-and-agents/#what-you-can-ask-your-agent-to-do)
  — from formulating a problem to analysing results

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
