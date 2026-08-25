# ado — accelerated discovery orchestrator

![PyPI Version](https://img.shields.io/pypi/v/ado-core)
![PyPI Python Version](https://img.shields.io/pypi/pyversions/ado-core)
![GitHub License](https://img.shields.io/github/license/ibm/ado)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.10304/status.svg)](https://doi.org/10.21105/joss.10304)

**`ado`** is a Python platform for **designing computational experiment
campaigns and executing them at scale**. It enables
AI Coding Agents to **autonomously formulate, run and analyze your
experiments**.

## Why ado?

* It provides coding agents with **tools for treating primary research
as a coding problem**, increasing their ability to quickly and reliably execute
the research loop.
* It's **plugin model**  enables you to easily
take advantage of this capability for your research domain.
* Its **SQL-backed storage** automatically persists your
experiment measurements and campaign metadata, and allows scaling from personal
use to team-wide collaboration.

> 💡 Like our approach to agent-driven science? Drop a ⭐ to support our
> open-source development!

## At its _core_

**ado** is built on three concepts:

| Concept             | Role                                                                                                                                                                                        |
| ------------------- |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Discovery Space** | Defines _what_ to measure, _how_ to measure it (via Experiments, which are pluggable python functions), and _where_ to store results.                                                       |
| **Operation**       | You explore or analyse a Discovery Space using operations. You can select from different operators to perform different types of operations. Operators are also pluggable python functions. |
| **Sample Store**    | Stores the results of measurements, and enables operations to transparently reuse existing results (memoization).                                                                           |

## Decorate a function, get a full-featured CLI

You can add experiments to `ado` using a decorated Python function:

```python
from typing import Any

from ado.modules.actuators.custom_experiments import custom_experiment


@custom_experiment(output_property_identifiers=["density"])
def calculate_density(mass: float, volume: float) -> dict[str, Any]:
    density_value = mass / volume if volume else None
    return {"density": density_value}
```

With this `ado` can understand the experiment, create valid
spaces of inputs for it, explore those spaces with operations
, and store the results in a samplestore,
all while keeping a record of your work:

![Terminal recording of ado listing the installed experiments, describing
calculate_density, viewing the discovery space definition, running an operation
across it and printing the resulting
measurements](docs/videos/readme_try_it_out.gif)

## Try It Out

The following toy example runs a small experiment campaign
that samples combinations of mass and volume, computes density at each point,
and stores the results.

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
ado create operation -f examples/density_example/operation.yaml --with space=examples/density_example/space.yaml
```

Once the operation finishes, inspect the collected measurements:

```shell
ado show measurements operation --use-latest
```

For a deeper walkthrough, see the
[density example tutorial](https://ibm.github.io/ado/user-guide/examples/tutorials/density-example/).

## ado 🤝 agents

* 🧱 _Validated schemas_: research intent is expressed as structured, validated
  configurations — constraining the agent to well-defined inputs rather than
  free-form code generation, reducing hallucinations and keeping experiments
  repeatable
* ✅ _Safe execution loop_: `ado template` and `--dry-run` support a tight
  **generate → validate → fix → run** cycle before any work is committed
* 🔍 _Self-describing resources_: experiments and operators declare their
  required properties, so an agent can discover what's available and what's
  needed without parsing code
* 📦 _Structured & queryable results_: all measurements and metadata are stored
  in a structured database, giving agents clean access to data for analysis and
  refinement
* 🔗 _Full provenance_: every result is annotated with resource relationships
  and plugin versions, so an agent always knows where data came from and how to
  reproduce it
* 🤖 _Bundled agent skills_: skills guide agents through
  [end-to-end discovery workflows](https://ibm.github.io/ado/latest/user-guide/ado-and-agents/#what-you-can-ask-your-agent-to-do)
  — from formulating a problem to analysing results

## Use Cases

Here are some examples of what the team has built with `ado`:

* 🧠
  [Fine-tuning performance benchmarking](https://ibm.github.io/ado/latest/user-guide/examples/finetune-remotely/)
* 📈
  [Inference performance benchmarking](https://ibm.github.io/ado/latest/user-guide/examples/vllm-performance-endpoint/)
  (using [vLLM bench](https://docs.vllm.ai/en/stable/cli/bench/serve/) or
  [guidellm](https://github.com/vllm-project/guidellm))
* 🔮
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
