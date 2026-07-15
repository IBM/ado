# ado — accelerated discovery orchestrator

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

ado is built around three key concepts that power these features (explore them
all at <https://ibm.github.io/ado/concepts/>):

| Concept             | Role                                                                                                                 |
| ------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Discovery Space** | Defines _what_ to measure (Entity Space), _how_ to measure it (Experiments), and _where_ to store results.           |
| **Operation**       | Defines _which_ operator to use (e.g. Ray Tune) and _how_ to parameterise it to explore or analyse the entity space. |
| **Sample Store**    | Stores measurements and allows transparently reusing prior results across Discovery Spaces and team members.         |

## Try It Out

The following example runs a small experiment campaign that samples combinations
of `mass` and `volume`, computes `density` at each point, and stores the
results.

Install `ado-core` (a virtual environment is recommended). For complete
instructions see the
[install guide](https://ibm.github.io/ado/user-guide/install/):

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
[density example tutorial](https://ibm.github.io/ado/user-guide/examples/density-example/).

## ado ❤️ agents

ado's typed resources, expressive CLI, and bundled agent skills make it a
natural fit for agentic research workflows. Once prompted with a research
problem, an agent can design the Discovery Space, write new experiments or reuse
existing ones, and run the full exploration loop:

- 🤖 _Bundled agent skills_: ready-made skills guide agents through
  [end-to-end discovery workflows](https://ibm.github.io/ado/user-guide/getting-started/#__tabbed_1_2)
  — from formulating a problem to analysing results
- 🔍 _Self-describing resources_: experiments and operators declare their
  required properties, so an agent can discover what's available and what's
  needed without parsing code
- 🧱 _Validated schemas_: research intent is expressed as structured, validated
  configurations — constraining the agent to well-defined inputs rather than
  free-form code generation, reducing hallucinations and keeping experiments
  repeatable
- ✅ _Safe execution loop_: `ado template` and `--dry-run` support a tight
  **generate → validate → fix → run** cycle before any work is committed
- 📦 _Structured & queryable results_: all measurements and metadata are stored
  in a structured database, giving agents clean access to data for analysis and
  refinement
- 🔗 _Full provenance_: every result is annotated with resource relationships
  and plugin versions, so an agent always knows where data came from and how to
  reproduce it

## Use Cases

Here are some examples of what the team has built with `ado`:

- 🧠
  [Fine-tuning performance benchmarking](https://ibm.github.io/ado/user-guide/examples/finetune-remotely/)
- 📈
  [Inference performance benchmarking](https://ibm.github.io/ado/examples/vllm-performance-endpoint/)
  (using [vLLM bench](https://docs.vllm.ai/en/stable/cli/bench/serve/) or
  [guidellm](https://github.com/vllm-project/guidellm))
- 🔮
  [Predictive performance model creation](https://ibm.github.io/ado/user-guide/examples/trim/)

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
