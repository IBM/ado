# ado — accelerated discovery orchestrator

![PyPI Version](https://img.shields.io/pypi/v/ado-core)
![PyPI Python Version](https://img.shields.io/pypi/pyversions/ado-core)
![GitHub License](https://img.shields.io/github/license/ibm/ado)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.10304/status.svg)](https://doi.org/10.21105/joss.10304)

**`ado`** is a Python platform for **designing experiment campaigns and
executing them at scale**. It enables distributed teams of researchers and
engineers to collaborate, execute experiments, and share data.

## At its _core_

ado is built around four key concepts:

| Concept             | Role                                                                                                                     |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Discovery Space** | Defines _what_ to measure (Entity Space), _how_ to measure it (Experiments), and _where_ to store results.               |
| **Experiments**     | Pluggable measurement functions — take entity properties as input, produce new properties as output.                     |
| **Operation**       | Defines _which_ operator to use (e.g. Ray Tune) and _how_ to parameterise it to drive experiments over the entity space. |
| **Sample Store**    | Stores measurements and transparently reuses prior results across Discovery Spaces and team members.                     |

Learn more about these concepts at <https://ibm.github.io/ado/concepts/>.

## Key Features

- 🔌 _Extensible_: quickly add
  [new experiments](https://ibm.github.io/ado/actuators/creating-custom-experiments/)
  or [operators](https://ibm.github.io/ado/operators/creating-operators/), often
  as simply as decorating a Python function
- ⚙️ _Scalable execution_: automatically leverage [Ray](https://www.ray.io/) for
  parallel and multi-node experiment runs out of the box
- 🔎 _Optimization & sampling_: run optimizations with our
  [Ray Tune operator](https://ibm.github.io/ado/operators/optimisation-with-ray-tune/)
  or a
  [flexible random-walk sampler](https://ibm.github.io/ado/operators/random-walk/)
- ♻️ _Automatic data reuse_: reuse existing results transparently with our
  [memoization features](https://ibm.github.io/ado/core-concepts/data-sharing/)
- 🔗 _Full provenance_: results and resources are annotated with relationships
  and the plugin versions used to produce them
- 🤝 _Collaborative projects_: distributed teams can
  [share a common data store and results](https://ibm.github.io/ado/resources/metastore/)
- 💻 _Human-centric CLI_: intuitively inspect, create, and manage resources from
  the terminal
- 🤖 _AI-agent ready_: typed resources and bundled skills let coding agents
  [autonomously formulate and run experiments](https://ibm.github.io/ado/how-to/)

## Quick Start

Install `ado` and the example actuator plugin (a virtual environment is
recommended). For complete instructions see the
[install guide](https://ibm.github.io/ado/getting-started/install/):

```shell
pip install ado-core
pip install git+https://github.com/IBM/ado.git#subdirectory=plugins/actuators/example_actuator
```

Download the example Discovery Space and Operation YAML files:

```shell
curl -O https://raw.githubusercontent.com/IBM/ado/refs/heads/main/plugins/actuators/example_actuator/yamls/discoveryspace.yaml
curl -O https://raw.githubusercontent.com/IBM/ado/refs/heads/main/plugins/actuators/example_actuator/yamls/random_walk_operation.yaml
```

Create a Discovery Space and run a random-walk operation over it — `ado`
resolves the space reference automatically:

```shell
ado create operation -f random_walk_operation.yaml --with space=discoveryspace.yaml
```

`ado` will create the Discovery Space, sample entities from it using the
built-in random-walk operator, execute the `peptide_mineralization` experiment
for each entity, and store the results locally.

Once the operation finishes, inspect the collected measurements:

```shell
ado show measurements space --use-latest
```

For a deeper walkthrough, see the
[random-walk tutorial](https://ibm.github.io/ado/examples/random-walk/).

## See What We've Built

Here are some examples of what the team has built with `ado`:

- 🧠
  [Fine-tuning performance benchmarking](https://ibm.github.io/ado/actuators/sft-trainer/)
- 📈
  [Inference performance benchmarking](https://ibm.github.io/ado/examples/vllm-performance-endpoint/)
  (using [vLLM bench](https://docs.vllm.ai/en/stable/cli/bench/serve/) or
  [guidellm](https://github.com/vllm-project/guidellm))
- 🔮
  [Predictive performance model creation](https://ibm.github.io/ado/operators/trim/)

## Contributing

To set up a development environment, run the test suite, or understand code
style and commit conventions, see [DEVELOPING.md](DEVELOPING.md) and
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
