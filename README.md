# ado — accelerated discovery orchestrator

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

- 💻 _CLI_: Our human-centric CLI follows [best practices](https://clig.dev)
- 🤝 _Projects_: Allow distributed groups of users to
  [collaborate and share data](https://ibm.github.io/ado/resources/metastore/)
- 🔌 _Extendable_: Easily
  [add new experiments](https://ibm.github.io/ado/actuators/creating-custom-experiments/)
  or
  [optimizers and other tools](https://ibm.github.io/ado/operators/creating-operators/)
- ⚙️ _Scalable_: We use [Ray](https://www.ray.io/) as our execution engine,
  allowing experiments and tools to scale easily
- ♻️ _Automatic data-reuse_: Avoid repeating work with
  [transparent reuse of experiment results](https://ibm.github.io/ado/core-concepts/data-sharing/);
  `ado`'s internal protocols ensure this happens only when it makes sense
- 🔗 _Provenance_: Relationships between data and operations are
  [automatically tracked](https://ibm.github.io/ado/getting-started/ado/#ado-show-related).
  The versions of `ado-core` and every plugin used to create a resource are also
  recorded, keeping results reproducible and debuggable
- 🔎 _Optimization and sampling_: Out-of-the-box, leverage powerful optimization
  methods
  [via Ray Tune](https://ibm.github.io/ado/operators/optimisation-with-ray-tune/)
  or use our
  [flexible built-in sampler](https://ibm.github.io/ado/operators/random-walk/)
- 🤖 _Coding agents_: Supercharge your workflow. `ado`'s typed resources and
  bundled skills enable AI assistants to autonomously formulate, validate, and
  run experiments. [Learn more](https://ibm.github.io/ado/how-to/)

### Foundation Model Experimentation

We have developed `ado` plugins providing advanced capabilities for performance
testing of foundation models:

- ⏱️
  [Fine-tuning performance benchmarking](https://ibm.github.io/ado/actuators/sft-trainer)
- ⏱️
  [Inference performance benchmarking](https://ibm.github.io/ado/examples/vllm-performance-endpoint/)
  (using [vLLM bench](https://docs.vllm.ai/en/stable/cli/bench/serve/) or
  [guidellm](https://github.com/vllm-project/guidellm))
- 🔮
  [Predictive performance model creation](https://ibm.github.io/ado/operators/trim/)

## Quick Start

Install `ado` from PyPI (a virtual environment is recommended). For complete
instructions see the
[install guide](https://ibm.github.io/ado/getting-started/install/):

```shell
pip install ado-core
```

Then try:

```shell
ado get contexts
```

You will see a context named `local` — a local sandbox you can use straight
away. To see the built-in operators:

```shell
ado get operators
```

Next, follow the short
[random-walk tutorial](https://ibm.github.io/ado/examples/random-walk/) for a
hands-on introduction to how `ado` works.

## Requirements

A basic installation of `ado` only requires a recent Python version (3.10 to
3.14). This will allow you to run
[many of our examples](https://ibm.github.io/ado/examples/examples/) and explore
`ado` features.

### Additional Requirements

Some advanced features have additional requirements:

<!-- markdownlint-disable descriptive-link-text -->

- **Distributed Projects** **_(Optional)_**: To support projects with multiple
  users you will need a remote, accessible MySQL database. See
  [here](https://ibm.github.io/ado/getting-started/installing-backend-services/#using-the-distributed-mysql-backend-for-ado)
  for more details
- **Multi-Node Execution** **_(Optional)_**: To support multi-node or scaled
execution you may need a multi-node RayCluster. See
[here](https://ibm.github.io/ado/getting-started/installing-backend-services/#deploying-kuberay-and-creating-a-raycluster)
for more details
<!-- markdownlint-enable descriptive-link-text -->

In addition, `ado` plugins may have additional requirements for executing
**_realistic_** experiments. For example:

- **_Fine-Tuning Benchmarking_**: Requires a
  [RayCluster with GPUs](https://ibm.github.io/ado/actuators/sft-trainer/#configure-your-raycluster)
- **_vLLM Performance Benchmarking_**: Requires an OpenShift cluster with GPUs

## Install from Source

To install `ado` from source (for development or to track the latest changes):

```shell
git clone https://github.com/IBM/ado.git
cd ado
pip install .
```

Detailed installation instructions are available at
<https://ibm.github.io/ado/getting-started/install/>.

## Contributing

To set up a development environment, run the test suite, or understand code
style and commit conventions, see [DEVELOPING.md](DEVELOPING.md) and
[tests/README.md](tests/README.md).

## Example

This video shows listing
[actuators](https://ibm.github.io/ado/actuators/working-with-actuators/) and
getting the details of an experiment.

Check the [demo page](https://ibm.github.io/ado/getting-started/demo) for more
videos.

[![Watch the video](docs/videos/step1_trimmed_thumbnail.png)](https://github.com/user-attachments/assets/fc4862f3-763b-4967-ab3c-4bd359900a50)

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
