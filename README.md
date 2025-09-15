# Introduction

This is the repository for the **a**ccelerated **d**iscovery **o**rchestrator
(**`ado`**).

**`ado`** is a unified platform for **executing computational experiments at
scale** and **analysing their results**. It can be extended with new experiments
or new analysis tools. It allows distributed teams of researchers and engineers
to collaborate on projects, execute experiments, and share data.

You can run the experiments and analysis tools already available in **`ado`** in
a distributed, shared, environment with your team. You can also use **`ado`** to
get features like data-tracking, data-sharing, tool integration and a CLI, for
your analysis method or experiment for free.

🧑‍💻 Using **`ado`** assumes familiarity with command line tools.

🛠️ Developing **`ado`** requires knowledge of python.

## Key Features

- 💻 _CLI_: Our human-centric CLI follows [best practices](https://clig.dev)
- 🤝 _Projects_: Allow distributed groups of users to
  [collaborate and share data](https://ibm.github.io/ado/resources/metastore)
- 🔌 _Extendable_: Easily
  [add new experiments](https://ibm.github.io/ado/actuators/creating-custom-experiments),
  [optimizers or other tools.](https://ibm.github.io/ado/operators/creating-operators)
- ⚙️ _Scalable_: We use [ray](https://ray.io) as our execution engine allowing
  experiments and tools to easily scale
- ♻️ _Automatic data-reuse_: Avoid repeating work with
  [transparent reuse of experiment results](https://ibm.github.io/ado/core-concepts/data-sharing).
  `ado` internal protocols ensure this happens only when it makes sense
- 🔗 _Provenance_: As you work, the relationship between the data you create and
  operations you perform are
  [automatically tracked](https://ibm.github.io/ado/getting-started/ado#ado-show-related)
- 🔎 _Optimization and sampling_: Out-of-the-box, leverage powerful optimization
  methods
  [via `raytune`](https://ibm.github.io/ado/operators/optimisation-with-ray-tune)
  or use our
  [flexible in built sampler](https://ibm.github.io/ado/operators/random-walk)

### Foundation Model Experimentation

We have developed `ado` plugins providing advanced experiments for testing
foundation-models:

- ⏱️
  [fine-tuning performance benchmarking](https://ibm.github.io/ado/actuators/sft-trainer)
- ⏱️ inference performance benchmarking (using the
  [vLLM performance benchmark](https://docs.vllm.ai/en/stable/api/vllm/benchmarks/serve.html))
- **COMING SOON** 🔮 inference and fine-tuning prediction

## Requirements

A basic installation of `ado` only requires a recent Python version (3.10+).
This will allow you to run
[many of our examples](https://ibm.github.io/ado/examples/examples) and explore
ado features.

### Additional Requirements

Some advanced features have additional requirements:

<!-- markdownlint-disable descriptive-link-text -->
- **Distributed Projects** **_(Optional)_**: To support projects with multiple
  users you will need a remote, accessible, MySQL database. See
  [here](https://ibm.github.io/ado/getting-started/installing-backend-services#using-the-distributed-mysql-backend-for-ado)
  for more
- **Multi-Node Execution** **_(Optional)_**: To support multi-node or scaling
  execution you may need a multi-node RayCluster. See
  [here](https://ibm.github.io/ado/getting-started/installing-backend-services#deploying-kuberay-and-creating-a-raycluster)
  for more details
<!-- markdownlint-enable descriptive-link-text -->

In addition `ado` plugins may have additional requirements for executing
**_realistic_** experiments. For example,

- **_Fine-Tuning Benchmarking_**: Requires a
  [RayCluster with GPUs](https://ibm.github.io/ado/actuators/sft-trainer#configure-your-raycluster)
- **_vLLM Performance Benchmarking_**: Requires an OpenShift cluster with GPUs

## Install

To install you can execute the following (we recommend you set up a virtual
environment)

```commandline
git clone https://github.com/IBM/ado.git
cd ado
pip install .
```

Alternate instructions to install `ado` can be found here:
<https://ibm.github.io/ado/getting-started/install/>

## Development

Instructions for developing ado are available in [DEVELOPING](DEVELOPING.md).

### Testing

To run unit-tests read [tests/README.md](tests/README.md).

## Example

This video shows listing
[actuators](website/docs/actuators/working-with-actuators.md) and getting the
details of an experiment.

Check [demo](https://ibm.github.io/ado/getting-started/demo) for more videos.

[![Watch the video](website/docs/getting-started/videos/step1_trimmed_thumbnail.png)](https://github.com/user-attachments/assets/fc4862f3-763b-4967-ab3c-4bd359900a50)

## Technical Report

For more details on the Discovery Spaces concept underlying ado, please refer to
this [technical report](https://arxiv.org/abs/2506.21467).

## Acknowledgement

This project is partially funded by the European Union through the Smart
Networks and Services Joint Undertaking (SNS JU) under grant agreement No.
101192750 (Project 6G-DALI).
