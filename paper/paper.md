---
title:
  "ado: a Python framework for computational experimentation and benchmarking"
tags:
  - Python
  - benchmarking
  - design-of-experiments
  - experiment campaigns
  - optimization
  - provenance
  - data sharing
  - data reuse
  - foundation-models
authors:
  - name: Michael A. Johnston
    orcid: 0000-0003-1337-440X
    corresponding: true
    equal-contrib: true
    affiliation: 1
  - name: Alessandro Pomponio
    equal-contrib: true
    affiliation: 1
    orcid: 0000-0003-1655-7500
affiliations:
  - index: 1
    name: IBM Research - Ireland
    ror: 04jnxr720
date: 9 February 2026
bibliography: paper.bib
---

<!-- markdownlint-disable MD025 -->

# Summary

The **Accelerated Discovery Orchestrator (ado)** is a Python package that
addresses a recurring challenge in research software development: implementing
common capabilities for design of experiments (DoE) and execution of related
computational experiment campaigns. These cross-cutting capabilities span
methodology (design-space specification, sampling, analysis), interface (CLI and
configuration management), execution (parallel and scale-out), and data
(sharing, provenance, and reuse).

ado delivers these capabilities across domains through a lightweight plugin
model, where integrating new components can be as simple as decorating a Python
function. This is enabled by ado's core abstraction: the _Discovery Space_.

Out-of-the-box, ado includes state-of-the-art optimization algorithms and
predictive modeling tools, alongside experiments targeting foundation-model
performance. Our aim is for ado to become a focal point for developing and
consuming advanced capabilities for defining and executing experiment campaigns.

# Statement of need

While the domains of computational science are diverse, spanning machine
learning, physics simulation, and hardware design, the process of
experimentation is remarkably uniform. Whether tuning hyperparameters,
benchmarking foundation models, or sweeping simulation parameters, researchers
consistently follow a structured pattern: define a configuration space; select
points; execute experiments; record results; and analyze the outcomes to guide
the next iteration. This workflow, the _experiment campaign_, is central to
modern research and development, yet it is often managed with tools that fail to
capture its essential structure.

Scientific and ML workflow systems like Galaxy, AiiDA, and Kubeflow excel at
executing general directed acyclic graphs (DAGs). However, they are
fundamentally context-free, treating each step as a black box
[@10.1093/nar/gkae410; @Huber2020; @George2022EndtoendML]. When it comes to
experiment campaigns, this forces researchers to implement mechanisms for trial
submission, parameter handling, and result collation.

ado directly addresses this gap. Instead of orchestrating arbitrary DAGs, ado
focuses on the generic experiment campaign process. Users define configuration
spaces and operations on them declaratively. ado then orchestrates the
operations using its own protocols. For example, in sampling workflows, it
handles reuse of prior measurements, trial execution and monitoring, and
time-resolved measurement recording, maintaining consistency over a shared
sample store. This approach mirrors the advantages of declarative systems like
SQL or Terraform: reduced boilerplate, fewer errors, and greater clarity. It
also aids code generators in creating experiment definitions and design spaces.

![A schematic overview ado's architecture. \label{fig:ado}](ADOSchematic.png)

ado extends its core model with valuable support capabilities. Specifications
(configuration spaces, operations) and measurements are stored in a database
with flexible deployment options (local or shared remote). It leverages Ray to
seamlessly scale from a researcher's laptop to a large remote cluster
[@Moritz2018], with all functionality accessible via a human-centric CLI and
Python API. Researchers can contribute custom experiments or operators through a
simple plugin interface (see \autoref{fig:ado}).

# State of the field

Mature workflow managers such as Galaxy, AiiDA, and Kubeflow excel at scalable,
reliable DAG orchestration with strong provenance and tight alignment to common
execution platforms (e.g., HPC for AiiDA; Kubernetes for Kubeflow). As discussed
in the previous section, they are not ideal for implementing experiment
campaigns. ML lifecycle management tools like MLflow provide robust experiment
tracking, metric logging, and artifact management features for individual runs
[@Zaharia2018AcceleratingTM]. However, they lack a higher-level semantic
construct for an experiment campaign.

General black-box optimization frameworks like Optuna, Ax, Nevergrad, and Ray
Tune are also key components for executing experiment campaigns [@Akiba2019;
@olson2025ax; @10.1145/3460310.3460312; @Liaw2018]. While these tools are
beginning to add data management features, for example, persistent storage for
resuming studies, they require users to define the optimizer, objective, logging
etc. in code.

Emerging robotic lab frameworks also highlight the need for integrated campaign
management. For instance, the Experiment Orchestration System (EOS) provides
rigorous, repeatable execution for physical experiments [@Angelopoulos2025_EOS].
The scope of these frameworks is intentionally physical execution, answering how
to carry out a specific repeatable experiment with lab instruments.

We identified that a new approach was necessary, as these existing tools lack
the core semantic model for an experiment campaign. Adding this to workflow
managers would conflict with their open-ended DAG design, while adding it to a
single optimizer library would not generalize and would retain a code-first,
fragmented approach. For automated lab systems, their focus is on managing
operational complexity, not providing domain-agnostic, declarative campaign
semantics above the lab layer.

ado synergizes with, rather than replaces, these tools. It can use workflow
managers as experiment executors, integrate optimization frameworks, and
orchestrate physical experiments by coupling with robotic lab systems. At the
same time, individual experiment implementations within ado's plugin
architecture can leverage frameworks like MLflow for fine-grained,
domain-specific tracking. The fact that ado integrates cleanly with these
systems validates the existence of the gap it fills.

# Software design

## TRACE Design Requirements

We first established TRACE, a set of five requirements for managing the
artifacts of an experimental campaign or study.

<!-- markdownlint-disable line-length -->

| Characteristic     | Description                                                                                                           |
| ------------------ | --------------------------------------------------------------------------------------------------------------------- |
| **Time-Resolved**  | The time series of sampling processes adding data to a study is preserved.                                            |
| **Reconcilable**   | There is a consistent protocol for adding data from a common context into a specific study.                           |
| **Actionable**     | A study must contain all necessary information for adding a new measurement to itself.                                |
| **Common Context** | There is a schema and storage mechanism allowing data to be shared across multiple studies.                           |
| **Encapsulated**   | The study rigorously defines the valid configurations and measurements, preventing contamination from unrelated data. |

<!-- markdownlint-enable line-length -->

A system satisfying these requirements would ensure that a DoE and its
associated data can be understood, shared, extended, and analyzed without
introducing inconsistencies. In this way, the TRACE requirements offer a
concrete implementation of the FAIR Principles (Findable, Accessible,
Interoperable, and Reusable) [@wilkinson2016fair]. Where FAIR describes _what_
qualities a digital asset should possess, TRACE defines _how_ to construct
systems that generate inherently FAIR data from the inception of an experiment
campaign.

## Discovery Space as a Core Abstraction

The TRACE characteristics guided our search for a data model. First, we noted
that configuration search campaigns have well-defined mathematical properties:

- **A Configuration Probability Space**: the definition of the dimensions of the
  configuration space being explored (the "what") and the probability
  distribution governing the selection process (the "how likely").
- **An Action Space:** the set of experiments that can be applied to a
  configuration to measure its properties (the "how to measure").
- **A sample set:** the set of points currently sampled and measured for a given
  combination of a configuration probability space and an action space. It is
  the union of the **sample time series** of operations on that combination.

![A view of a Discovery Space data model instance. The left-hand side
shows the key data components of the model. The right-hand side shows the
sampling and measurement process for a point.\label{fig:discoveryspace}](discovery_space_v1.drawio.svg){
width=50% }

The **Discovery Space**, the central data model in the system's architecture,
combines these properties (see \autoref{fig:discoveryspace}). By containing the
configuration probability space and action space definitions it is
**encapsulated** and **actionable**; it is **time-resolved** as it contains the
sample set.

![Data sharing between Discovery Spaces. Each space reads/writes
samples and time series
membership details from a shared sample store (_common context_).
In this case both spaces have the same action space and contain point X.
If point X is measured on Discovery Space A, it will not appear in the sample set
of Discovery Space B until it is requested to be measured via B (_reconcilable_).
 \label{fig:ds_interaction}](ds_interaction_v2.drawio.png){
width=80% }

We obtain a **common context** by storing the sample time series in a shared
sample store with a common schema. Finally, it is **reconcilable** as it
enforces a strict membership rule for its sample set: only samples in the
common-context that are associated with an operation conducted on that space are
included. (see \autoref{fig:ds_interaction}). Hence, it displays the TRACE
characteristics.

The Discovery Space abstraction effectively decouples workload-specific
experiments from the search and optimization algorithms, enabling the kind of
versatile, workload-agnostic capabilities that are a key goal of the program.

## Pydantic-based Core Architecture

ado is implemented in Python, chosen for its ubiquity in scientific domains.
Central to ado is the Pydantic framework, which we use for all data modeling.
These Pydantic models serve as the primary user-facing components, providing
automatic validation, self-documenting schemas, and a clear target for AI code
generation.

They also serve as the data-contract for ado's extensible plugin architecture.
This allows domain experts to contribute self-contained plugins for experiments,
sampling, and analysis tools, which immediately and safely inherit all the
platform's core capabilities

# Research impact statement

ado has been internally battle-tested on complex industrial workloads and
research questions [@johnston2025efficientreuseablecloudconfiguration]. Its
impact and utility are demonstrated by a range of publicly available artifacts,
which provide a strong foundation for community adoption.

- **Large-Scale Benchmarking:** We generated all fine-tuning benchmarks for
  IBM's watsonx.ai platform. The resulting artifacts, including the
  [sft-trainer plugin](https://ibm.github.io/ado/actuators/sft-trainer/) and
  [recommender models built from this data](https://github.com/IBM/ado/tree/main/plugins/custom_experiments/autoconf),
  are publicly available.
- **Advanced Performance Analysis:** The framework was used for detailed
  performance analysis of geospatial models on vLLM [@10.1145/3600006.3613165].
  The resulting
  [vllm-performance plugin](https://ibm.github.io/ado/actuators/vllm_performance/),
  which includes unique features like automated deployment and tear-down, has
  been open sourced.
- **Accelerated Benchmarking:** We developed a method for rapidly building
  performance models from prior data to accelerate benchmarking. This novel
  capability is delivered via
  [the TRIM operator plugin](https://ibm.github.io/ado/operators/trim/). TRIM
  applies feature-importance-guided active learning to select and measure a
  minimal set of configurations for building an AutoGluon tabular surrogate.

ado is a community-ready platform for reproducible research, released as open
source code with extensive documentation. Its plugin architecture provides a
direct path for contributions, and we are actively developing the framework to
accelerate our own research, believing others can derive similar advantages.

# AI Usage Disclosure

Generative AI was used for the manuscript and codebase, with human authors
reviewing and taking responsibility for the final content. For the manuscript,
AI helped refine sentences and check for compliance with submission guidelines.
For code, AI agents assisted with development tasks within a structured
framework that includes coding rules, self-describing schemas with validation,
and mandatory human review of all contributions.

# Acknowledgements

ado is partially funded by the European Union through the Smart Networks and
Services Joint Undertaking (SNS JU) under grant agreement No. 101192750 (Project
6G-DALI).

We acknowledge contributions from many people in the development of ado:
Vassilis Vassiliadis, Christian Pinto, Srikumar Venugopal, Daniele Lotito,
Michele Gazzetti, Burkhard Ringelin, Boris Lublinsky, Renato Maia, Renato
Cerqueira, Gabriela Pinheiro, Raphael Melo, Christoph Hagleitner.

# References
