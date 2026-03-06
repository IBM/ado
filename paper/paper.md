---
title:
  "ado: A python framework for computational experimentation and benchmarking"
tags:
  - Python
  - benchmarking
  - optimization
  - provenance
  - foundation-models
authors:
  - name: Michael A. Johnston
    orcid: 0000-0000-0000-0000
    corresponding: true
    equal-contrib: true
    affiliation: 1
  - name: Alessandro Pomponio
    equal-contrib: true
    affiliation: 1
affiliations:
  - name: IBM Research Europe - Ireland
    index: 1
    ror: 00hx57361
date: 9 February 2026
bibliography: paper.bib
---

<!-- markdownlint-disable MD025 -->

# Summary

The Accelerated Discovery Orchestrator (ado) is a Python package that addresses
a recurring challenge in research software development: the need to implement
common capabilities for design of experiments (DoE) and execution of the related
computational experiment campaigns. These cross‑cutting capabilities span
methodology (design‑space specification, sampling, analysis), interface (CLI and
configuration management), execution (parallel and scale‑out), and data
(sharing, provenance, and reuse). ado provides these capabilities out-of-the-box
to all domains through a lightweight plugin model, where integrating new
components can be as simple as decorating a Python function. This is enabled by
the core abstraction, _the discovery space_, ado is built on.

Out-of-the-box, ado includes state-of-the-art optimization algorithms and
predictive modeling tools, alongside concrete experiments targeting
foundation-model performance. Our aim is for ado to become a focal point for
developing and consuming advanced capabilities for defining and executing
experiment campaigns that accelerate computational research, from initial design
to final analysis.

# Statement of need

While the domains of computational science are diverse, spanning machine
learning, physics simulation, and hardware design, the process of systematic
experimentation is remarkably uniform. Whether tuning hyperparameters,
benchmarking foundation models, or sweeping simulation parameters, researchers
consistently follow a structured pattern: define a configuration space; select
points within it; execute experiments at those points; record results; and
analyze the outcomes to guide the next iteration. This recurring workflow, the
experiment campaign, is central to modern research and development, yet it is
often managed with tools that fail to capture its essential structure.

While scientific workflow systems like Galaxy, AiiDa, and Pachyderm excel at
executing general directed acyclic graphs (DAGs), they are fundamentally
context-free. They treat each step as a black box to be scheduled, forcing
researchers to repeatedly re-implement common mechanisms for trial submission,
parameter handling, logging, and result collation. This imperative approach
leads to duplicated engineering effort, inconsistent practices, and slower
scientific progress.

ado directly addresses this gap. Instead of orchestrating arbitrary DAGs, ado
provides a semantic experimentation model centered on experiment campaigns.
Users define configuration spaces and operations on them (e.g. sampling or
analysis), declaratively. ado then applies the required orchestration using its
own protocols. For example in sampling workflows, it handles reuse of prior
measurements, trial execution and monitoring, and time‑resolved measurement
recording, maintaining consistency over a shared sample store.  
This approach mirrors the advantages of declarative systems like SQL or
Terraform: reduced boilerplate, fewer errors, and greater clarity.
This declarative, structured, representation of experimental campaigns also
aids code
generation tools to automatically produce experiment definitions and to
formulate design spaces (declarative YAML).

ado extends its core model with valuable support capabilities. It uses a
database for specifications and results that can be distributed to support team
collaboration and transparent result reuse. Built on the Ray execution engine,
ado seamlessly scales from a researcher's laptop to a large remote cluster, with
all functionality accessible via a human-centric CLI and Python API.
Researchers can contribute custom experiments or operators through a simple plugin
interface. The result is a system that is context‑specific yet domain‑agnostic.

# State of the field

Mature workflow managers such as Galaxy, AiiDA, and Pachyderm excel at scalable,
reliable DAG orchestration with strong provenance/lineage and tight alignment to
common execution substrates (HPC/cloud for Galaxy/AiiDA; Kubernetes‑native,
data‑versioned pipelines for Pachyderm). As discussed in the previous section,
they are strong for general workflow execution and provenance but are not ideal
for implementing experiment campaigns as first‑class, semantically constrained
objects.

General black‑box optimization frameworks like Optuna,Ax,Nevergrad and RayTune
are also key components for executing experiment campaigns, providing
gradient-free, multi‑fidelity and multi‑objective optimization. While these tools
are beginning to add data management features, for example, Optuna and RayTuen,
support persistent storage for study resumption, they remain fundamentally
code-centric. This approach requires users to define the optimizer, objective,
and logging in code, which complicates turning individual runs into portable,
team-level campaigns and fragments data reuse patterns.

Emerging robotic lab frameworks also highlight the need for integrated campaign
management. For instance, the Experiment Orchestration System (EOS) provides
rigorous, repeatable execution for physical experiments using a plugin model to
orchestrate lab equipment over Ray. Its scope, however, is intentionally the
physical execution layer—answering how to carry out a specific experiment with
instruments, rather than providing the domain-agnostic, declarative campaign
semantics that ado offers.

We identified that a new approach was necessary as these existing tools lack the
core semantic model for an experiment campaign. Adding this to workflow managers
would conflict with their open-ended DAG design, while adding it to a single
optimizer library would not generalize and would retain a code-first, fragmented
approach. For automated lab systems, their focus is on operational complexity of
physical execution, examples like EOS do not aim to provide domain‑agnostic,
declarative campaign semantics above the lab layer.

ado synergizes with, rather than replaces, existing tools. It can use workflow
managers like Galaxy as trial executors, integrate optimizers like Optuna and Ax
via a stable adapter, and orchestrate physical experiments by coupling with
robotic lab systems like EOS. The fact that ado integrates cleanly with these
systems validates the existence of the semantic gap it fills.

# Software design

## TRACE Characteristics as Design Requirements

We first established TRACE, a set of five governing principles for managing the
artifacts of an experimental campaign. These principles provide a framework for
the entire information lifecycle, ensuring the transparent handling of the
configuration space definition, the execution of experiments, and the data they
produce.

<!-- markdownlint-disable line-length -->

| Characteristic     | Description                                                                                                                      |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| **Time-Resolved**  | Tracks when and how data is added to a study, preserving the time-series of sampling processes.                                  |
| **Reconcilable**   | Provides a mechanism to consistently add data from a common context into a specific study.                                       |
| **Actionable**     | Enables the execution of new measurements to add information, with the necessary instructions contained within the study itself. |
| **Common Context** | Utilizes a shared storage mechanism and a unified schema, allowing data to be shared across multiple studies.                    |
| **Encapsulated**   | Defines what configurations and actions are valid for a study, preventing contamination from unrelated data.                     |

<!-- markdownlint-enable line-length -->

These principles ensure that a design of experiments (DoE) and its associated
data can be understood, shared, extended, and analyzed without introducing
inconsistencies. The TRACE requirements guided our search for a data model that
would inherently exhibit these properties. Discovery Space as a Core Abstraction

## Discovery Space as a Core Abstraction

The Discovery Space is the central abstraction in the system's architecture,
chosen because its structure naturally enables the TRACE characteristics. We
observed that configuration search campaigns have a well-defined mathematical
structure, which we encoded directly into our data model.

A Discovery Space is composed of:

- **A Configuration Probability Space:** This defines the dimensions of the
  configuration space being explored (the "what") and the probability
  distribution governing the selection process (the "how likely").
- **An Action Space:** This defines the set of experiments that can be applied
  to a configuration to measure its properties (the "how to measure").

By explicitly associating the experiments (Action Space) with the configuration
space and linking the resulting measurements (samples) to the space, the Discovery
Space model inherently becomes:

- **Encapsulated and Actionable:** The space itself defines the valid
  configurations and the exact experiments to run on them.
- **Time-Resolved**: It records the sequence of samples generated by each
  operation as a distinct time-series.
- **Common Context**: It relies on a generic, shared schema for storing all
  sample information, making it workload-agnostic and reusable.
- **Reconcilable**: It enforces a clear protocol where data can only be
  associated with a study via an explicit sample operation, ensuring consistency
  even when reading from a shared data store.

This abstraction effectively decouples workload-specific experiments from the
search and optimization algorithms, enabling the kind of versatile,
workload-agnostic capabilities that are a key goal of the program.

## Data Modeling with Python and Pydantic

The core of our architecture is implemented in Python, chosen for its ubiquity
in scientific and research domains and its extensive ecosystem. For all data
modeling, we leverage Pydantic.
Its declarative, type-hinted models provide automatic
validation for all core abstractions (e.g., Discovery Space, configurations),
ensuring data integrity, providing self-documenting schemas, and creating a
clear target for AI code generation.

## Extensibility Through a Plugin Architecture

ado is built on a plugin architecture to ensure extensibility while maintaining
a stable core. This design allows adding new experiments (actuators) analysis
tools (operators) and storage backends. This extensibility is unified through
our Pydantic data models, which serve as the data contract for the system.
Plugins consume and produce validated Pydantic models for configurations,
experiment definitions, and measurement results. This approach enables domain
experts to contribute their specialized knowledge as self-contained plugins
while immediately inheriting all of the platform's core capabilities, such as
the CLI, data provenance, and distributed execution.

## Distributed Execution with Ray

ado leverages Ray for distributed, scale-out execution of operations. This
choice provides a seamless path from local, single-machine prototyping to
large-scale cluster execution without requiring changes to the experiment
definitions. By building on Ray, we delegate complex distributed computing
concerns—such as resource management, scheduling, and fault tolerance—to a
robust, industry-standard framework, allowing ado's core logic to focus on
domain-specific orchestration.

A key feature is that while the ado operations run as Ray applications,
individual plugins (actuators and operators) are not required to use Ray's
constructs. This gives plugin authors flexibility: they can choose to leverage
Ray actors and tasks to scale their own internal logic, or they can orchestrate
experiments through other means, such as by spawning external workflows. This
ensures that ado can accommodate a wide range of use cases and integration
patterns.

# Research impact statement

Although ado is a newly open-source framework, it has been internally
battle-tested on complex industrial workloads and research questions. Its impact
and utility are demonstrated by a range of publicly available artifacts derived
from its extensive internal use, which providing a strong foundation for
community adoption.

- Large-Scale Benchmarking: We generated all fine-tuning benchmarks for IBM's
  watsonx.ai platform. The resulting artifacts, including the sft-trainer plugin
  and recommender models built from this data, are now publicly available.
- Advanced Performance Analysis: The framework was used for detailed performance
  analysis of geospatial models on vLLm. The resulting vllm-performance plugin,
  which includes unique features like automated deployment and tear-down, has
  been open-sourced.
- Accelerated Experimentation: We developed a method for rapidly building
  performance models from prior data to accelerate benchmarking. This novel
  capability is delivered via the trim operator plugin. ado is designed as a
  platform to support reproducible and extensible research.

ado is a community-ready platform for reproducible research, released as
open-source code with extensive documentation. Its plugin architecture provides
a direct path for contributions, and we are actively developing the framework to
accelerate our own research, believing others can derive similar advantages.

# AI usage disclosure

- We target GenAI - DoE, structure problem formulation, validation via models
- AgentMD

# Acknowledgements

We acknowledge contributions from many people during the internal genesis of
ado:

# References
