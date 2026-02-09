---
title: 'ado: A python framework for computational experimentation and
  benchmarking'
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

The **a**ccelerated **d**iscovery **o**rchestrator (`ado`) is a Python
package that provides a unified platform for executing computational
experiments at scale and analyzing their results. `ado` addresses a
fundamental challenge faced by developers of research software: the
repeated implementation of common infrastructure features such as
command-line interfaces, experiment configuration management, distributed
execution, data provenance tracking, and collaborative data sharing.
By providing these capabilities through a common framework, `ado` allows
researchers to focus on domain-specific problems rather than
reimplementing shared infrastructure.

At the core of `ado` is the **Discovery Space** abstraction
[@Johnston2025], which formalizes the description of computational
experiment campaigns. A Discovery Space combines an entity space (the
configurations to be explored) with a measurement space (the experiments
to be performed), along with a shared sample store that enables
transparent data reuse across multiple research runs and teams. This
structured approach to describing experimental workflows enables several
key features: (1) workload-agnostic optimization through integration with
state-of-the-art frameworks like Ray Tune [@Liaw2018], (2) automatic
provenance tracking of all experimental data, (3) transparent reuse of
experimental results to avoid redundant computation, and (4) support for
distributed teams through shared, versioned experiment data.

`ado` is designed with extensibility as a core principle, featuring a
plugin architecture that allows researchers to add new experiments
(actuators) and analysis tools (operators) while inheriting the
platform's built-in capabilities. The system leverages Ray [@Moritz2018]
for scalable distributed execution and provides specialized support for
foundation model experimentation, including performance benchmarking for
inference and fine-tuning workloads. By establishing a common base for
computational research infrastructure, `ado` enables the development
community to focus effort on advanced features that benefit the entire
research ecosystem rather than repeatedly solving the same infrastructure
problems.

# Statement of need

Research in systems and computational science frequently requires
executing large-scale experimental campaigns to benchmark performance,
optimize configurations, or explore parameter spaces. Researchers
developing such tools face a common pattern: they must implement
infrastructure for experiment management, distributed execution, data
storage, and result analysis before they can focus on domain-specific
research questions. This repetitive work across projects leads to
fragmented tooling ecosystems where each tool reinvents similar
capabilities with varying levels of robustness and features. The lack of
critical mass around any single infrastructure limits the sophistication
of "nice-to-have" features like comprehensive provenance tracking,
collaborative data sharing, and advanced optimization integration that
could benefit all domains.

`ado` addresses this problem by providing a reusable, extensible harness
for computational experimentation that separates infrastructure concerns
from domain logic. The target audience includes both developers of
research software who need a robust foundation for their tools, and
research groups who want to leverage state-of-the-art optimization and
benchmarking capabilities without building custom infrastructure. By
standardizing how experiments are described, executed, and data is
managed through the Discovery Space abstraction, `ado` enables several
critical improvements over ad-hoc solutions: (1) guaranteed provenance
tracking ensures all experimental data maintains its lineage, (2)
transparent data reuse across experiments and teams reduces redundant
computation, (3) integration with production-grade optimization
frameworks (Optuna [@Akiba2019], Ray Tune) provides access to
sophisticated search algorithms, and (4) a plugin architecture allows
domain experts to contribute experiments while inheriting all platform
capabilities.

`ado` has been applied across diverse domains including large language
model fine-tuning benchmarking, predictive model building for
learning-augmented optimization, geospatial inference performance
analysis, quantum chromodynamics simulations, and tabular data generation
exploration. This breadth of application demonstrates the generality of
the underlying abstractions. Looking forward, the structured
representation of experimental campaigns in `ado` provides a foundation
for AI-assisted research workflows, enabling code generation tools to
automatically produce experiment definitions and allowing natural
language interfaces to formulate executable experimental designs. The
standardized data models and validation mechanisms in `ado` make it an
ideal target for such automation, bridging the gap between high-level
research intent and executable computational experiments.

# State of the field

Guidelines: description of how this software compares to other
commonly-used packages in the research area. If related tools exist,
provide a clear "build vs. contribute" justification explaining your
unique scholarly contribution and why existing alternatives are
insufficient.

- Tools for configuration space exploration
- Other tools for provenance???

# Software design

The design of `ado` centers on providing robust infrastructure for
collaborative computational experimentation while maintaining
extensibility and ease of use. Our design choices reflect deliberate
trade-offs between performance, rigor, and developer experience.

## TRACE Characteristics as Design Requirements

We began by identifying five characteristics necessary for transparent
sharing and reuse of experimental data, collectively termed TRACE
[@Johnston2025]: **T**ime-Resolved (tracking when and how data is added),
**R**econcilable (consistent data representation across operations),
**A**ctionable (enabling execution of measurements), **C**ommon Context
(shared storage with unified schema), and **E**ncapsulated (defining
valid configurations and actions). These characteristics drove our
architectural decisions, as they ensure that multiple users and
operations can understand, use, and extend shared experimental data
without adverse effects or inconsistencies. The TRACE requirements
naturally led us to seek a data model that would inherently exhibit these
properties.

## Discovery Space as Core Abstraction

The Discovery Space abstraction serves as the foundation of `ado`'s
architecture. This choice emerged from the observation that experimental
campaigns have well-defined mathematical structure: a configuration
probability space (what to sample), an action space (what experiments to
run), and a sample set (what has been measured). By encoding this
structure explicitly in the data model, we ensure that Discovery Spaces
naturally exhibit TRACE characteristics—they are encapsulated (defining
valid configurations), actionable (specifying how to measure), and
maintain common context through shared storage. This abstraction also
decouples workload-specific experiments from optimization algorithms,
enabling workload-agnostic search capabilities that are a key
distinguishing feature of `ado`.

## Python and Pydantic: Structure with Validation

We chose Python as the implementation language combined with Pydantic for
data modeling. This decision prioritizes development velocity, ecosystem
integration, and data integrity over raw computational performance.
Pydantic provides strong typing and automatic validation of complex
nested data structures, ensuring that Discovery Spaces, entity
configurations, and measurement results maintain internal consistency.
The declarative nature of Pydantic models also serves as living
documentation of the data schema. We accept the trade-off of lower
performance compared to compiled languages, reasoning that (1) experiment
execution time typically dominates over orchestration overhead, and (2)
Python's rich scientific computing ecosystem provides integration paths
to high-performance components when needed. The validation guarantees
from Pydantic prove particularly valuable for collaborative settings
where multiple users define experiments and configurations.

## Ray for Distributed Execution

For scale-out execution, `ado` leverages Ray [@Moritz2018], a distributed
computing framework for Python. This choice follows naturally from our
Python foundation and provides several benefits: Ray's actor model maps
cleanly to distributed experiment execution, its integration with
optimization libraries (Ray Tune) enables seamless access to
state-of-the-art algorithms, and its transparent distribution of Python
functions minimizes the gap between local prototyping and large-scale
execution. Ray handles complex concerns like fault tolerance, resource
management, and distributed scheduling, allowing `ado` to focus on
domain-specific orchestration logic. Users can scale from single-machine
exploration to multi-node clusters without modifying their experiment
definitions.

## Explicit Schema Design

Throughout `ado`, we favor explicit, verbose schema definitions over
condensed syntax. Entity spaces, measurement spaces, and experimental
configurations are described in structured YAML files with full type
information and validation rules. While this approach can feel verbose
compared to more terse domain-specific languages, it provides several
advantages: (1) the schema is self-documenting and accessible to both
humans and tools, (2) validation can occur before expensive computation
begins, (3) the explicit structure enables tool-assisted creation and
modification of experiments, and (4) the rigorous specification supports
future AI-assisted workflows where code generation tools can reliably
produce valid configurations. We view this choice as optimizing for
long-term maintainability and tool integration over initial authoring
convenience, with the option to add more concise syntactic sugar as usage
patterns emerge.

## Plugin Architecture for Extensibility

`ado` employs a plugin architecture for actuators (experiments),
operators (analysis tools), and storage backends. This design allows the
platform to be extended with domain-specific functionality while
maintaining a stable core. Critically, the plugin system is not limited
to adding experiments—it also enables alternative implementations of core
infrastructure components. For example, different storage backends can be
provided as plugins to optimize for specific performance requirements or
deployment constraints, while alternative sampling strategies or
optimization methods can be contributed as operator plugins. This
architecture supports `ado`'s goal of serving as a common base: domain
experts can contribute their specialized knowledge through plugins while
inheriting all platform capabilities (CLI, provenance, data sharing,
distributed execution). The plugin interface is defined through Python
abstract base classes with Pydantic-validated configuration, ensuring
that plugins integrate cleanly with the type-checked core.

# Research impact statement

Research impact statement: Evidence of realized impact (publications,
external use, integrations) or credible near-term significance
(benchmarks, reproducible materials, community-readiness signals). The
evidence should be compelling and specific, not aspirational.

- Fine-tuning
- Model Building
- Geospatial Performance
- QCD
- TRIM
- Tabular data investigations

Examples/Video

# AI usage disclosure

No generative AI tools were used in the development of this software, the
writing of this manuscript, or the preparation of supporting materials.

- We target GenAI - DoE, structure problem formulation, validation via
  models

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and
Semyeong Oh, and support from Kathryn Johnston during the genesis of this
project.

# References
