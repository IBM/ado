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
a recurring challenge in research software development: the need to repeatedly
re‑implement common capabilities for designing and executing computational
experiment campaigns. These cross‑cutting capabilities span methodology
(design‑space specification, sampling, analysis), interface (CLI and
configuration management), execution (parallel and scale‑out runtimes), and data
(sharing, provenance, and reuse). ado provides domain‑agnostic implementations
of these capabilities that developers can leverage through a lightweight plugin
programming model. In many cases integrating a new experiment, sampling method
or analysis tool, can be as simple as decorating a Python function.

ado includes state-of-the-art optimization and sampling algorithms, as well as
predictive model construction methods, out-of-the-box. Beyond these general
capabilities, it also includes experiments targeting foundation‑model inference
and fine‑tuning performance. In addition to being useful to researchers in the
Systems for AI domain, these experiments are a concrete illustration of how ado
can be used to enhance research work. Our aim is for ado to become a focal point
for developing and consuming advanced cross‑cutting capabilities that streamline
the design, execution, and analysis of computational experiment campaigns across
the research ecosystem.

# Statement of need

Research in systems and computational science frequently relies on large‑scale
experimental campaigns to benchmark performance, optimize configurations, and
explore complex parameter spaces. Developers of research tools repeatedly
implement the same cross‑cutting capabilities—such as experiment design,
optimization workflows, configuration management, distributed execution, data
handling, and results analysis—alongside their domain‑specific logic. This
duplication leads to a fragmented ecosystem in which each project re‑creates
similar infrastructure with varying robustness and limited interoperability. A
consequence of this fragmentation is high friction around reuse of general
experimental methods. Tools often become tightly coupled to their domain, making
it difficult to share or extend methods across projects or research groups. The
absence of a widely adopted platform also limits the development of advanced
shared capabilities such as comprehensive provenance tracking, collaborative
data sharing, and seamless integration of optimization or analysis techniques.

[ADD BRIEF STATE-OF-THE-ART GAPS]

ado addresses these challenges by providing a reusable, extensible harness for
computational experimentation that cleanly separates platform‑level concerns
from domain‑specific logic. [ADD USP]

Its target audience includes (1) developers of
research software who require a reliable foundation for implementing
experimental workflows, and (2) research groups seeking state‑of‑the‑art
optimization, sampling, and benchmarking capabilities without maintaining these
core components themselves. By standardizing how experiments, sampling
procedures, and analysis methods are described, executed, and managed, ado
provides several advantages over ad‑hoc solutions:

1. Transparent data reuse across experiments and teams reduces redundant
   computation.
2. Comprehensive provenance tracking preserves lineage for all data products.
3. A structured DOE description language ensures campaigns are validated and
   reproducible.
4. A lightweight plugin architecture, based on simple function decoration,
   allows domain experts to contribute experiments and analysis tools while
   inheriting the full set of platform capabilities.
5. Automatic interoperability between experiments, samplers, and analysis
   methods—for example, all experiments can directly leverage advanced
   optimization algorithms from systems such as Ray Tune.

[ADD CONCRETE EXAMPLE]

# State of the field

_Guidelines: description of how this software compares to other commonly-used
packages in the research area. If related tools exist, provide a clear "build
vs. contribute" justification explaining your unique scholarly contribution and
why existing alternatives are insufficient._

A. Key challenges for configuration search Table I presents a sample of the
state-of-the-art for general optimization and configuration search systems
juxtaposed with the contributions of this research.

The first challenge is that the featured techniques are cus- tomised to specific
workloads which limits their applicability. Workload agnostic optimization could
be implemented through black box optimization (BBO) frameworks such as Vizier
[9], Optuna [10] and BOAH [11] that provide robust and scalable implementations
of techniques such as Bayesian optimization. While BBO frameworks have been
applied to domains such as hyperparameter tuning of LLM, computational
chemistry, and finance, these have not found wide application to configuration
search. This is in part due to the effort required to map configuration
parameters to the formats and techniques used by these tools.

Another challenge is to manage the costs of sampling con- figuration spaces. One
possible solution would be to store and reuse samples gathered from previous
explorations of the con- figuration space. This would not only speed up
configuration search through "bootstrapping" but would also amortise the costs
of sampling across multiple explorations. In addition, this could accelerate
creation of prediction models that require sub- stantial amount of training data
which is time-consuming and expensive to collect [2], [9]. Maintaining
provenance of the configuration samples also enables checking if performance
models are consistent, thereby enhancing reproducibility [14]. Last but not the
least, is the related challenge of managing configuration spaces in dynamic
environments. Configuration spaces are impacted by changes in the underlying
software and hardware infrastructure, common to cloud environments, which could
render existing search solutions obsolete. Repeat- ing the search (regularly)
adds time and cost overheads which could be avoided if the validity of the
solution can be assessed.

B. Our Objectives

Our goal was to develop a configuration search framework that would enable
efficient and reproducible search of multi- dimensional configuration spaces.
Table I lists our objectives to achieve this goal. Our operational objectives
were to support configuration search for any workload with multiple optimization
methods using the same framework (Workload agnostic and Multiple Optimization
Methods). Our data-centered objectives were to identify where existing data is
available and transparently use it to save the cost of acquiring it again
(Transparent Sharing). This enables incre- mental exploration, where a search
can reuse (partial) results from previous searches, as well as aiding
reproducibility of the results. Going further than the state-of-the-art, we also
aimed at reusing data acquired in one configuration space to inform a search on
a different but related configuration space (Distributed Sharing). Satisfying
both these objectives requires a data model that provides a robust and flexible
representation of configuration spaces and their relationship to the data
gathered by testing sample configurations. This data model should also abstract
the configuration space from the actual optimization techniques used for search.
While recent publications have introduced abstractions for specifying the
configuration space [2], [7], as yet there is no goal for extending these
abstractions to allow reusing and building upon existing experimental data from
configuration search studies.

# Software design

The design of `ado` centers on providing a robust platform providing common
capabilities for collaborative computational experimentation while maintaining
extensibility and ease of use. Our design choices reflect deliberate trade-offs
between performance, rigor, and developer experience.

[ + KUBERNETES MODEL]

## TRACE Characteristics as Design Requirements

We began by identifying five characteristics necessary for transparent sharing
and reuse of experimental data, collectively termed TRACE [@Johnston2025]:
**T**ime-Resolved (tracking when and how data is added), **R**econcilable
(consistent data representation across operations), **A**ctionable (enabling
execution of measurements), **C**ommon Context (shared storage with unified
schema), and **E**ncapsulated (defining valid configurations and actions). These
characteristics drove our architectural decisions, as they ensure that multiple
users and operations can understand, use, and extend shared experimental data
without adverse effects or inconsistencies. The TRACE requirements naturally led
us to seek a data model that would inherently exhibit these properties.

## Discovery Space as Core Abstraction

- The key is
- (a) moving the experiment into the design space
- (b) associating the measurements of operations on the design space with the
  design space.

The Discovery Space abstraction serves as the foundation of `ado`'s
architecture. This choice emerged from the observation that experimental
campaigns have well-defined mathematical structure: a configuration probability
space (what to sample), an action space (what experiments to run), and a sample
set (what has been measured). By encoding this structure explicitly in the data
model, we ensure that Discovery Spaces naturally exhibit TRACE
characteristics—they are encapsulated (defining valid configurations),
actionable (specifying how to measure), and maintain common context through
shared storage. This abstraction also decouples workload-specific experiments
from optimization algorithms, enabling workload-agnostic search capabilities
that are a key distinguishing feature of `ado`.

## Python and Pydantic: Structure with Validation

We chose Python as the implementation language combined with Pydantic for data
modeling. This decision prioritizes development velocity, ecosystem integration,
and data integrity over raw computational performance. Pydantic provides strong
typing and automatic validation of complex nested data structures, ensuring that
Discovery Spaces, entity configurations, and measurement results maintain
internal consistency. The declarative nature of Pydantic models also serves as
living documentation of the data schema. We accept the trade-off of lower
performance compared to compiled languages, reasoning that (1) experiment
execution time typically dominates over orchestration overhead, and (2) Python's
rich scientific computing ecosystem provides integration paths to
high-performance components when needed. The validation guarantees from Pydantic
prove particularly valuable for collaborative settings where multiple users
define experiments and configurations.

## Explicit Schema Design

Throughout `ado`, we favor explicit, verbose schema definitions over condensed
syntax. Entity spaces, measurement spaces, and experimental configurations are
described in structured YAML files with full type information and validation
rules. While this approach can feel verbose compared to more terse
domain-specific languages, it provides several advantages:

1. the schema is self-documenting and accessible to both humans and tools
2. validation can occur before expensive computation begins
3. the explicit structure enables tool-assisted creation and modification of
   experiments,
4. the rigorous specification supports future AI-assisted workflows where code
   generation tools can reliably produce valid configurations.

We view this choice as optimizing for long-term maintainability and tool
integration over initial authoring convenience, with the option to add more
concise syntactic sugar as usage patterns emerge.

## Ray for Distributed Execution

For scale-out execution, `ado` leverages Ray [@Moritz2018], a distributed
computing framework for Python. This choice follows naturally from our Python
foundation and provides several benefits: Ray's actor model maps cleanly to
distributed experiment execution, its integration with optimization libraries
(Ray Tune) enables seamless access to state-of-the-art algorithms, and its
transparent distribution of Python functions minimizes the gap between local
prototyping and large-scale execution. Ray handles complex concerns like fault
tolerance, resource management, and distributed scheduling, allowing `ado` to
focus on domain-specific orchestration logic. Users can scale from
single-machine exploration to multi-node clusters without modifying their
experiment definitions.

## Plugin Architecture for Extensibility

`ado` employs a plugin architecture for actuators (experiments), operators
(analysis tools), and storage backends. This design allows the platform to be
extended with domain-specific functionality while maintaining a stable core.
Critically, the plugin system is not limited to adding experiments—it also
enables alternative implementations of core infrastructure components. For
example, different storage backends can be provided as plugins to optimize for
specific performance requirements or deployment constraints, while alternative
sampling strategies or optimization methods can be contributed as operator
plugins. This architecture supports `ado`'s goal of serving as a common base:
domain experts can contribute their specialized knowledge through plugins while
inheriting all platform capabilities (CLI, provenance, data sharing, distributed
execution). The plugin interface is defined through Python abstract base classes
with Pydantic-validated configuration, ensuring that plugins integrate cleanly
with the type-checked core.

# Research impact statement

_Research impact statement: Evidence of realized impact (publications, external
use, integrations) or **credible near-term significance (benchmarks,
reproducible materials, community-readiness signals)**. The evidence should be
compelling and specific, not aspirational._

`ado` has been applied across diverse domains including large language model
fine-tuning benchmarking, predictive model building for learning-augmented
optimization, geospatial inference performance analysis, quantum chromodynamics
simulations, and tabular data generation exploration. This breadth of
application demonstrates the generality of the underlying abstractions.

- Internal usage (watsonx)
- Geospatial model enablement
- Models produced (autconf)
- Model building (trim)
- Provide code for all these capabilities both for the relevant communities and
  as exemplars.
- We have provided extensive documentation including examples and reference for
  the capabilities.

Looking forward, the structured representation of experimental campaigns in
`ado` provides a foundation for AI-assisted research workflows, enabling code
generation tools to automatically produce experiment definitions and allowing
natural language interfaces to formulate design spaces. The standardized data
models and validation mechanisms in `ado` make it an ideal target for such
automation, bridging the gap between high-level research intent and executable
computational experiments.

# AI usage disclosure

No generative AI tools were used in the development of this software, the
writing of this manuscript, or the preparation of supporting materials.

- We target GenAI - DoE, structure problem formulation, validation via models
- AgentMD

# Acknowledgements

We acknowledge contributions from many people during the internal genesis of
ado:

# References
