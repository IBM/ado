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
common capabilities for design of experiments and executing the related
computational experiment campaigns. These cross‑cutting capabilities span
methodology (design‑space specification, sampling, analysis), interface (CLI and
configuration management), execution (parallel and scale‑out), and data
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

Across computational science and engineering, experiment campaigns, structured
sequences of measurements guided by design‑of‑experiments principles, are now
central to research and development. Such campaigns underpin hyperparameter
optimization in machine learning, ablation and benchmarking of foundation
models, digital‑twin and physics‑based simulation sweeps, compiler and hardware
configuration tuning, and many other activities. Despite the diversity of these
domains, they share a recurring pattern: define a configuration space; select
points within it; execute experiments at those points; record results; and
analyze the outcomes to determine subsequent steps.

Widely used scientific workflow systems, such as Galaxy, AiiDa, and Pachyderm,
address part of this need by enabling users to construct and execute general
directed acyclic graphs (DAGs) of tasks across HPC and cloud environments. They
do not ascribe scientific meaning to steps, treating each node as a black-box
that the engine schedules and monitors.Researchers use them to automate
workflows, defining nodes, wiring edges, pass artifacts, and capturing outputs,
a fundamentally imperative programming model Workflow engines emerging from a
particular domain may have domain specific glue to aid implementation e.g.
pre-defined tasks, connectors to database of certain types. In this way they are
context-free but often have a domain-specific flavour.

However, the core loop of an experiment campaign is structurally uniform and
does not require arbitrary DAG construction. Selecting points from a
configuration space (via sampling or optimization), launching experiments at
those points, collecting measurements, and storing results is a recurrent,
domain‑agnostic pattern. Hence when using general‑purpose workflow engines for
experiment campaigns, researchers must repeatedly re‑implement common
mechanisms: trial submission, parameter handling, logging, measurement
collection, and output collation. Higher‑level capabilities, like systematic
parameter/result management or optimization‑driven steering, are added in an
ad‑hoc manner, leading to duplicated engineering effort, inconsistent practices,
reduced portability, and slower progress.

ADO directly addresses this gap. Instead of orchestrating arbitrary DAGs, ADO
provides a semantic experimentation model centered on experiment campaigns.
Users define configuration spaces and, independently, operations on them (e.g.
sampling or analysis), declaratively. ado then applies the required
orchestration:  
for example in sampling workflows, handling point selection, reuse of prior
measurements, trial execution and monitoring, and time‑resolved measurement
recording. This approach mirrors the advantages of declarative systems like SQL
or Terraform: reduced boilerplate, fewer errors, and greater clarity.
Researchers can contribute custom experiments or operators through a simple
plugin interface. The result is a system that is context‑specific yet
domain‑agnostic.

ado extends its core-model features with valuable support capabilities.
Declarative specifications (operations, configuration spaces) are stored in a
database and their relationships tracked. Both the sample database and the
specification database can be local or distributed, allowing distributed teams
to collaborate and in the case of distributed sample databases, transparently
reuse results. With Ray as the default (but optional) execution engine this
means ado can support a single researcher running on a laptop, to a distributed
team executing on a large remote cluster. All functionality is accessible via a
human‑centric CLI and a Python API. The target audience includes research and
engineering teams that routinely conduct structured experiments, such as
ML/GenAI benchmarking and fine‑tuning, simulation‑based design studies, systems
and hardware tuning, and data/ETL configuration sweeps.

# State of the field

_Guidelines: description of how this software compares to other commonly-used
packages in the research area. If related tools exist, provide a clear "build
vs. contribute" justification explaining your unique scholarly contribution and
why existing alternatives are insufficient._

A. Key challenges for configuration search Table I presents a sample of the
state-of-the-art for general optimization and configuration search systems
juxtaposed with the contributions of this research.

Workload Specificity

State-of-the-Art

- Existing optimization techniques are usually tailored to specific workloads,
  limiting general applicability [Cite]
- Black‑box optimization frameworks (e.g., Vizier, Optuna, BOAH) could offer
  workload‑agnostic search, but mapping configuration parameters into their
  formats is non‑trivial. (Weak) What exists and what it solves. Three mature
  systems—Galaxy, AiiDA, and Pachyderm—are the de facto state of the art for
  orchestrating computational work:

Galaxy provides a general scientific workflow system with a GUI for constructing
multi‑step analyses and, despite its origins in computational biology, is now
explicitly characterized as domain‑agnostic across disciplines. AiiDA offers a
general workflow engine with Python‑defined work chains and deep, graph‑based
provenance that is used broadly in computational science, not just a single
discipline. Pachyderm, populate in data-science and ML, supplies
Kubernetes‑native orchestration where pipelines are containerized
transformations connected in DAGs and triggered by data dependencies, with
immutable, Git‑like data lineage. [aiida.net], [arxiv.org] [github.com],
[backend.orbit.dtu.dk]

General DAG orchestration.

State‑of‑the‑art workflow managers excel at general DAG orchestration and scale:
Galaxy’s GUI workflows let users chain arbitrary tools; AiiDA’s work chains
provide flexible multi‑step automation; Pachyderm’s YAML‑defined, containerized
pipelines run as DAGs on Kubernetes with autoscaling and immutable lineage.
These engines are domain‑agnostic at the core, but appear domain‑shaped because
their ecosystems (wrappers, plugins, communities) cluster around certain
disciplines and infrastructures. The target and automated on communiities
infrastructure of choice well: Galaxy integrates with schedulers; AiiDA
automates HPC runs; Pachyderm is Kubernetes‑native with autoscaling.
[aiida.net], [aiida.net]

Domain Specificity

Their domain flavor comes from ecosystems—tool wrappers, plugins, and common
workloads—not from constraints enforced by the engines themselves. (Galaxy’s
community, for example, spans far beyond genomics because the engine is
domain‑agnostic; AiiDA is positioned as a general, provenance‑rich workflow
manager; Pachyderm positions itself as data‑centric and language‑agnostic.)

Provenance and lineage.

Galaxy captures histories; AiiDA maintains a rich, queryable provenance graph;
Pachyderm versions data with immutable lineage and DAG views. [arxiv.org],
[aiida.readthedocs.io]

Where the fit ends.

Despite their strengths, these systems remain context‑free: they orchestrate
black‑box steps and preserve lineage, but they do not model experiments
(configurations, measurements, campaigns, reuse, optimization feedback) as
first‑class concepts. Users who need an experimentation engine must layer
external libraries and bespoke data models atop a DAG substrate — reintroducing
the disadvantages outlined above (imperative boilerplate, limited reuse,
fragmented optimization, and no shared experimental memory).

Build vs. contribute.

Could one contribute an “experiment‑campaign mode” to an existing engine rather
than build a new one? In practice, the unique scholarly contribution here is
semantic constraint: intentionally limiting users to an experiment lifecycle to
gain higher semantic leverage—native optimization, principled reuse across
configurations, and portable, schema‑validated measurement stores. Retrofitting
this into a context‑free DAG engine would either (a) remain a bolt‑on that
cannot become the engine’s first‑class abstraction (preserving DAG generality
but forfeiting semantic guarantees), or (b) conflict with core design goals that
emphasize open‑ended composition and domain neutrality in the engine itself
(Galaxy’s general workflows, AiiDA’s flexible work chains, Pachyderm’s
containerized DAGs)

Unique contribution.

ADO proposes a complementary paradigm: a domain‑agnostic, context‑specific
experimentation engine that elevates experiment campaigns to the primary unit of
work. By providing a declarative way to express experimentation intent,\
first‑class optimization, transparent measurement reuse, and a shared
experimental memory, ADO closes the gap left by context‑free DAG orchestrators
while remaining interoperable with them for execution (e.g., calling
containerized tools, HPC jobs, or services as actuators). In short, where the
field offers powerful, scalable orchestration, ADO contributes the semantic
machinery of experimentation—the missing layer that turns repeated, costly
“pipelines” into an accumulating, optimization‑aware research process.

[FLOW FROM LAPTOP SINGLE-USER to DISTRIBTUED-CLUSTER DISTRIBUTED-TEAM]

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

Concretely, ADO offers:

First‑class Discovery Space (typed configurations + measurements + sample store)
for comparability and long‑term interpretability. Declarative experiment
campaigns with imperative extensibility, minimizing boilerplate while preserving
power. Built‑in optimization/search (DoE, Bayesian/HPO, multi‑objective) as core
primitives rather than addons. Transparent measurement reuse (experiment‑aware
memoization) to avoid redundant trials and reduce compute cost. A shared,
structured experimental memory (metastore + access‑controlled sample stores)
that compounds knowledge across projects and teams.

# AI usage disclosure

No generative AI tools were used in the development of this software, the
writing of this manuscript, or the preparation of supporting materials.

- We target GenAI - DoE, structure problem formulation, validation via models
- AgentMD

# Acknowledgements

We acknowledge contributions from many people during the internal genesis of
ado:

# References
