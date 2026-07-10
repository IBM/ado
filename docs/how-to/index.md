# How-To Guides

These guides help you get the most out of `ado` depending on what you want to
do. Choose a path that matches your role, or read on to see how `ado` creates a
powerful partnership with coding agents.

## Choose Your Path

- :octicons-beaker-24: **[Researchers & Benchmarkers](researchers.md)** — You
want to run experiments, collect data, and analyse results, using the tools
already available in `ado`, or building new ones as needed.
- :electric_plug: **[Plugin Developers](plugin-developers.md)** — You want to
write new experiments, analysis tools or sampling methods, extending `ado`'s
capabilities for everyone.
- :octicons-tools-24: **[Core Developers](core-developers.md)** — You want to
contribute to the `ado` framework itself

---

## `ado` and Coding Agents

`ado` is designed from the ground up to partner with coding agents, creating a
powerful automated research assistant. This isn't just about agent skills; `ado`'s
core design allows an agent to reason about and execute a complete research
workflow.

<!-- markdownlint-disable line-length -->

| Step                    | Capability                                   | Benefit for Agent-Driven Research                                                                                                                                                                                          |
| :---------------------- | -------------------------------------------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **1. Discover**         | **Self-Describing Experiments and Operators**| Before acting, an agent must understand what's possible. `ado` enables agents to discover exactly what capabilities are available, and what's required to use them, in a structured manner rather than parsing code.       |
| **2. Model**            | **Clear Separation of Concerns**             | Once possibilities are known, the agent structures the problem. `ado` provides clear separation between the _what_, the _how_, and the _action_, allowing the agent to reason about each part of the problem independently.|
| **3. Act & Verify**     | **Pre-Run Validation**                       | With a model in mind, the agent can safely execute. Using `ado template` and `ado create --dry-run`, it can create a tight **generate → validate → fix → run** loop.                                                       |
| **4. Analyze & Refine** | **Structured & Queryable Data**              | All data and metadata created via `ado` is stored in a structured database. This allows the agent to analyze, compare, link, and synthesize data to decide on the next course of action.                                   |

<!-- markdownlint-enable line-length -->

Together, these properties enable a **closed research loop**: an agent can
describe a problem, run experiments, read the measurements, and refine its
approach, all while operating at a high level of abstraction — manipulating spaces
and operators rather than writing bespoke glue code.

See [Core Concepts](../core-concepts/concepts.md) and
[The ado CLI](../getting-started/ado.md) for more.
