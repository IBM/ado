# ado and Agents

ado's typed resources, expressive CLI, and bundled agent skills make it a
natural fit for agentic research workflows. Once prompted with a research
problem, an agent can design the Discovery Space, write new experiments or reuse
existing ones, and run the full exploration loop.

## Why ado works well with agents

| Feature | How it helps agents |
| :------ | :------------------ |
| 🤖 **Bundled agent skills** | Ready-made skills guide agents through end-to-end discovery workflows — from formulating a problem to analysing results. |
| 🔍 **Self-describing resources** | Experiments and operators declare their required properties, so an agent can discover what's available and what's needed without parsing code. |
| 🧱 **Validated schemas** | Research intent is expressed as structured, validated configurations — constraining the agent to well-defined inputs rather than free-form code generation, reducing hallucinations and keeping experiments repeatable. |
| ✅ **Safe execution loop** | `ado template` and `--dry-run` support a tight **generate → validate → fix → run** cycle before any work is committed. |
| 📦 **Structured & queryable results** | All measurements and metadata are stored in a structured database, giving agents clean access to data for analysis and refinement. |
| 🔗 **Full provenance** | Every result is annotated with resource relationships and plugin versions, so an agent always knows where data came from and how to reproduce it. |

## What you can ask your agent to do

With the skills we provide, you can ask your agent to handle complex tasks in
plain language:

<!-- markdownlint-disable line-length -->

| Ask your agent to…    | Example prompt                                                                        | Skill used                    |
| :-------------------- | :------------------------------------------------------------------------------------ | :---------------------------- |
| **Run a full study**  | "Design, run, and analyse an experiment to find the best vLLM config for throughput." | `conduct-empirical-study`     |
| **Create YAML files** | "Formulate a discovery space for my new component."                                   | `formulate-discovery-problem` |
| **Summarise results** | "Examine the operation I just ran and tell me what it found."                         | `examining-ado-operations`    |
| **Inspect a project** | "Give me an overview of all experiments run in this project so far."                  | `examining-ado-project`       |
| **Query data**        | "Find all entities where `lora_rank` was 8 and export their `validation_loss`."       | `query-ado-data`              |

<!-- markdownlint-enable line-length -->

## Getting set up for agent-assisted workflows

If you haven't already followed Path B in [Getting Started](getting-started.md),
clone the repository and set up the full environment:

```shell
git clone https://github.com/IBM/ado.git
cd ado
uv sync --group test
source .venv/bin/activate
```

Open the cloned `ado` folder as your workspace root in an agent-enabled IDE
(Claude, Cursor, Bob, and others will automatically detect and load the
built-in skills).

---

## Next steps

<!-- markdownlint-disable no-inline-html -->
<div class="grid cards" markdown>

- :octicons-beaker-24:{ .lg .middle } **See ado in action**

    ---

    Walk through end-to-end examples that cover common research workflows.

    [Choose an example :octicons-arrow-right-24:](examples/index.md)

- :octicons-code-24:{ .lg .middle } **Extend ado**

    ---

    Add custom experiments, benchmarks, or search strategies via the plugin model.

    [Developer Guide :octicons-arrow-right-24:](../developer-guide/getting-started.md)

</div>
<!-- markdownlint-enable no-inline-html -->
