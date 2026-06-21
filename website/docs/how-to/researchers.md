# Performing Research with ado

This guide is for **researchers and benchmarkers**. You're in the right place if
you want to run experiments, explore parameter spaces, and analyze results
without getting bogged down in custom scripting and data management.

We'll cover the two primary ways to work with `ado` and show you how to
seamlessly transition from _using_ tools to _building_ them when your research
demands it.

---

## Choose Your Workflow: Manual Control or AI-Assisted

`ado` is powerful on its own and even more so with a coding agent. Your choice
depends on how you like to work. Do you prefer direct, hands-on control via a
familiar CLI, or do you want an AI partner to accelerate complex and exploratory
tasks?

<!-- markdownlint-disable line-length -->

|                 | **Option A: The Direct CLI Path**                                                    | **Option B: The AI-Assisted Path**                                                                             |
| :-------------- | :----------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------- |
| **Best For**    | Quick starts, existing scripts, and experienced CLI users who prefer manual control. | Exploratory research, complex parameter spaces, and iterative analysis where an AI can handle the boilerplate. |
| **Setup**       | `pip install ado-core` into any virtual environment.                                 | `git clone` the repository and use `uv sync` to create a comprehensive local environment.                      |
| **Key Feature** | Lightweight and integrates into any standard Python workflow.                        | Unlocks powerful, pre-built **agent skills** that automate complex research tasks.                             |

<!-- markdownlint-enable line-length -->

---

## Option A: The Direct CLI Path

For direct, manual control, install `ado-core` from PyPI. This is the fastest
way to get started and is perfect for integrating `ado` into existing scripts or
for users who are comfortable working directly with the CLI.

    python -m venv ado-venv
    source ado-venv/bin/activate
    pip install ado-core

This simple setup is all you need to run built-in experiments and operators,
follow our examples, and leverage the full power of `ado` from your terminal.

See [Installation](../getting-started/install.md) for more options, including
installing plugins for specific experiment types.

---

## Option B: The AI-Assisted Path

To unlock the full potential of `ado` as an automated research partner, clone
the repository and work inside it. This approach gives your coding agent access
to a library of built-in **skills**, enabling it to perform complex workflows on
your behalf.

### 1. Clone and Install

    git clone https://github.com/IBM/ado.git
    cd ado
    uv sync --group test
    source .venv/bin/activate

!!! info

`uv sync` installs `ado` in editable mode along with all its development and
testing dependencies, creating the perfect environment for both research and
future plugin development.

### 2. Activate the Agent

Simply open the cloned `ado` folder as the workspace root in your agent-enabled
IDE. Many coding agents, include Claude, Cursor and Bob, should automatically
detect and load the skills. If your IDE doesn't create a link to `.cursor` for
it.

### 3. What Your Agent Can Do For You

With the skills loaded, you can ask the agent to handle complex tasks in natural
language. Instead of writing YAML and CLI commands by hand, you can delegate:

<!-- markdownlint-disable line-length -->

| Ask Your Agent to...  | Example Command (Natural Language)                                                        | Underlying Skill              |
| :-------------------- | :---------------------------------------------------------------------------------------- | :---------------------------- |
| **Run a full study**  | "Design, run, and analyze an experiment to find the best vLLM config for max throughput." | `conduct-empirical-study`     |
| **Create YAML files** | "Formulate a discovery space for my new component."                                       | `formulate-discovery-problem` |
| **Summarize results** | "Examine the operation I just ran and tell me what it found."                             | `examining-ado-operations`    |
| **Inspect a project** | "Give me an overview of all experiments run in this project so far."                      | `examining-ado-project`       |
| **Query data**        | "Find all entities where the `lora_rank` was 8 and export their `validation_loss`."       | `query-ado-data`              |

<!-- markdownlint-enable line-length -->

These skills are powerful because `ado`'s core design, with its self-describing
experiments, clear separation of concerns, and structured data, gives the agent
a solid foundation to work with. For more on this, see the
[`ado` and coding agents](index.md#ado-and-coding-agents) guide.

---

## The Universal `ado` Research Workflow

Whether you choose the CLI path or the agent path, the core research process in
`ado` remains the same:

1. **Choose Your Tools:** Select the `experiments` you want to use
2. **Define Your Space:** Describe points you want to explore with the
   experiments in a `discoveryspace` YAML file
3. **Configure Your Strategy:** Create an `explore operation` YAML that defines
   the search strategy
4. **Execute:** Run the operation locally or on a
   [remote cluster](../getting-started/remote_run.md) with
   `ado create operation ...`
5. **Analyze:** Examine data with `ado show measurements`, run analysis
   `operators` to get deeper insights, then refine and explore further

See [Core Concepts](../core-concepts/concepts.md) for a full explanation and our
[Examples](../examples/examples.md) for end-to-end case studies.

---

## When to Build New Tools

Eventually, your research will require something `ado` doesn't have out of the
box, like a benchmark for a custom system or a novel search algorithm.

This is where you transition from a researcher to a **plugin developer**. If you
chose **Option B (The AI-Assisted Path)**, this transition is often seamless. In
fact, when your coding agent discovers that a required tool is missing to answer
your question, it will typically just start scaffolding the new plugin code for
you right there in your workspace.

Because your environment is already fully prepped for development, there is no
setup change required. To understand how `ado` registers and loads the new tools
that you (or your agent) are building, head over to the
[Developing ado Plugins](plugin-developers.md) guide.
