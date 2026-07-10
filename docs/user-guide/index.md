# User Guide

This guide is for **researchers and benchmarkers** who want to run experiments,
explore parameter spaces, and analyze results without getting bogged down in
custom scripting and data management.

## Choose Your Workflow: Manual Control or AI-Assisted

`ado` is powerful on its own and even more so with a coding agent. Your choice
depends on how you like to work: direct, hands-on control via a familiar CLI,
or an AI partner to accelerate complex and exploratory tasks.

<!-- markdownlint-disable line-length -->

|                 | **Option A: The Direct CLI Path**                                                    | **Option B: The AI-Assisted Path**                                                                             |
| :-------------- | :----------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------- |
| **Best For**    | Quick starts, existing scripts, and experienced CLI users who prefer manual control. | Exploratory research, complex parameter spaces, and iterative analysis where an AI can handle the boilerplate. |
| **Setup**       | `pip install ado-core` into any virtual environment.                                 | `git clone` the repository and use `uv sync` to create a comprehensive local environment.                      |
| **Key Feature** | Lightweight and integrates into any standard Python workflow.                        | Unlocks powerful, pre-built **agent skills** that automate complex research tasks.                             |

<!-- markdownlint-enable line-length -->

## The Universal `ado` Research Workflow

Whether you choose the CLI path or the agent path, the core research process in
`ado` remains the same:

1. **Choose Your Tools:** Select the `experiments` you want to use
2. **Define Your Space:** Describe points you want to explore with the
   experiments in a `discoveryspace` YAML file
3. **Configure Your Strategy:** Create an `explore operation` YAML that defines
   the search strategy
4. **Execute:** Run the operation locally or on a
   [remote cluster](remote-execution.md) with `ado create operation ...`
5. **Analyze:** Examine data with `ado show measurements`, run analysis
   `operators` to get deeper insights, then refine and explore further

See [Concepts](../concepts/index.md) for a full explanation and our
[Examples](examples/index.md) for end-to-end case studies.

## `ado` and Coding Agents

With agent skills loaded, you can ask your coding agent to handle complex tasks
in natural language:

<!-- markdownlint-disable line-length -->

| Ask Your Agent to...  | Example Command (Natural Language)                                                        | Underlying Skill              |
| :-------------------- | :---------------------------------------------------------------------------------------- | :---------------------------- |
| **Run a full study**  | "Design, run, and analyze an experiment to find the best vLLM config for max throughput." | `conduct-empirical-study`     |
| **Create YAML files** | "Formulate a discovery space for my new component."                                       | `formulate-discovery-problem` |
| **Summarize results** | "Examine the operation I just ran and tell me what it found."                             | `examining-ado-operations`    |
| **Inspect a project** | "Give me an overview of all experiments run in this project so far."                      | `examining-ado-project`       |
| **Query data**        | "Find all entities where the `lora_rank` was 8 and export their `validation_loss`."       | `query-ado-data`              |

<!-- markdownlint-enable line-length -->

To unlock the full potential of `ado` as an automated research partner, clone
the repository and work inside it. This approach gives your coding agent access
to a library of built-in **skills**:

```bash
git clone https://github.com/IBM/ado.git
cd ado
uv sync --group test
source .venv/bin/activate
```

Open the cloned `ado` folder as the workspace root in your agent-enabled IDE.
Many coding agents, including Claude, Cursor, and Bob, will automatically detect
and load the skills.

## When to Build New Tools

Eventually, your research may require something `ado` doesn't have out of the
box — a benchmark for a custom system or a novel search algorithm. This is where
you transition from a researcher to a **plugin developer**. Head over to the
[Plugin Developers](../developer-guide/plugin-developers.md) guide.
