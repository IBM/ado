# Getting Started

This guide is for **researchers and benchmarkers** who want to run experiments,
explore parameter spaces, and analyse results without writing bespoke data
management scripts from scratch. If you want to extend `ado` with new
experiments or search strategies, the
[Developer Guide](../developer-guide/plugin-developers.md) is your starting
point.

## Choose your workflow

`ado` works well on its own and even better alongside a coding agent. Pick the
path that matches how you prefer to work.

<!-- markdownlint-disable line-length -->

|                 | **Path A — Direct CLI**                                                              | **Path B — AI-assisted**                                                                                       |
| :-------------- | :----------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------- |
| **Best for**    | Quick starts, existing scripts, and experienced CLI users who prefer manual control. | Exploratory research, complex parameter spaces, and iterative analysis where an AI can handle the boilerplate. |
| **Setup**       | `pip install ado-core`                                                               | `git clone` the repo and run `uv sync`                                                                         |
| **Key benefit** | Lightweight — integrates into any standard Python workflow.                          | Unlocks built-in **agent skills** that automate complex research tasks.                                        |

<!-- markdownlint-enable line-length -->

=== "Path A — Direct CLI"

    Make sure you are on **Python 3.10 – 3.14** (`python --version`), and work
    inside a virtual environment to avoid dependency conflicts:

    ```shell
    python -m venv ado-venv && source ado-venv/bin/activate
    ```

    Install `ado-core` from PyPI:

    ```shell
    pip install ado-core
    ```

    You drive everything through the `ado` CLI. See the
    [CLI reference](../cli-reference/index.md) and work through the
    [examples](examples/index.md) to get up to speed.

=== "Path B — AI-assisted"

    Clone the repository and set up the full environment:

    ```shell
    git clone https://github.com/IBM/ado.git
    cd ado
    uv sync --group test
    source .venv/bin/activate
    ```

    Open the cloned `ado` folder as your workspace root in an agent-enabled IDE
    (Claude, Cursor, Bob, and others will automatically detect and load the
    built-in skills).

    With the skills loaded you can ask your agent to handle complex tasks in
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

## How ado works — the core research loop

Both paths follow the same five-step process:

1. **Choose your tools** — select the `experiments` you want to run
2. **Define your space** — describe what you want to explore in a
   `discoveryspace` YAML file
3. **Configure your strategy** — write an `operation` YAML that sets the search
   or sampling strategy
4. **Execute** — run the operation locally or on a
   [remote cluster](remote-execution.md) with `ado create operation -f operation.yaml`
5. **Analyse** — inspect results with `ado show measurements`, run analysis
   `operators`, then refine and continue

See [Concepts](../concepts/index.md) for a full explanation and
[Examples](examples/index.md) for end-to-end walkthroughs.

---

## Need something ado doesn't have out of the box?

If your research requires a custom benchmark or a novel search algorithm, you
are ready to move from *researcher* to *plugin developer*. Head to the
[Developer Guide](../developer-guide/plugin-developers.md) to learn how to add
new experiments (actuators, custom experiments) and search strategies
(operators) via `ado`'s plugin model.
