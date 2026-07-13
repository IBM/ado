# Getting Started

This is the **User Guide** — for **researchers and benchmarkers** who want to
run experiments, explore parameter spaces, and analyse results without writing
bespoke data management scripts from scratch. If you want to extend `ado` with
new experiments or search strategies, the
[Developer Guide](../developer-guide/index.md) is your starting point.

!!! tip "New to ado?"

    Before diving in, familiarise yourself with the
    [key concepts](../concepts/index.md) so the YAML examples and CLI commands
    throughout these examples make immediate sense.

## The core research loop with ado

`ado` organises research around a repeatable five-step process, regardless of
how you interact with it:

1. **Choose your experiments** — select what you want to run
2. **Define your space** — describe the parameter space you want to explore
3. **Configure your strategy** — choose how to search or sample that space
4. **Execute** — run the operation locally or on a remote cluster
5. **Analyse** — inspect results, then refine and continue

## How do you want to use ado?

The loop above applies to both paths. The difference is how you drive it: Path A
is a lightweight PyPI install you control entirely through the CLI. Path B clones
the repo and adds built-in agent skills so an AI coding assistant can handle the
steps for you in plain language.

<!-- markdownlint-disable line-length -->

|                 | **Path A — Direct CLI**                                                              | **Path B — AI-assisted**                                                                                       |
| :-------------- | :----------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------- |
| **Best for**    | Quick starts, existing scripts, and experienced CLI users who prefer manual control. | Exploratory research, complex parameter spaces, and iterative analysis where an AI can handle the boilerplate. |
| **Key benefit** | Lightweight — integrates into any standard Python workflow.                          | Unlocks built-in **agent skills** that automate complex research tasks.                                        |

<!-- markdownlint-enable line-length -->

## Installing

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

---

## Next steps

<!-- markdownlint-disable no-inline-html -->
<div class="grid cards" markdown>

- :octicons-beaker-24:{ .lg .middle } **See ado in action**

    ---

    Walk through end-to-end examples that cover common research workflows.

    [Choose an example :octicons-arrow-right-24:](examples/choose-an-example.md)

- :octicons-code-24:{ .lg .middle } **Extend ado**

    ---

    Add custom experiments, benchmarks, or search strategies via the plugin model.

    [Developer Guide :octicons-arrow-right-24:](../developer-guide/index.md)

</div>
<!-- markdownlint-enable no-inline-html -->
