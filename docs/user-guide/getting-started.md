# Getting Started

This is the **User Guide** — for **researchers and benchmarkers** who want to
run experiments, explore parameter spaces, and analyse results without writing
bespoke data management scripts from scratch. If you want to extend `ado` with
new experiments or search strategies, the
[Developer Guide](../developer-guide/getting-started.md) is your starting point.

!!! question "New to ado?"

    Before diving in, familiarise yourself with the
    [key concepts](../concepts/core-concepts.md) so the YAML examples and CLI commands
    throughout these examples make immediate sense.

## The research loop with ado

`ado` organises research around a repeatable five-step process:

1. **Choose your experiments** — select what you want to run
2. **Define your space** — describe the parameter space you want to explore
3. **Configure your strategy** — choose how to search or analyse that space
4. **Execute** — apply the experiments on the points in the space
5. **Analyse** — inspect results, then refine and continue

## Core concepts

The loop is possible thanks to these three building blocks (explore them all in
the [concepts](../concepts/core-concepts.md) section):

| Concept             | Role                                                                                                                                                                                        |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Discovery Space** | Defines _what_ to measure, _how_ to measure it (via Experiments, which are pluggable python functions), and _where_ to store results.                                                       |
| **Operation**       | You explore or analyse a Discovery Space using operations. You can select from different operators to perform different types of operations. Operators are also pluggable python functions. |
| **Sample Store**    | Stores the results of measurements, and enables operations to transparently reuse existing results (memoization).                                                                           |

## How do you want to use ado?

You can run this loop yourself through the CLI, or let an AI coding assistant
drive it for you using built-in agent skills.

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
    uv sync --no-default-groups
    source .venv/bin/activate
    ```

    Open the cloned `ado` folder as your workspace root in an agent-enabled IDE
    (Claude, Cursor, Bob, and others will automatically detect and load the
    built-in skills).

    See [ado and Agents](ado-and-agents.md) for a full overview of what agents
    can do with ado and example prompts to get started.

---

## Next steps

<!-- prettier-ignore-start -->
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

<!-- prettier-ignore-end -->
