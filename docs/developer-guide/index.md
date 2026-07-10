# Developer Guide

This guide is for contributors who want to extend `ado` with new experiments or
search strategies, or work on the core framework itself.

## Choose your path

<!-- markdownlint-disable line-length -->

|                 | **Path A — Plugin Development**                                                                                          | **Path B — Core Development**                                                                                                      |
| :-------------- | :----------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------- |
| **Best for**    | Adding new experiments (actuators, custom experiments) or search strategies (operators) without touching the core.       | Contributing to `ado`'s schema, CLI, metastore, execution engine, or test suite.                                                   |
| **Setup**       | `pip install ado-core` for out-of-tree, or `uv sync` for in-tree development.                                            | Clone the repo and run `uv sync --group test --reinstall`.                                                                         |
| **Key benefit** | No core changes needed — plugins register themselves at install time and are immediately available to the CLI and tools. | Full access to the framework internals, test suite, and pre-commit hooks that enforce code style and conventional commit messages. |

<!-- markdownlint-enable line-length -->

=== "Path A — Plugin Development"

    A plugin adds one of two things:

    - An **experiment**: An [actuator](../concepts/actuators.md) or a
      [custom experiment](./creating-custom-experiments.md) that measures or
      evaluates a system (e.g., a benchmark, a simulation).
    - An **analysis tool**: An [operator](../user-guide/operators/index.md)
      that decides what to measure next or post-processes results (e.g., a
      search strategy, an optimiser).

    Use these guides as your primary technical reference:

    - [Creating Custom Experiments](./creating-custom-experiments.md)
    - [Creating Actuator Classes](./creating-actuator-classes.md)
    - [Creating Operators](./creating-operators.md)

    See [Developing ado](developing.md) for full environment setup instructions.

=== "Path B — Core Development"

    The quick start to get you running:

    ```shell
    git clone https://github.com/IBM/ado.git
    cd ado
    uv sync --group test --reinstall
    source .venv/bin/activate
    ```

    Install pre-commit hooks right after setup:

    ```shell
    pre-commit install
    ```

    This automates checks for formatting, secrets, headers, and conventional
    commit messages.

    Before submitting a pull request, ensure your changes meet this checklist:

    - **Code Style:** Follow **PEP8** naming, include all **type annotations**,
      and use **Google-style docstrings**.
    - **Formatting & Linting:** Code must be formatted with **ruff format** and
      pass **ruff check**.
    - **Testing:** All new features or fixes must include or update **tests**.
      We prefer an integration-first, TDD-style workflow.
    - **Commit Hooks:** All changes must pass our **pre-commit** hooks.

    See [Developing ado](./developing.md) and [Contributing](./contributing.md)
    for full details.

## Agent-assisted development

Both paths benefit from opening the `ado` repository in an agent-enabled editor.
The agent will automatically find all necessary rules and skills in `.cursor/`.

- **In-tree:** Open the `ado` repository root — the agent loads everything
  automatically.
- **Out-of-tree:** Provide both your plugin directory and the cloned `ado`
  directory as context to your agent.

For core contributors, you can ask the agent to review your work before creating
a pull request using the template in
[`.cursor/commands/review.md`](https://github.com/IBM/ado/blob/main/.cursor/commands/review.md).
