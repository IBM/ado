# Developer Guide

This is the **Developer Guide** — for **contributors** who want to extend `ado`
with new experiments or search strategies, or work on the core framework itself.
If you want to run experiments and analyse results without writing new plugins,
the [User Guide](../user-guide/getting-started.md) is your starting point.

!!! question "New to ado?"

    Before diving in, familiarise yourself with the
    [key concepts](../concepts/index.md) so the plugin model and extension
    points make immediate sense.

## The core development loop with ado

All plugin and core development in `ado` follows the same workflow, regardless
of what you are building:

1. **Choose your extension point** — actuator, custom experiment, or operator
2. **Set up your environment** — install dependencies and register your plugin
3. **Implement and test** — write code, run the linter, iterate with TDD
4. **Validate** — dry-run YAML files, confirm CLI registration
5. **Contribute** — open a pull request against the main repo

## How do you want to contribute?

The loop above applies to both paths. The difference is scope: Path A adds
new experiments or strategies as self-contained plugins. Path B changes the
core framework itself.

<!-- markdownlint-disable line-length -->

|                 | **Path A — Plugin Development**                                                                                          | **Path B — Core Development**                                                                                                      |
| :-------------- | :----------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------- |
| **Best for**    | Adding new experiments (actuators, custom experiments) or search strategies (operators) without touching the core.       | Contributing to `ado`'s schema, CLI, metastore, execution engine, or test suite.                                                   |
| **Setup**       | `pip install ado-core` for out-of-tree, or `uv sync` for in-tree development.                                            | Clone the repo and run `uv sync --group test --reinstall`.                                                                         |
| **Key benefit** | No core changes needed — plugins register themselves at install time and are immediately available to the CLI and tools. | Full access to the framework internals, test suite, and pre-commit hooks that enforce code style and conventional commit messages. |

<!-- markdownlint-enable line-length -->

## Setting up

=== "Path A — Plugin Development"

    A plugin adds one of two things:

    - An **experiment**: An [actuator](../concepts/actuators.md) or a
      [custom experiment](./creating-custom-experiments.md) that measures or
      evaluates a system (e.g., a benchmark, a simulation).
    - An **analysis tool**: An [operator](../user-guide/operators/working-with-operators.md)
      that decides what to measure next or post-processes results (e.g., a
      search strategy, an optimiser).

    Install `ado-core` from PyPI to develop an out-of-tree plugin:

    ```shell
    python -m venv ado-venv && source ado-venv/bin/activate
    pip install ado-core
    ```

    For in-tree development, clone the repo instead:

    ```shell
    git clone https://github.com/IBM/ado.git
    cd ado
    uv sync --reinstall
    source .venv/bin/activate
    ```

    Use these guides as your primary technical reference:

    - [Creating Custom Experiments](./creating-custom-experiments.md)
    - [Creating Actuator Classes](./creating-actuator-classes.md)
    - [Creating Operators](./creating-operators.md)

    See [Developing ado](developing.md) for full environment setup instructions.

=== "Path B — Core Development"

    Clone the repository and set up the full environment:

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

---

## Next steps

<!-- markdownlint-disable no-inline-html -->
<div class="grid cards" markdown>

- :octicons-plug-24:{ .lg .middle } **Build a plugin**

    ---

    Add custom experiments, benchmarks, or search strategies via the plugin
    model.

    [Creating Custom Experiments :octicons-arrow-right-24:](creating-custom-experiments.md)

- :octicons-tools-24:{ .lg .middle } **Contribute to core**

    ---

    Set up the full development environment and learn the contribution
    workflow.

    [Developing ado :octicons-arrow-right-24:](developing.md)

</div>
<!-- markdownlint-enable no-inline-html -->
