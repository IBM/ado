# Developing ado Plugins

A plugin is how you extend `ado`'s capabilities. It's a self-contained package
that adds one of two things:

- An **experiment**: An [actuator](../concepts/actuators.md) or a
  [custom experiment](./creating-custom-experiments.md) that measures
  or evaluates a system (e.g., a benchmark, a simulation).
- An **analysis tool**: An [operator](../user-guide/operators/index.md)
  that decides what to measure next or post-processes results (e.g., a search
  strategy, an optimiser).

Writing a plugin does not require modifying the core `ado` framework. Plugins
register themselves at install time and are immediately available to the CLI and
other tools.

---

## 1. Choose Your Development Strategy

First, decide where your plugin's code will live. This choice determines your
setup process.

<!-- markdownlint-disable line-length -->

| Development Strategy        | Best For                                                                                                                                                                                    |
| :-------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **In-Tree Development**     | Getting started quickly, developing plugins tightly coupled to `ado`'s main branch, or planning to contribute the plugin upstream. This is the easiest path for agent-assisted development. |
| **Out-of-Tree Development** | Creating proprietary plugins, managing a separate release cycle and versioning, or distributing your plugin independently on PyPI.                                                          |

<!-- markdownlint-enable line-length -->

---

## 2. Set Up Your Environment

Based on your chosen strategy, follow the appropriate setup guide below.

### In-Tree Development Workflow

1. **Set up the environment:**

        git clone https://github.com/IBM/ado.git
        cd ado
        uv sync --group test
        source .venv/bin/activate

2. **Create your plugin's directory** under the appropriate `plugins/` folder
    (e.g., `plugins/actuators/my-new-plugin`).

3. **Install your plugin in editable mode.** Run this command from the
    **top-level of the `ado` repository**:

        uv pip install -e plugins/<type>/<plugin-name>

### Out-of-Tree Development Workflow

1. **Create your project** and set up a virtual environment.

2. **Add `ado-core` as a dependency** in your plugin's `pyproject.toml` file:

3. **Install your dependencies**, including `ado-core` and your own plugin in
    editable mode.

        uv pip install -e path/to/your/<plugin-name>

---

## 3. Implement Your Plugin

Now that your environment is ready, it's time to write the code. `ado` uses
**Python entry points** to discover plugins at runtime. Use these guides as your
primary technical reference for package layout, registration, and implementation
details:

- [Creating Custom Experiments](./creating-custom-experiments.md):
  Use Python functions as experiments via the `@custom_experiment` decorator.
- [Creating Actuator Classes](./creating-actuator-classes.md): Define
  full actuator classes registered via Python entry points. A complete
  [template actuator](https://github.com/IBM/ado/tree/main/plugins/actuators/example_actuator)
  is available as a reference.
- [Creating Operators](./creating-operators.md): Implement new search
  strategies and analysis tools.

---

## 4. Agent-Assisted Development

You can significantly speed up development by using a coding agent. To do this,
the agent needs context from both your plugin code and the `ado` repository,
which contains the specialized rules and skills for the framework.

- **In-Tree Setup:** Simply open the `ado` repository root in your agent-enabled
  editor. The agent will automatically find all the necessary files in
  `.cursor/`.

- **Out-of-Tree Setup:** If your plugin lives in a separate repository, you must
  provide the agent with context from both directories.
  - **For VS Code / Cursor (Multi-root):** Create a `.code-workspace` file
    listing both folders.

    { "folders": [ { "path": "/path/to/cloned/ado" }, { "path":
    "/path/to/your/plugin" } ] }

  - **For CLI-based Agents (e.g., Claude):** Use your agent's flags to provide
    both directories as context:

    my-agent --context-dir /path/to/your/plugin --context-dir
    /path/to/cloned/ado "Scaffold a new test"

---

## 5. Quality and Testing

Ensure your plugin meets the project's quality standards before use or
contribution.

1. **Verify Registration:** Check that `ado` recognizes your plugin.

        ado get actuators --details
        ado get operators

2. **Run Tests:** Place tests in a `tests/` subdirectory inside your plugin and
    run them with pytest.

        pytest path/to/your/plugin/tests/

3. **Check Formatting:** Format with **ruff format** and lint with **ruff check**
    to stay consistent with the core framework.

4. **Validate Resources:** Always use `--dry-run` to check your resource files
    before running them.

        ado create operation -f operation.yaml --dry-run

The checklist inside the
[plugin-development.mdc](https://github.com/IBM/ado/blob/main/.cursor/rules/plugin-development.mdc)
file is written for agents but serves as an excellent final reference for the
standards your plugin should meet.
