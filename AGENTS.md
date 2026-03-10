---
description: General development guidelines for ado - code style, testing, quality checks, and commit standards
---

# General Development Guidelines for ado

These guidelines apply to all code development in the ado codebase.

## Code Structure

- **orchestrator**: main Python package
  - **schema**: pydantic models for properties, entities, experiments,
    and measurement results
  - **core**: pydantic models and associated code for the core resource
    types managed by ado:
    - discoveryspace
    - operation
    - samplestore
    - datacontainer
    - actuatorconfiguration
  - **modules/actuators**: defines actuators, custom experiments, and their
    associated management code (plugins, registry)
  - **modules/operators**: defines operators and their associated management
    code (plugins, collections, orchestration)
  - **utilities**: common utilities
  - **cli**: ado CLI
  - **metastore**: code defining and interacting with the metastore, which
    stores core resource types
- **tests**: unit and integration tests (pytest)
- **plugins**: actuator, operator, and custom_experiment plugins
- **website**: mkdocs website and documentation
- **examples**: examples of using ado

### Structure Guidelines

- Place new code in the most specific existing subpackage.
- Do not create new top-level packages unless explicitly instructed.

---

## Code Style

- Use PEP8 naming conventions for new code.
- **Exception**: use camelCase for fields of pydantic models.
- Do not modify existing names unless explicitly asked, even if they do not
  follow PEP8.
- Use type annotations on all functions and methods, including return types.
- Add docstrings to all functions and methods.
- Use the pydantic annotated form for pydantic fields (see
  `orchestrator/schema/entity.py`). Assign the default value to the Annotated
  variable, not inside pydantic.Field. For any mutable default values such as
  dictionaries, lists, tuples, or sets, use `default_factory` inside
  pydantic.Field and do not assign a default to the Annotation.
- Use discriminated unions when a type is a union
  (see `ExperimentType` in `orchestrator/schema/experiment.py`).
- Use the `Defaultable` type from `orchestrator/utilities/pydantic` for
  pydantic fields that:
  - accept `None`, but
  - are always defaulted to a different type.
- Use absolute imports within the repository unless the file already uses
  relative imports.
- Use `orchestrator.utilities.output.pydantic_model_as_yaml` for serializing
  pydantic models to YAML.
- Use Google style for docstrings.

---

## Developer Tools

- All development tools (ruff, black, pytest, etc.) are available in the
  project's **uv-managed virtual environment**.
- Do not install tools globally.
- Use the following pattern to execute tools:

    uv run TOOLNAME

---

## Code Development

Use Test Driven Development

- Write tests first
    - Do not create mock implementations
- Run pytest: confirm tests fail
- Implement the code to be tested
- Run pytest: check tests
- Iterate until tests pass

### Writing Tests

- Check for existing fixtures before creating new ones:
  - `tests/fixtures/`
- Do not mock by default; prefer integration tests.
- Test the full lifecycle for pydantic models:
    create → dump → create from dump

### Code Linting

- Linting must be run after any code changes and must pass before running tests.
- Run **black** after changes:

    uv run black $DIR

- Run **ruff** after changes:

    uv run ruff check --fix $DIR

- Fix any issues reported by ruff that it could not fix automatically.
- Run black and ruff at directory level for efficiency (e.g. `orchestrator/`,
  `plugins/`, `tests/`).
- Run the mkdocs linter on markdown files (*.md) that have added or modified:

    uv run markdownlint-cli2 NEW_OR_CHANGED_MARKDOWN_FILE --fix

- Run if YAML changed or added

    pre-commit run yamlfmt

- Run if TOML changed or added

    uv run tombi fmt

---

## Code Installation & Execution

The project has a top-level virtual environment managed by uv.
Plugins and examples within the repo are also uv managed and may have venvs.

When installing packages, install into the top-level virtual environment.
When executing code with uv, including pip, execute
from the top level of this repo.
This is avoid accidentally using the local uv environments of plugins and examples
within the repo.

---

## Testing

### Code Testing

- Each subpackage has a corresponding test directory under `tests/`, for example:
  - `schema/`
  - `core/`
  - `actuators/`
  - `operators/`
  - `metastore/`
  - `cli/`
- As a final validation step, run tests for all impacted subpackages.
- All tests must pass before submitting changes.
- Ensure the virtual environment is correctly set up before running tests:

    uv sync --reinstall --group test --group dev

- Run tests in parallel (pytest-xdist) for quicker execution e.g.

   uv run pytest -n auto tests/

### YAML Testing

- Test any new or modified ado resource YAML using:

    uv run ado create RESOURCETYPE -f FILE --dry-run

### ado CLI command-line construction and testing

- Confirm all ado CLI commands and options written in documentation are correct

   uv run ado [COMMAND] --help
   uv run ado [COMMAND] [SUBCOMMAND1] ... --help

- Leverage the --use-latest ado CLI command arg when writing documentation if
an "ado create" or "ado show" command requires the identifier of a previously
created resource

---

## Links

- For plugin development, see [plugin-development.mdc](.cursor/rules/plugin-development.mdc)
- For formulating problems with ado, see [formulate problems for ado](.cursor/skills/formulate-discovery-problem/)
- For using the ado CLI, see [using the ado CLI](.cursor/skills/using-ado-cli/)
