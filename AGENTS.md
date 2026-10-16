---
description:
  General development guidelines for ado - code style, testing, quality checks,
  and commit standards
---

# General Development Guidelines for ado

These guidelines apply to all code development in the ado codebase.

## Project Structure

- **ado**: main Python package
  - **api**: REST API and Ray Serve deployment
  - **schema**: pydantic models for properties, entities, experiments, and
    measurement results
  - **core**: pydantic models and associated code for the core resource types
    managed by ado:
    - discoveryspace
    - operation
    - samplestore
    - datacontainer
    - actuatorconfiguration
    - document
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
- **docs**: mkdocs website and documentation
- **examples**: examples of using ado

### Structure Guidelines

- Place new code in the most specific existing subpackage.
- Do not create new top-level packages unless explicitly instructed.

---

## Development Guidelines

### Test-Driven Development

- For changes to existing code: first search for tests that call this code and
  update them so the new behaviour is tested.
- For new functionality: write tests first.
- Run pytest: confirm tests fail.
- Implement the code to be tested.
- Run pytest: check tests pass.
- Iterate until tests pass.

### Writing Code

#### General Conventions

- Use PEP8 naming conventions for new code.
- **Exception**: use camelCase for fields of pydantic models.
- Do not modify existing names unless explicitly asked, even if they do not
  follow PEP8.
- Use type annotations on all functions and methods, including return types.
- Add docstrings to all functions and methods. Use Google style for docstrings.

#### Pydantic

- Use the pydantic annotated form for pydantic fields (see
  `ado/schema/entity.py`). Assign the default value to the Annotated
  variable, not inside pydantic.Field. For any mutable default values such as
  dictionaries, lists, tuples, or sets, use `default_factory` inside
  pydantic.Field and do not assign a default to the Annotation.
- Use discriminated unions when a type is a union (see `ExperimentType` in
  `ado/schema/experiment.py`).
- Use the `Defaultable` type from `ado/utilities/pydantic` for pydantic
  fields that:
  - accept `None`, but
  - are always defaulted to a different type.

#### Imports and Serialization

- Use absolute imports within the repository unless the file already uses
  relative imports.
- Use `ado.utilities.output.pydantic_model_as_yaml` for serializing
  pydantic models to YAML.

#### Linting

- After making changes, run `uv run pre-commit run -a` and fix any issues
  it reports.

### Writing Documentation and Comments

- Describe only what a module, function, or section does. Do not describe what
  it does not do, what it is not responsible for, or what is out of its scope.
- Describe functions in terms of their own inputs, outputs, and immediate
  behavior. Do not reference how other functions use them or explain their
  existence in terms of another component's needs.
- Do not add comments that narrate the TDD process, such as "written before
  implementation" or "this test is expected to fail". Tests are code; write
  them without process annotations.

### Writing Tests

- Check for existing fixtures before creating new ones:
  - `tests/fixtures/`
- Do not mock by default; prefer integration tests.
  - Search for existing fixtures that provide the same functionality.
  - If you really feel a mock is correct, confirm with the user before
    implementing.
- Test the full lifecycle for pydantic models:
  create → dump → create from dump.

### Writing Agent Skills

- Be brief and to the point.
- Avoid ambiguous statements.
- Skills should be instructions on how to perform a specific task.
- Avoid duplication — before writing, check the following sources and link
  existing relevant data:
  - all skills under `.agents/skills/`
  - the examples under `examples/`
  - the documentation under `docs/`
- After writing a new skill:
  - review if any information is more appropriate in an existing skill or
    `AGENTS.md`; if so, move it there.
  - verify all file and directory paths referenced exist in the repo.
  - check each section is within the scope declared in the skill's description
    field.
- When creating YAML or code examples, prefer:
  - using an external file
  - linking it or including its contents in SKILL.md
  - writing tests for such files
- Ensure the metadata of the skill is sufficient so it triggers when likely to
  be required.

---

## Setup

All development tools (ruff, pytest, etc.) are available in the project's
**uv-managed virtual environment**. Do not install tools globally. Use
`uv run TOOLNAME` to execute tools. **Exception: for linting and formatting,
always use `uv run pre-commit run -a`** rather than invoking individual tools
directly.

The project has a top-level virtual environment managed by uv. Plugins and
examples within the repo are also uv managed and may have their own venvs.
When installing packages or executing code with uv (including pip), always
run from the top level of this repo to avoid accidentally using a local
plugin or example venv.

Ensure the virtual environment is set up before running tests:

```sh
uv sync --reinstall --group test --group dev
```

---

## Running Tests

### Running Code Tests

- Each subpackage has a corresponding test directory under `tests/`, for
  example:
  - `tests/schema/`
  - `tests/core/`
  - `tests/actuators/`
  - `tests/operators/`
  - `tests/metastore/`
  - `tests/cli/`
  - `tests/ado/`
  - `tests/samplestore/`
  - `tests/utilities/`
  - `tests/resources/`
- Test files are often named after the **class or concept** being tested.
  For example, `MeasurementResult` (defined in `result.py`) is tested in
  `test_measurement_result.py`. When changing a class, look for a test file
  whose name matches the class name before grepping.
- To find all tests relevant to a change, search by the name of each modified
  function, method, or field.
- As a final validation step after a change, run tests for all impacted
  subpackages.
- All tests must pass before submitting changes.
- Run tests in parallel (pytest-xdist) for quicker execution:

  ```sh
  uv run pytest -n auto tests/
  ```

### Testing YAML Resources

Test any new or modified ado resource YAML using:

```sh
uv run ado create RESOURCETYPE -f FILE --dry-run
```

### Testing ado CLI Commands

- Confirm all ado CLI commands and options written in documentation are
  correct:

  ```sh
  uv run ado [COMMAND] --help
  uv run ado [COMMAND] [SUBCOMMAND1] ... --help
  ```

- Leverage the `--use-latest` ado CLI argument when writing documentation if
  an `ado create` or `ado show` command requires the identifier of a
  previously created resource.

---

## Links

- For plugin development, see
  [plugin-development](.agents/skills/plugin-development/SKILL.md)
- For formulating problems with ado, see
  [define-experiment-campaign](.agents/skills/define-experiment-campaign/SKILL.md)
- For using the ado CLI, see
  [using-ado-cli](.agents/skills/using-ado-cli/SKILL.md)
- For creating resource YAML files, see
  [resource-yaml-creation](.agents/skills/resource-yaml-creation/SKILL.md)
- For querying catalog and measurement data, see
  [query-ado-data](.agents/skills/query-ado-data/SKILL.md)
