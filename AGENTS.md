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
- Never call `yaml.safe_load(...)` or `yaml.safe_dump(...)` directly — these
  always use the pure-Python path. Instead, import the C-backed loader/dumper
  with a pure-Python fallback and pass them explicitly:

  ```python
  try:
      from yaml import CSafeLoader as SafeLoader, CSafeDumper as SafeDumper
  except ImportError:  # pragma: nocover
      from yaml import SafeLoader, SafeDumper  # type: ignore[assignment]
  ```

  Then call `yaml.load(stream, Loader=SafeLoader)` and
  `yaml.dump(data, Dumper=SafeDumper)`.

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

Agent skills in `.agents/skills/` must conform to the
[Agent Skills Specification](https://agentskills.io/home) (see also
[upstream documentation](https://github.com/agentskills/agentskills/tree/main/docs/skill-creation)):

- **Directory Layout**:
  - `SKILL.md` (required): metadata frontmatter and concise instructions
    (< 500 lines / ~5,000 tokens).
  - `references/` (optional): on-demand reference documentation, secondary
    guides, and deep schemas.
  - `assets/` (optional): static resources, configuration templates, and YAML
    example manifests.
  - `scripts/` (optional): executable helper scripts.
- **Frontmatter**:
  - `name` (required): 1–64 characters, lowercase alphanumeric and hyphens
    only, matching the parent directory name exactly.
  - `description` (required): non-empty, maximum 1024 characters. Must use
    imperative phrasing ("Use when...") specifying clear activation
    conditions, intent, and domain context.
- **Progressive Disclosure**:
  - Keep `SKILL.md` focused on core workflows, decision rules, and high-value
    gotchas.
  - Move extended reference guides into `references/` and link them with
    explicit loading triggers.
  - Store reusable YAML manifests, templates, and example files in `assets/`
    and link them using relative paths (e.g. `assets/my-example.yaml`).
  - Write automated tests under `tests/` for all YAML examples and templates
    stored in `assets/`.
- **Quality & Scope**:
  - Be brief and to the point; avoid ambiguous statements.
  - Avoid duplication — check existing skills under `.agents/skills/`,
    examples under `examples/`, and documentation under `docs/`.
  - After writing or editing a skill, verify that all referenced file and
    directory paths exist.

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
  [ado-develop-plugin](.agents/skills/ado-develop-plugin/SKILL.md)
- For formulating problems with ado, see
  [ado-define-experiment-campaign](.agents/skills/ado-define-experiment-campaign/SKILL.md)
- For using the ado CLI, see
  [ado-use-cli](.agents/skills/ado-use-cli/SKILL.md)
- For creating resource YAML files, see
  [ado-create-resource-yaml](.agents/skills/ado-create-resource-yaml/SKILL.md)
- For querying catalog and measurement data, see
  [ado-query-database](.agents/skills/ado-query-database/SKILL.md)
