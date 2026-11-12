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
- Give variables, functions, and methods clear, self-explanatory names that
  describe what they hold or do. Avoid single-letter or abbreviated names
  except for conventional loop indices.
- Use type annotations on all functions and methods, including return types.
- Add docstrings to all functions and methods. Use Google style for docstrings.

#### Pydantic

- Use the pydantic annotated form for pydantic fields (see
  `ado/schema/entity.py`). Assign the default value to the Annotated
  variable, not inside pydantic.Field. For any mutable default values such as
  dictionaries, lists, tuples, or sets, use `default_factory` inside
  pydantic.Field and do not assign a default to the Annotation.

  ```python
  name: Annotated[str, pydantic.Field(description="A name")] = "unk"
  results: Annotated[
      list[Result],
      pydantic.Field(default_factory=list, description="A list of results"),
  ]
  ```

- Use discriminated unions when a type is a union (see `ExperimentType` in
  `ado/schema/experiment.py`).
- Use the `Defaultable` type from `ado/utilities/pydantic` for pydantic
  fields that:
  - accept `None`, but
  - are always defaulted to a different type.

  ```python
  identifier: Annotated[
      Defaultable[str],
      pydantic.Field(default_factory=lambda: f"document-{uuid.uuid4().hex[:8]}"),
  ]
  ```

#### Helpers

- Do not introduce private helpers (leading underscore) that are one-line
  wrappers, renames of an existing call, or called from a single place when
  inlining reads as clearly. Avoid many small private helpers in one module.
- Add a private helper when it is reused, or when it isolates a non-trivial
  step whose name makes the caller easier to read.

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
- Run the command as-is and read its full output. Do not pipe it through
  `tail`, `head`, `grep`, or similar: truncating the output hides failing hooks
  and the pipe masks the command's exit status.

### Writing Documentation and Comments

- Describe only what a module, function, or section does. Do not describe what
  it does not do, what it is not responsible for, or what is out of its scope.
- Describe functions in terms of their own inputs, outputs, and immediate
  behavior. Do not reference how other functions use them or explain their
  existence in terms of another component's needs.
- Do not add comments that narrate the TDD process, such as "written before
  implementation" or "this test is expected to fail". Tests are code; write
  them without process annotations.
- Do not leak the conversation, prompt, or iteration history into code
  comments, pull request descriptions, or other maintainer-facing prose. Write
  for readers who have not seen the conversation.

### Writing Tests

- Check for existing fixtures before creating new ones:
  - `tests/fixtures/`
  - `conftest.py` files, starting with `tests/conftest.py` and any in the
    test directory you are working in
- Do not mock by default; prefer integration tests.
  - Search for existing fixtures that provide the same functionality.
  - If you really feel a mock is correct, confirm with the user before
    implementing.
- Test the full lifecycle for pydantic models:
  create → dump → create from dump.
- Write tests that can fail for a meaningful reason. Do not assert a value
  that was just set, a constant, or a default; do not assert only that a call
  does not raise; do not assert only that a mock was called; do not copy the
  implementation's logic into the test.
- Before adding a test, search `tests/` for existing tests of the same
  function, class, or code path. Extend an existing test instead of adding one
  with the same assertions.
- Test behaviour whose regression would matter: business logic, error handling
  a caller relies on, and edge cases the code explicitly handles. Do not test
  trivial code, pass-through wrappers, or framework behaviour, and do not
  enumerate every input combination.

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
- The Python versions and dependency modes tested by `tox` are defined in
  [tox.ini](tox.ini).
- Run a single test file or test while iterating:

  ```sh
  uv run pytest tests/PATH/test_FILE.py::test_NAME
  ```

- Run tests in parallel (pytest-xdist) for quicker execution. Use
  `--dist loadgroup` so tests marked with `xdist_group` run on the same worker:

  ```sh
  uv run pytest -n auto --dist loadgroup -rFEx tests/
  ```

- Run the test command as-is and read its full output. Do not pipe it through
  `tail`, `head`, `grep`, or similar: truncating the output hides failing tests
  and the pipe masks the command's exit status.

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

## Commits and Pull Requests

- Sign off every commit with `git commit -s` (Developer Certificate of Origin).
- Write commit messages and PR titles in the conventional commits format
  `<type>[optional scope]: <description>`, including the scope. The allowed
  types and scopes are listed in
  [CONTRIBUTING.md](CONTRIBUTING.md#commit-and-pr-title-guidelines).

---

## Security

- Report vulnerabilities as described in [SECURITY.md](SECURITY.md).

---

## Links

- For contribution requirements, see [CONTRIBUTING.md](CONTRIBUTING.md)
- For setting up a development environment, see
  [DEVELOPING.md](DEVELOPING.md)
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
- For writing agent skills, see
  [.agents/skills/AGENTS.md](.agents/skills/AGENTS.md)
