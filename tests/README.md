# Testing `ado`

We provide a comprehensive set of tests to ensure the correctness of `ado-core`
and its plugins. The tests are written using the `pytest` framework.

## Prerequisites

### uv

We recommend using `uv` to manage the development environment. `uv` can be
installed following one of the installation methods provided in the
[uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

### Synchronizing the dependencies

> [!CAUTION]
>
> `uv sync` will uninstall any existing dependencies that are not listed in the
> `uv.lock` file. This ensure consistent environments.

Synchronize the development environment with the `uv.lock` file by running the
following command:

```commandline
uv sync --group test --reinstall
```

### Resource Store and Sample Store-specific requirements

Resource Store and Sample Store tests have external dependencies on SQLite
(version 3.38.0 or higher) and a container runtime compatible with
Testcontainers.

#### Checking the SQLite version

To check the SQLite version, run the following command:

```commandline
python -c 'import sqlite3; print(sqlite3.sqlite_version)'
```

The result should be `3.38.0` or higher.

#### Checking the container runtime

Testcontainers requires a container runtime that is compatible with the Docker
API. At the time of writing, the official Testcontainers website mentions
explicit support for Docker Desktop, Colima, Rancher Desktop, and Podman.

While Docker Desktop works out-of-the-box, instructions for configuring the
other runtimes are available
[on the Testcontainers website](https://java.testcontainers.org/supported_docker_environment/).

## Running the tests

> [!CAUTION]
>
> The tests **must** be run from the root directory of the `ado` repository.

There are two ways to run the tests:

1. **(RECOMMENDED)** Using one of the available `tox` environments.
   - Tox can provision an appropriate Python version if it is not already
     installed.
   - Tox will ensure the Python environment is set up with all the required
     dependencies.
   - The complete suite of tests will be run, avoiding unexpected side-effects.
2. Manually running `pytest`. This option allows running only a subset of the
   tests but requires manual setup.

### Using tox

The list of supported Tox environments can always be queried by running
`tox list`. As of the time of writing, we support testing on Python versions
3.10-3.13 on Linux and macOS using locked and non-locked dependencies.

To run the tests using Python 3.11, locked dependencies on a macOS machine, run
the following command:

```commandline
tox -re py311-locked-macos
```

### Using pytest

Before running tests with `pytest`, run `uv sync --group test --reinstall` to
synchronize the dependencies, in case they have changed since the last time the
tests were run.

To run the tests for `ado-core`, run the following command:

<!-- markdownlint-disable line-length -->

```commandline
pytest -n auto --cov=orchestrator --cov=plugins/operators --dist worksteal -rx -vv --log-level=INFO --color=yes tests/
```

<!-- markdownlint-enable line-length -->

To run other tests, we recommend checking the `commands` section of the
`tox.ini` file

## Coverage

If you want to export the coverage report as HTML for further analysis, first
export the following env-var:

```commandline
export COVERAGE_PROCESS_START=.coveragerc
```

This is required for obtaining coverage from tests involving remote ray actors.

Then either use (required `pytest-cov`)

```commandline
 pytest --cov --cov-report=html:$OUTPUT_DIR tests
```

or

```commandline
coverage run -m pytest tests
```

After the tests have finished you need to combine the results obtained from
different ray processes

```commandline
coverage combine
```

To view the coverage report as html run

```commandline
coverage html
```

This produces a directory called `htmlcov`. Open `htmlcov/index.html` to browse
the coverage. Other `coverage` options produce reports in different formats e.g.
`coverage json`. See `coverage -h` for details.
