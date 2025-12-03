# Tests

The tests assume you are running for the directory containing this directory
i.e. the top directory of the `ado` repo.

We recommend using `uv sync` to install all test requirements:

```commandline
uv sync --group test --reinstall
```

Note: This will also remove all packages not required for testing and sync
all package versions to the lockfile.
Local packages will be rebuilt.

To execute all tests run:

```commandline
pytest tests/
```

**NOTE**: Some tests require a SQLite version higher than 3.38.0. You can check
the SQLite version you have installed with:

```commandline
python -c 'import sqlite3; print(sqlite3.sqlite_version)'
```

## Setting up access to ephemeral resource store

Some tests leverage the `testcontainers` python package to create an ephemeral
resource store to run tests against. If you have docker/docker desktop running
the tests should work once you have the following environment variable set

```commandline
TESTCONTAINERS_RYUK_DISABLED=true
```

If you do not have docker you can use
[rancher desktop](https://docs.rancherdesktop.io/getting-started/installation/).

When setting up rancher desktop ensure:

- Administrative Access is enabled (required for exposing routes from test
  containers without port-forward)
- The container engine is `dockerd(moby)`

both these are set from the rancher desktop UI preferences plane (the
`Application` and `Container Engine` panes respectively).

**Note**: the tests do not stop the `testcontainer` containers. You must do this
manually via the docker cli or the rancher desktop UI.

## Coverage

If you want to run with coverage first export the following env-var

```commandline
export COVERAGE_PROCESS_START=.coveragerc
```

this is required for obtaining coverage from tests involving remote ray actors.

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
