# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import os
import pathlib
import sqlite3

import pytest
from typer.testing import CliRunner

from orchestrator.cli.core.cli import app as ado

sqlite3_version = sqlite3.sqlite_version_info


# AP: the -> and ->> syntax in SQLite is only supported from version 3.38.0
# ref: https://sqlite.org/json1.html#jptr
@pytest.mark.skipif(
    sqlite3_version < (3, 38, 0), reason="SQLite version 3.38.0 or higher is required"
)
def test_space_exists(
    tmp_path: pathlib.Path,
    mysql_test_instance,
    valid_ado_project_context,
    create_active_ado_context,
    pfas_space,
):

    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    result = runner.invoke(ado, ["--override-ado-app-dir", tmp_path, "get", "spaces"])
    assert result.exit_code == 0
    # Travis CI cannot capture output reliably
    if os.environ.get("CI", "false") != "true":
        assert pfas_space.uri in result.output
