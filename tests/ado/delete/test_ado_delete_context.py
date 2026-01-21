# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import os
import pathlib
from collections.abc import Callable

from typer.testing import CliRunner

from orchestrator.cli.core.cli import app as ado
from orchestrator.metastore.project import ProjectContext


def test_delete_nonexistent_context(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    nonexistent_name = "i-do-not-exist"
    result = runner.invoke(
        ado, ["--override-ado-app-dir", tmp_path, "delete", "context", nonexistent_name]
    )
    assert result.exit_code == 1
    # Travis CI cannot capture output reliably
    if os.environ.get("CI", "false") != "true":
        assert (
            f"{nonexistent_name} is not in the available contexts."
            in result.output.strip()
        )
