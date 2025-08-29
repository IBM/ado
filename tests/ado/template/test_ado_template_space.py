# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pathlib

from typer.testing import CliRunner

from orchestrator.cli.core.cli import app as ado


def test_template_space(tmp_path: pathlib.Path, random_identifier):
    runner = CliRunner()
    file_name = tmp_path / random_identifier()
    result = runner.invoke(
        ado,
        ["--override-ado-app-dir", tmp_path, "template", "space", "-o", file_name],
    )
    assert result.exit_code == 0
    assert f"Success! File saved as {file_name}" in result.output
