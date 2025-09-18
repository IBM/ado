# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import os
import pathlib

from typer.testing import CliRunner

from orchestrator.cli.core.cli import app as ado


def test_describe_nonexistent_space(
    tmp_path: pathlib.Path,
    mysql_test_instance,
    valid_ado_project_context,
    create_active_ado_context,
):
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    nonexistent_space_id = "i-do-not-exist"
    result = runner.invoke(
        ado,
        ["--override-ado-app-dir", tmp_path, "describe", "space", nonexistent_space_id],
    )
    assert result.exit_code == 1
    # Travis CI cannot capture output reliably
    if os.environ.get("CI", "false") != "true":
        assert (
            f"The database does not contain a resource with id {nonexistent_space_id}"
            in result.output
        )


def test_describe_valid_space(
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

    result = runner.invoke(
        ado, ["--override-ado-app-dir", tmp_path, "describe", "space", pfas_space.uri]
    )
    assert result.exit_code == 0
    # AP: TODO: find something actually meaningful to test


def test_describe_peptide_mineralization_experiment():
    runner = CliRunner()
    result = runner.invoke(ado, ["describe", "experiment", "peptide_mineralization"])
    assert result.exit_code == 0
    assert (
        "Identifier: robotic_lab.peptide_mineralization\n\n"
        "Measures adsorption of peptide lanthanide combinations"
    ) in result.output
