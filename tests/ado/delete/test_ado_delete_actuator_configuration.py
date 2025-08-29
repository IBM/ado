# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pathlib

from typer.testing import CliRunner

from orchestrator.cli.core.cli import app as ado


def test_delete_actuator_configuration_no_related(
    tmp_path: pathlib.Path,
    valid_ado_project_context,
    create_active_ado_context,
    ml_multi_cloud_correct_actuatorconfiguration,
):
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "delete",
            "actuatorconfiguration",
            ml_multi_cloud_correct_actuatorconfiguration.identifier,
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Success!" in result.output.strip()


def test_delete_actuator_configuration_with_related_resource(
    tmp_path: pathlib.Path,
    valid_ado_project_context,
    create_active_ado_context,
    sql_store,
    ml_multi_cloud_correct_actuatorconfiguration,
    ml_multi_cloud_space,
):
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    sql_store.addRelationship(
        ml_multi_cloud_correct_actuatorconfiguration.identifier,
        ml_multi_cloud_space.uri,
    )
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "delete",
            "actuatorconfiguration",
            ml_multi_cloud_correct_actuatorconfiguration.identifier,
        ],
    )
    assert result.exit_code == 1, result.output
    assert (
        f"ERROR:  Cannot delete actuatorconfiguration {ml_multi_cloud_correct_actuatorconfiguration.identifier} "
        "as it has children resources:\n\n"
        "            IDENTIFIER            TYPE\n"
        f"0  {ml_multi_cloud_space.uri}  discoveryspace\n\n"
        "HINT:   You must delete each of them them first."
    ) in result.output.strip(), result.output


def test_delete_nonexistent_actuator_configuration(
    tmp_path: pathlib.Path,
    valid_ado_project_context,
    create_active_ado_context,
):
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "delete",
            "actuatorconfiguration",
            "does-not-exist",
        ],
    )
    assert result.exit_code == 1, result.output
    assert (
        "ERROR:  The database does not contain a resource with id does-not-exist and kind actuatorconfiguration."
    ) in result.output.strip()
