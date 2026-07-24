# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import os
import pathlib
from collections.abc import Callable

from testcontainers.mysql import MySqlContainer
from typer.testing import CliRunner

from ado.cli.core.cli import app as ado
from ado.core import DataContainerResource
from ado.core.datacontainer.resource import DataContainer
from ado.core.discoveryspace.space import DiscoverySpace
from ado.metastore.project import ProjectContext
from ado.metastore.sqlstore import SQLStore
from tests.conftest import requires_sqlite_3_38


def test_describe_nonexistent_space(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    nonexistent_space_id = "i-do-not-exist"
    result = runner.invoke(
        ado,
        ["describe", "space", nonexistent_space_id],
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
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    pfas_space: DiscoverySpace,
) -> None:
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    result = runner.invoke(ado, ["describe", "space", pfas_space.uri])
    assert result.exit_code == 0
    # AP: TODO: find something actually meaningful to test


def test_describe_peptide_mineralization_experiment() -> None:
    runner = CliRunner()
    result = runner.invoke(ado, ["describe", "experiment", "peptide_mineralization"])
    assert result.exit_code == 0
    assert ("Identifier: robotic_lab.peptide_mineralization") in result.output

    assert "Measures adsorption of peptide lanthanide combinations" in result.output


def test_describe_calculate_density_experiment() -> None:
    runner = CliRunner()
    result = runner.invoke(ado, ["describe", "experiment", "calculate_density"])
    assert result.exit_code == 0
    assert "calculate_density" in result.output


def test_describe_vllm_test_deployment_experiment() -> None:
    runner = CliRunner()
    result = runner.invoke(ado, ["describe", "experiment", "test-deployment-v1"])
    assert result.exit_code == 0
    assert "test-deployment-v1" in result.output


@requires_sqlite_3_38
def test_describe_datacontainer_with_use_latest(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    sql_store: SQLStore,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    """Test that ado describe datacontainer --use-latest resolves the latest datacontainer."""
    from datetime import datetime, timezone

    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    # Create two datacontainer resources with different identifiers and timestamps
    dc_1 = DataContainerResource(config=DataContainer(data={"key": "value1"}))
    dc_1.identifier = "dc-test-older"
    dc_1.created = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    sql_store.addResource(dc_1)

    dc_2 = DataContainerResource(config=DataContainer(data={"key": "value2"}))
    dc_2.identifier = "dc-test-latest"
    dc_2.created = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    sql_store.addResource(dc_2)

    result = runner.invoke(ado, ["describe", "datacontainer", "--use-latest"])
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert dc_2.identifier in result.output


def test_describe_use_latest_rejected_for_experiment() -> None:
    """Test that ado describe experiment --use-latest exits with code 1."""
    runner = CliRunner()
    result = runner.invoke(ado, ["describe", "experiment", "--use-latest"])
    assert result.exit_code == 1
