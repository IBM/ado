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


def test_describe_vllm_bench_deployment_experiment() -> None:
    runner = CliRunner()
    result = runner.invoke(ado, ["describe", "experiment", "vllm-bench-deployment"])
    assert result.exit_code == 0
    assert "vllm-bench-deployment" in result.output


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


def test_describe_document(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    sql_store: SQLStore,
) -> None:
    """Describe renders markdown with rich and includes resource metadata."""
    from ado.core.document.config import DocumentConfiguration, RelatedResource
    from ado.core.document.resource import DocumentResource

    config = DocumentConfiguration(
        content="# Operation report\n\nExample body for describe.",
        relatedResources=[
            RelatedResource(id="operation-test-12345678", role="parent"),
        ],
        metadata={"name": "Describe test report"},
    )
    resource = DocumentResource(config=config)
    sql_store.addResource(resource)

    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "describe",
            "document",
            resource.identifier,
        ],
    )
    assert result.exit_code == 0, result.output
    assert resource.identifier in result.output
    assert "Describe test report" in result.output
    assert "operation-test-12345678 (parent)" in result.output
    assert "Operation report" in result.output
    assert "# Operation report" not in result.output
    assert "Example body for describe" in result.output


def test_describe_document_html(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    sql_store: SQLStore,
) -> None:
    """Describe prints HTML document content without opening a browser."""
    from ado.core.document.config import DocumentConfiguration
    from ado.core.document.resource import DocumentResource

    html_body = (
        "<html><body><h1>HTML report</h1><p>Opened via describe.</p></body></html>"
    )
    config = DocumentConfiguration(
        content=html_body,
        contentType="html",
        metadata={"name": "HTML describe test"},
    )
    resource = DocumentResource(config=config)
    sql_store.addResource(resource)

    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "describe",
            "document",
            resource.identifier,
        ],
    )
    assert result.exit_code == 0, result.output
    assert resource.identifier in result.output
    assert "HTML describe test" in result.output
    assert html_body in result.output
