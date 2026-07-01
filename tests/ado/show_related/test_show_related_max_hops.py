# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for the --max-hops flag in the ado show related command."""

import pathlib
from collections.abc import Callable

from typer.testing import CliRunner

from orchestrator.cli.core.cli import app as ado
from orchestrator.core import (
    DiscoverySpaceResource,
    OperationResource,
    SampleStoreResource,
)
from orchestrator.metastore.project import ProjectContext
from orchestrator.metastore.sql.statements import _MAX_HIERARCHY_HOPS
from orchestrator.metastore.sqlstore import SQLStore
from tests.conftest import requires_sqlite_3_38


@requires_sqlite_3_38
def test_show_related_max_hops_default_traverses_full_hierarchy(
    tmp_path: pathlib.Path,
    sql_store_with_resources_preloaded: SQLStore,
    valid_ado_project_context: ProjectContext,
    discovery_space_resource: DiscoverySpaceResource,
    operation_resource: OperationResource,
    sample_store_resource: SampleStoreResource,
    create_active_ado_context: Callable,
) -> None:
    """Without --max-hops the full hierarchy is returned (samplestore visible from operation)."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner,
        path=tmp_path,
        project_context=valid_ado_project_context,
    )

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            str(tmp_path),
            "show",
            "related",
            "operation",
            operation_resource.identifier,
        ],
    )

    assert result.exit_code == 0
    assert discovery_space_resource.identifier in result.output
    assert sample_store_resource.identifier in result.output


@requires_sqlite_3_38
def test_show_related_max_hops_1_excludes_grandparent(
    tmp_path: pathlib.Path,
    sql_store_with_resources_preloaded: SQLStore,
    valid_ado_project_context: ProjectContext,
    discovery_space_resource: DiscoverySpaceResource,
    operation_resource: OperationResource,
    sample_store_resource: SampleStoreResource,
    create_active_ado_context: Callable,
) -> None:
    """--max-hops 1 from an operation returns only the direct parent (discoveryspace)."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner,
        path=tmp_path,
        project_context=valid_ado_project_context,
    )

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            str(tmp_path),
            "show",
            "related",
            "operation",
            operation_resource.identifier,
            "--max-hops",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert discovery_space_resource.identifier in result.output
    # samplestore is 2 hops away; must be absent
    assert sample_store_resource.identifier not in result.output


@requires_sqlite_3_38
def test_show_related_max_hops_2_includes_grandparent(
    tmp_path: pathlib.Path,
    sql_store_with_resources_preloaded: SQLStore,
    valid_ado_project_context: ProjectContext,
    discovery_space_resource: DiscoverySpaceResource,
    operation_resource: OperationResource,
    sample_store_resource: SampleStoreResource,
    create_active_ado_context: Callable,
) -> None:
    """--max-hops 2 from an operation returns both discoveryspace and samplestore."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner,
        path=tmp_path,
        project_context=valid_ado_project_context,
    )

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            str(tmp_path),
            "show",
            "related",
            "operation",
            operation_resource.identifier,
            "--max-hops",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert discovery_space_resource.identifier in result.output
    assert sample_store_resource.identifier in result.output


@requires_sqlite_3_38
def test_show_related_max_hops_zero_is_rejected(
    tmp_path: pathlib.Path,
    sql_store_with_resources_preloaded: SQLStore,
    valid_ado_project_context: ProjectContext,
    operation_resource: OperationResource,
    create_active_ado_context: Callable,
) -> None:
    """--max-hops 0 must be rejected (minimum valid value is 1)."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner,
        path=tmp_path,
        project_context=valid_ado_project_context,
    )

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            str(tmp_path),
            "show",
            "related",
            "operation",
            operation_resource.identifier,
            "--max-hops",
            "0",
        ],
    )

    assert result.exit_code != 0


@requires_sqlite_3_38
def test_show_related_max_hops_above_maximum_is_rejected(
    tmp_path: pathlib.Path,
    sql_store_with_resources_preloaded: SQLStore,
    valid_ado_project_context: ProjectContext,
    operation_resource: OperationResource,
    create_active_ado_context: Callable,
) -> None:
    """--max-hops values above _MAX_HIERARCHY_HOPS must be rejected."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner,
        path=tmp_path,
        project_context=valid_ado_project_context,
    )

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            str(tmp_path),
            "show",
            "related",
            "operation",
            operation_resource.identifier,
            "--max-hops",
            str(_MAX_HIERARCHY_HOPS + 1),
        ],
    )

    assert result.exit_code != 0
