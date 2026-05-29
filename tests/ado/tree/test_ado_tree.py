# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import os
import pathlib
from collections.abc import Callable

from typer.testing import CliRunner

from orchestrator.cli.core.cli import app as ado
from orchestrator.core.discoveryspace.resource import DiscoverySpaceResource
from orchestrator.core.operation.resource import OperationResource
from orchestrator.core.resources import ADOResource
from orchestrator.core.samplestore.resource import SampleStoreResource
from orchestrator.metastore.project import ProjectContext
from orchestrator.metastore.sqlstore import SQLStore
from tests.conftest import requires_sqlite_3_38


@requires_sqlite_3_38
def test_ado_tree_help() -> None:
    runner = CliRunner()
    result = runner.invoke(ado, ["tree", "--help"])
    assert result.exit_code == 0
    assert "--all-relationships" in result.output
    assert "--names" in result.output
    assert "--sort" in result.output
    assert "--metadata" in result.output


@requires_sqlite_3_38
def test_ado_tree_renders_workflow_forest(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    create_resource_with_related_identifiers: Callable[
        [ADOResource, list[str], SQLStore], None
    ],
    random_space_resource_from_file: Callable[[str | None], DiscoverySpaceResource],
    ml_multi_cloud_operation_resource: Callable[[str | None], OperationResource],
    sql_store: SQLStore,
) -> None:
    import yaml

    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    store_id = "store-cli-tree"
    sample_store = SampleStoreResource.model_validate(
        yaml.safe_load(
            pathlib.Path(
                "tests/resources/samplestore/sample_store_07c0fa.yaml"
            ).read_text()
        )
    )
    sample_store.identifier = store_id
    sql_store.addResource(sample_store)

    space = random_space_resource_from_file(sample_store_id=store_id)
    space.config.metadata.name = "baseline space"
    create_resource_with_related_identifiers(space, [store_id])

    operation = ml_multi_cloud_operation_resource(space_id=space.identifier)
    operation.config.metadata.name = "parent operation"
    create_resource_with_related_identifiers(operation, [space.identifier])

    result = runner.invoke(
        ado,
        ["--override-ado-app-dir", tmp_path, "tree"],
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert store_id in result.output
        assert space.identifier in result.output
        assert operation.identifier in result.output
        assert "(baseline space)" not in result.output
        assert "(parent operation)" not in result.output

    named_result = runner.invoke(
        ado,
        ["--override-ado-app-dir", tmp_path, "tree", "--names"],
    )
    assert named_result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert "(baseline space)" in named_result.output
        assert "(parent operation)" in named_result.output


@requires_sqlite_3_38
def test_ado_tree_json_output(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    create_resource_with_related_identifiers: Callable[
        [ADOResource, list[str], SQLStore], None
    ],
    random_space_resource_from_file: Callable[[str | None], DiscoverySpaceResource],
    sql_store: SQLStore,
) -> None:
    import json

    import yaml

    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    store_id = "store-json-tree"
    sample_store = SampleStoreResource.model_validate(
        yaml.safe_load(
            pathlib.Path(
                "tests/resources/samplestore/sample_store_07c0fa.yaml"
            ).read_text()
        )
    )
    sample_store.identifier = store_id
    sql_store.addResource(sample_store)

    space = random_space_resource_from_file(sample_store_id=store_id)
    create_resource_with_related_identifiers(space, [store_id])

    result = runner.invoke(
        ado,
        ["--override-ado-app-dir", tmp_path, "tree", "-o", "json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload[0]["identifier"] == store_id
    assert payload[0]["children"][0]["identifier"] == space.identifier


@requires_sqlite_3_38
def test_ado_tree_excludes_actuator_configuration_by_default(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    create_resource_with_related_identifiers: Callable[
        [ADOResource, list[str], SQLStore], None
    ],
    random_space_resource_from_file: Callable[[str | None], DiscoverySpaceResource],
    ml_multi_cloud_operation_resource: Callable[[str | None], OperationResource],
    ml_multi_cloud_correct_actuatorconfiguration: ADOResource,
    sql_store: SQLStore,
) -> None:
    import yaml

    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    store_id = "store-ac-tree"
    sample_store = SampleStoreResource.model_validate(
        yaml.safe_load(
            pathlib.Path(
                "tests/resources/samplestore/sample_store_07c0fa.yaml"
            ).read_text()
        )
    )
    sample_store.identifier = store_id
    sql_store.addResource(sample_store)

    space = random_space_resource_from_file(sample_store_id=store_id)
    create_resource_with_related_identifiers(space, [store_id])

    operation = ml_multi_cloud_operation_resource(space_id=space.identifier)
    create_resource_with_related_identifiers(
        operation,
        [space.identifier, ml_multi_cloud_correct_actuatorconfiguration.identifier],
    )

    default_result = runner.invoke(
        ado,
        ["--override-ado-app-dir", tmp_path, "tree"],
    )
    assert default_result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert ml_multi_cloud_correct_actuatorconfiguration.identifier not in (
            default_result.output
        )

    full_result = runner.invoke(
        ado,
        ["--override-ado-app-dir", tmp_path, "tree", "--all-relationships"],
    )
    assert full_result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert (
            ml_multi_cloud_correct_actuatorconfiguration.identifier
            in full_result.output
        )


@requires_sqlite_3_38
def test_ado_tree_requires_scope_when_resource_type_given(
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

    result = runner.invoke(
        ado,
        ["--override-ado-app-dir", tmp_path, "tree", "operation"],
    )
    assert result.exit_code == 1
