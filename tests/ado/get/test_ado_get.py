# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
import importlib.metadata
import os
import pathlib
from collections.abc import Callable

import pandas as pd
import rich.box
import yaml
from testcontainers.mysql import MySqlContainer
from typer.testing import CliRunner

from ado.cli.core.cli import app as ado
from ado.core import (
    ActuatorConfigurationResource,
    OperationResource,
    SampleStoreResource,
)
from ado.core.discoveryspace.space import DiscoverySpace
from ado.metastore.project import ProjectContext
from ado.metastore.sqlstore import SQLStore
from ado.utilities.rich import dataframe_to_rich_table, render_to_string
from tests.conftest import requires_sqlite_3_38
from tests.utilities.cli_rendering import (
    render_ado_resources_to_cli_output,
)


@requires_sqlite_3_38
def test_space_exists(
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

    result = runner.invoke(ado, ["get", "spaces"])
    assert result.exit_code == 0
    # Travis CI cannot capture output reliably
    if os.environ.get("CI", "false") != "true":
        assert pfas_space.uri in result.output


def test_get_robotic_lab_actuator() -> None:

    runner = CliRunner()

    result = runner.invoke(ado, ["get", "actuator", "robotic_lab"])
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert "robotic_lab" in result.output

    result = runner.invoke(ado, ["get", "actuator", "robotic_lab", "--details"])
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        expected_output = pd.DataFrame(
            data={
                "ACTUATOR ID": "robotic_lab",
                "EXPERIMENTS": 1,
                "DESCRIPTION": "A template for creating an actuator",
                "VERSION": importlib.metadata.version("robotic_lab"),
            },
            index=pd.Index([0]),
        )
        rendered_output = render_to_string(
            dataframe_to_rich_table(
                expected_output, show_index=True, show_edge=True, box=rich.box.SQUARE
            )
        )
        assert rendered_output in result.output


def test_get_vllm_performance_actuator_details() -> None:
    runner = CliRunner()
    result = runner.invoke(ado, ["get", "actuators", "vllm_performance", "--details"])
    assert result.exit_code == 0
    assert "vllm_performance" in result.output


@requires_sqlite_3_38
def test_field_filtering(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    sql_store: SQLStore,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    sample_store_resource: SampleStoreResource,
) -> None:

    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    operation_d5c036 = OperationResource.model_validate(
        yaml.safe_load(
            pathlib.Path(
                "tests/resources/operation/randomwalk-1.0.2.dev17+5e50632.dirty-d5c036.yaml"
            ).read_text()
        )
    )
    sql_store.addResource(operation_d5c036)

    operation_43dfdf = OperationResource.model_validate(
        yaml.safe_load(
            pathlib.Path(
                "tests/resources/operation/randomwalk-1.0.2.dev39+7f0c421.dirty-43dfdf.yaml"
            ).read_text()
        )
    )
    sql_store.addResource(operation_43dfdf)

    sample_store_07c0fa = SampleStoreResource.model_validate(
        yaml.safe_load(
            pathlib.Path(
                "tests/resources/samplestore/sample_store_07c0fa.yaml"
            ).read_text()
        )
    )

    sql_store.addResource(sample_store_07c0fa)
    sql_store.addResource(sample_store_resource)

    actuator_config_with_underscores = ActuatorConfigurationResource.model_validate(
        yaml.safe_load(
            pathlib.Path(
                "tests/resources/actuatorconfiguration/mock-ac-with-snake-case.yaml"
            ).read_text()
        )
    )
    sql_store.addResource(actuator_config_with_underscores)

    # ---------------------------------------------------------
    # Query scalar int field with int
    # ---------------------------------------------------------
    result = runner.invoke(
        ado,
        [
            "get",
            "operations",
            "--filter",
            "config.operation.parameters.batchSize=1",
        ],
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert (
            render_ado_resources_to_cli_output(
                operation_d5c036, do_not_truncate_columns=["IDENTIFIER"]
            )
            == result.output
        ), result.output

    # ---------------------------------------------------------
    # Query scalar int field with float
    # ---------------------------------------------------------
    result = runner.invoke(
        ado,
        [
            "get",
            "operations",
            "--filter",
            "config.operation.parameters.batchSize=1.0",
        ],
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert (
            render_ado_resources_to_cli_output(
                operation_d5c036, do_not_truncate_columns=["IDENTIFIER"]
            )
            == result.output
        ), result.output

    # ---------------------------------------------------------
    # Query scalar int field with string
    # ---------------------------------------------------------
    result = runner.invoke(
        ado,
        [
            "get",
            "operations",
            "--filter",
            'config.parameters.batchSize="1"',
        ],
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert "Nothing was returned" in result.output

    # ---------------------------------------------------------
    # Query scalar null field with null
    # ---------------------------------------------------------
    result = runner.invoke(
        ado,
        [
            "get",
            "samplestores",
            "--filter",
            "config.metadata.name=null",
        ],
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert render_ado_resources_to_cli_output(sample_store_07c0fa) == result.output

    # ---------------------------------------------------------
    # Query scalar null field with string
    # ---------------------------------------------------------
    result = runner.invoke(
        ado,
        [
            "get",
            "samplestores",
            "--filter",
            'config.metadata.name="null"',
        ],
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert "Nothing was returned" in result.output, result.output

    # ---------------------------------------------------------
    # Query scalar boolean field with boolean
    # ---------------------------------------------------------
    result = runner.invoke(
        ado,
        [
            "get",
            "operations",
            "--filter",
            "config.operation.parameters.singleMeasurement=false",
        ],
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert (
            render_ado_resources_to_cli_output(
                operation_d5c036, do_not_truncate_columns=["IDENTIFIER"]
            )
            == result.output
        ), result.output

    # ---------------------------------------------------------
    # Query scalar boolean field with string
    # ---------------------------------------------------------
    result = runner.invoke(
        ado,
        [
            "get",
            "operations",
            "--filter",
            'config.parameters.singleMeasurement="false"',
        ],
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert "Nothing was returned" in result.output

    # ---------------------------------------------------------
    # Query array field with array
    # ---------------------------------------------------------
    result = runner.invoke(
        ado,
        [
            "get",
            "operations",
            "--filter",
            'status=[{"event": "finished", "exit_state": "success"}]',
        ],
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert (
            render_ado_resources_to_cli_output(
                [operation_d5c036, operation_43dfdf],
                do_not_truncate_columns=["IDENTIFIER"],
            )
            == result.output
        ), result.output

    # ---------------------------------------------------------
    # Query array field with scalar
    # ---------------------------------------------------------
    result = runner.invoke(
        ado,
        [
            "get",
            "operations",
            "--filter",
            "config.spaces=space-7dab39-c0c30f",
        ],
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert (
            render_ado_resources_to_cli_output(
                operation_43dfdf, do_not_truncate_columns=["IDENTIFIER"]
            )
            == result.output
        ), result.output

    # ---------------------------------------------------------
    # Query object field with object with nested array
    # ---------------------------------------------------------
    result = runner.invoke(
        ado,
        [
            "get",
            "operations",
            "--filter",
            'config={"spaces": ["space-7dab39-c0c30f"]}',
        ],
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert (
            render_ado_resources_to_cli_output(
                operation_43dfdf, do_not_truncate_columns=["IDENTIFIER"]
            )
            == result.output
        ), result.output

    # ---------------------------------------------------------
    # Query object field with nested objects
    # ---------------------------------------------------------
    result = runner.invoke(
        ado,
        [
            "get",
            "operations",
            "--filter",
            'config.operation.parameters={"batchSize": 2, "samplerConfig": {"mode": "sequential"}}',
        ],
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert (
            render_ado_resources_to_cli_output(
                operation_43dfdf, do_not_truncate_columns=["IDENTIFIER"]
            )
            == result.output
        ), result.output

    # ---------------------------------------------------------
    # Query nested fields with underscores
    # ---------------------------------------------------------
    # Query for nested underscore fields
    result = runner.invoke(
        ado,
        [
            "get",
            "actuatorconfigurations",
            "--filter",
            'config.parameters.outer_field.inner_field.test_value="found_it"',
        ],
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert (
            render_ado_resources_to_cli_output(
                actuator_config_with_underscores,
                do_not_truncate_columns=["IDENTIFIER"],
            )
            == result.output
        ), result.output

    # Query with JSON object containing underscore fields
    result = runner.invoke(
        ado,
        [
            "get",
            "actuatorconfigurations",
            "--filter",
            'config.parameters.outer_field={"inner_field": {"test_value": "found_it"}}',
        ],
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert (
            render_ado_resources_to_cli_output(
                actuator_config_with_underscores,
                do_not_truncate_columns=["IDENTIFIER"],
            )
            == result.output
        ), result.output

    # Query for mixed underscore and simple fields
    result = runner.invoke(
        ado,
        [
            "get",
            "actuatorconfigurations",
            "--filter",
            'config.parameters.outer_field.another_field="simple"',
        ],
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert (
            render_ado_resources_to_cli_output(
                actuator_config_with_underscores,
                do_not_truncate_columns=["IDENTIFIER"],
            )
            == result.output
        ), result.output


@requires_sqlite_3_38
def test_get_space_with_use_latest(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    sql_store: SQLStore,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    """Test getting the latest space using --use-latest flag"""
    from ado.core import DiscoverySpaceResource

    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    # Create two spaces explicitly
    from datetime import datetime, timezone

    # Load file content once
    space_data = yaml.safe_load(
        pathlib.Path("tests/resources/space/discoveryspace_resource.json").read_text()
    )

    # Create first space
    space_1 = DiscoverySpaceResource.model_validate(space_data)
    space_1.identifier = "space-test-older"
    space_1.created = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    sql_store.addResource(space_1)

    # Create second space
    space_2 = DiscoverySpaceResource.model_validate(space_data)
    space_2.identifier = "space-test-latest"
    space_2.created = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    sql_store.addResource(space_2)

    # Test with YAML output format
    result = runner.invoke(
        ado,
        [
            "get",
            "space",
            "--use-latest",
            "-o",
            "yaml",
        ],
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        # Should return the latest space (space_2)
        assert space_2.identifier in result.output
        # Verify it's using the correct space
        assert f"using space {space_2.identifier}" in result.output.lower()
        # Should NOT return the first space
        assert space_1.identifier not in result.output


@requires_sqlite_3_38
def test_get_operation_with_use_latest(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    sql_store: SQLStore,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    sample_store_resource: SampleStoreResource,
) -> None:
    """Test getting the latest operation using --use-latest flag"""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    # Create two operations with different identifiers
    from datetime import datetime, timezone

    # Load file content once
    operation_data = yaml.safe_load(
        pathlib.Path(
            "tests/resources/operation/randomwalk-1.0.2.dev17+5e50632.dirty-d5c036.yaml"
        ).read_text()
    )

    # Create first operation
    operation_1 = OperationResource.model_validate(operation_data)
    operation_1.identifier = "operation-test-older"
    operation_1.created = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    sql_store.addResource(operation_1)

    # Create second operation
    operation_2 = OperationResource.model_validate(operation_data)
    operation_2.identifier = "operation-test-latest"
    operation_2.created = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    sql_store.addResource(operation_2)

    # Test with YAML output format
    result = runner.invoke(
        ado,
        [
            "get",
            "operation",
            "--use-latest",
            "-o",
            "yaml",
        ],
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        # Should return the latest operation (operation_2)
        assert operation_2.identifier in result.output
        # Verify it's using the correct operation
        assert f"using operation {operation_2.identifier}" in result.output.lower()
        # Should NOT return the first operation
        assert operation_1.identifier not in result.output


@requires_sqlite_3_38
def test_get_with_use_latest_and_explicit_id(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    sql_store: SQLStore,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    """Test that explicit ID takes precedence over --use-latest"""
    from ado.core import DiscoverySpaceResource

    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    # Create two spaces with different identifiers
    from datetime import datetime, timezone

    # Load file content once
    space_data = yaml.safe_load(
        pathlib.Path("tests/resources/space/discoveryspace_resource.json").read_text()
    )

    # Create first space
    space_1 = DiscoverySpaceResource.model_validate(space_data)
    space_1.identifier = "space-test-older"
    space_1.created = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    sql_store.addResource(space_1)

    # Create second space
    space_2 = DiscoverySpaceResource.model_validate(space_data)
    space_2.identifier = "space-test-latest"
    space_2.created = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    sql_store.addResource(space_2)

    # Test with both explicit ID and --use-latest
    result = runner.invoke(
        ado,
        [
            "get",
            "space",
            space_1.identifier,
            "--use-latest",
            "-o",
            "yaml",
        ],
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        # Verify warning message about precedence
        assert (
            "explicitly specified resource ids take precedence" in result.output.lower()
        )
        # Verify the correct space is returned (space_1, not space_2)
        assert space_1.identifier in result.output
        # Should NOT return the latest space since explicit ID takes precedence
        assert space_2.identifier not in result.output


@requires_sqlite_3_38
def test_get_with_use_latest_table_format(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    sql_store: SQLStore,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    """Test --use-latest with table output format (default)"""
    from ado.core import DiscoverySpaceResource

    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    # Create two spaces with different identifiers
    from datetime import datetime, timezone

    # Load file content once
    space_data = yaml.safe_load(
        pathlib.Path("tests/resources/space/discoveryspace_resource.json").read_text()
    )

    # Create first space
    space_1 = DiscoverySpaceResource.model_validate(space_data)
    space_1.identifier = "space-test-older"
    space_1.created = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    sql_store.addResource(space_1)

    # Create second space
    space_2 = DiscoverySpaceResource.model_validate(space_data)
    space_2.identifier = "space-test-latest"
    space_2.created = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    sql_store.addResource(space_2)

    # Test with table format (default)
    result = runner.invoke(ado, ["get", "space", "--use-latest"])
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        # Should return the latest space (space_2)
        assert space_2.identifier in result.output
        # Should NOT return the first space
        assert space_1.identifier not in result.output


@requires_sqlite_3_38
def test_get_with_use_latest_name_format(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    sql_store: SQLStore,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    """Test --use-latest with name output format"""
    from ado.core import DiscoverySpaceResource

    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    # Create two spaces with different identifiers
    from datetime import datetime, timezone

    # Load file content once
    space_data = yaml.safe_load(
        pathlib.Path("tests/resources/space/discoveryspace_resource.json").read_text()
    )

    # Create first space
    space_1 = DiscoverySpaceResource.model_validate(space_data)
    space_1.identifier = "space-test-older"
    space_1.created = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    sql_store.addResource(space_1)

    # Create second space
    space_2 = DiscoverySpaceResource.model_validate(space_data)
    space_2.identifier = "space-test-latest"
    space_2.created = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    sql_store.addResource(space_2)

    # Test with name format
    result = runner.invoke(
        ado,
        [
            "get",
            "space",
            "--use-latest",
            "-o",
            "name",
        ],
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        # Name format should output just the identifier
        assert space_2.identifier in result.output
        # Should NOT return the first space
        assert space_1.identifier not in result.output
        # Should not contain table formatting
        assert "IDENTIFIER" not in result.output
