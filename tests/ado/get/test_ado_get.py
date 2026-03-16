# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
import importlib.metadata
import os
import pathlib
import sqlite3
from collections.abc import Callable

import pandas as pd
import pytest
import rich.box
import yaml
from testcontainers.mysql import MySqlContainer
from typer.testing import CliRunner

from orchestrator.cli.core.cli import app as ado
from orchestrator.core import OperationResource, SampleStoreResource
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.metastore.project import ProjectContext
from orchestrator.metastore.sqlstore import SQLStore
from orchestrator.utilities.rich import dataframe_to_rich_table, render_to_string
from tests.utilities.cli_rendering import (
    render_ado_resources_to_cli_output,
)

sqlite3_version = sqlite3.sqlite_version_info


# AP: the -> and ->> syntax in SQLite is only supported from version 3.38.0
# ref: https://sqlite.org/json1.html#jptr
@pytest.mark.skipif(
    sqlite3_version < (3, 38, 0), reason="SQLite version 3.38.0 or higher is required"
)
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

    result = runner.invoke(ado, ["--override-ado-app-dir", tmp_path, "get", "spaces"])
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
                expected_output, show_edge=True, box=rich.box.SQUARE
            )
        )
        assert rendered_output in result.output


# AP: the -> and ->> syntax in SQLite is only supported from version 3.38.0
# ref: https://sqlite.org/json1.html#jptr
@pytest.mark.skipif(
    sqlite3_version < (3, 38, 0), reason="SQLite version 3.38.0 or higher is required"
)
def test_field_querying(
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

    # ---------------------------------------------------------
    # Query scalar int field with int
    # ---------------------------------------------------------
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "get",
            "operations",
            "-q",
            "config.operation.parameters.batchSize=1",
        ],
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert (
            render_ado_resources_to_cli_output(operation_d5c036) == result.output
        ), result.output

    # ---------------------------------------------------------
    # Query scalar int field with float
    # ---------------------------------------------------------
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "get",
            "operations",
            "-q",
            "config.operation.parameters.batchSize=1.0",
        ],
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert (
            render_ado_resources_to_cli_output(operation_d5c036) == result.output
        ), result.output

    # ---------------------------------------------------------
    # Query scalar int field with string
    # ---------------------------------------------------------
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "get",
            "operations",
            "-q",
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
            "--override-ado-app-dir",
            tmp_path,
            "get",
            "samplestores",
            "-q",
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
            "--override-ado-app-dir",
            tmp_path,
            "get",
            "samplestores",
            "-q",
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
            "--override-ado-app-dir",
            tmp_path,
            "get",
            "operations",
            "-q",
            "config.operation.parameters.singleMeasurement=false",
        ],
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert (
            render_ado_resources_to_cli_output(operation_d5c036) == result.output
        ), result.output

    # ---------------------------------------------------------
    # Query scalar boolean field with string
    # ---------------------------------------------------------
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "get",
            "operations",
            "-q",
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
            "--override-ado-app-dir",
            tmp_path,
            "get",
            "operations",
            "-q",
            'status=[{"event": "finished", "exit_state": "success"}]',
        ],
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert (
            render_ado_resources_to_cli_output([operation_d5c036, operation_43dfdf])
            == result.output
        ), result.output

    # ---------------------------------------------------------
    # Query array field with scalar
    # ---------------------------------------------------------
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "get",
            "operations",
            "-q",
            "config.spaces=space-7dab39-c0c30f",
        ],
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert (
            render_ado_resources_to_cli_output(operation_43dfdf) == result.output
        ), result.output

    # ---------------------------------------------------------
    # Query object field with object with nested array
    # ---------------------------------------------------------
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "get",
            "operations",
            "-q",
            'config={"spaces": ["space-7dab39-c0c30f"]}',
        ],
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert (
            render_ado_resources_to_cli_output(operation_43dfdf) == result.output
        ), result.output

    # ---------------------------------------------------------
    # Query object field with nested objects
    # ---------------------------------------------------------
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "get",
            "operations",
            "-q",
            'config.operation.parameters={"batchSize": 2, "samplerConfig": {"mode": "sequential"}}',
        ],
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert (
            render_ado_resources_to_cli_output(operation_43dfdf) == result.output
        ), result.output
