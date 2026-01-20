# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import os
import pathlib
import sqlite3
from collections.abc import Callable

import pytest
import yaml
from testcontainers.mysql import MySqlContainer
from typer.testing import CliRunner

from orchestrator.cli.core.cli import app as ado
from orchestrator.core import OperationResource, SampleStoreResource
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.samplestore.sql import SQLSampleStore
from orchestrator.metastore.project import ProjectContext
from orchestrator.metastore.sqlstore import SQLStore

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
        assert "robotic_lab" in result.output
        assert "peptide_mineralization" in result.output


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
    empty_sample_store: SQLSampleStore,
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
        assert operation_d5c036.identifier in result.output
        assert operation_43dfdf.identifier not in result.output

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
        assert operation_d5c036.identifier in result.output
        assert operation_43dfdf.identifier not in result.output

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
        assert operation_d5c036.identifier not in result.output
        assert operation_43dfdf.identifier not in result.output

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
        assert empty_sample_store.identifier in result.output
        assert sample_store_resource.identifier not in result.output
        assert operation_d5c036.identifier not in result.output
        assert operation_43dfdf.identifier not in result.output

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
        assert empty_sample_store.identifier not in result.output
        assert sample_store_resource.identifier not in result.output
        assert operation_d5c036.identifier not in result.output
        assert operation_43dfdf.identifier not in result.output

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
        assert operation_d5c036.identifier in result.output
        assert operation_43dfdf.identifier not in result.output

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
        assert operation_d5c036.identifier not in result.output
        assert operation_43dfdf.identifier not in result.output

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
        assert operation_43dfdf.identifier in result.output
        assert operation_d5c036.identifier in result.output

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
        assert operation_43dfdf.identifier in result.output
        assert operation_d5c036.identifier not in result.output

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
        assert operation_43dfdf.identifier in result.output
        assert operation_d5c036.identifier not in result.output

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
        assert operation_43dfdf.identifier in result.output
        assert operation_d5c036.identifier not in result.output
