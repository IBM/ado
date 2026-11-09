# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for binding operation resource inputs from operator signatures."""

import pathlib
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pytest
from testcontainers.community.mysql import MySqlContainer
from typer.testing import CliRunner

from ado.cli.core.cli import app as ado
from ado.cli.resources.operation.create import _operator_input_name_for_kind
from ado.core.datacontainer.resource import DataContainerResource
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.config import FunctionOperationInfo, GenericOperatorParameters
from ado.core.operation.operation import OperationOutput
from ado.core.resources import CoreResourceKinds
from ado.metastore.project import ProjectContext
from ado.metastore.sqlstore import SQLStore
from ado.modules.operators.collections import (
    characterize,
    characterize_operation,
    compare,
    compare_operation,
    fuse,
    fuse_operation,
)


def _stub(
    *args: object,
    **kwargs: object,
) -> OperationOutput: ...


def _one_container(
    measurements: DataContainerResource,
    operationInfo: FunctionOperationInfo | None = None,
    *,
    parameters: GenericOperatorParameters,
) -> OperationOutput:
    return _stub(measurements, operationInfo, parameters=parameters)


def _two_containers(
    baseline: DataContainerResource,
    candidate: DataContainerResource,
    operationInfo: FunctionOperationInfo | None = None,
    *,
    parameters: GenericOperatorParameters,
) -> OperationOutput:
    return _stub(baseline, candidate, operationInfo, parameters=parameters)


def _two_spaces(
    left: DiscoverySpace,
    right: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
    *,
    parameters: GenericOperatorParameters,
) -> OperationOutput:
    return _stub(left, right, operationInfo, parameters=parameters)


@contextmanager
def _temporary_operator(
    collection_operators: dict,
    decorator: Callable[..., Callable[..., OperationOutput]],
    fn: Callable[..., OperationOutput],
) -> Iterator[str]:
    """Register *fn* under a unique name and remove it afterwards."""
    name = f"test-op-{uuid.uuid4().hex[:8]}"
    decorator(
        name=name,
        version="1.0.0",
        configuration_model=GenericOperatorParameters,
        example_configuration=GenericOperatorParameters(),
        description="temporary test operator",
    )(fn)
    try:
        yield name
    finally:
        collection_operators.pop(name, None)


def _raw_operation(
    name: str,
    operation_type: str,
    inputs: dict | None = None,
) -> dict:
    """Return a raw operation configuration for *name*."""
    raw: dict = {
        "operation": {
            "module": {
                "operatorName": name,
                "operationType": operation_type,
            },
            "parameters": {},
        }
    }
    if inputs is not None:
        raw["inputs"] = inputs
    return raw


def _write_operation(
    path: pathlib.Path,
    name: str,
    operation_type: str,
    inputs_yaml: str = "",
) -> None:
    """Write an operation YAML that references *name*."""
    path.write_text(
        "operation:\n"
        "  module:\n"
        f"    operatorName: {name}\n"
        f"    operationType: {operation_type}\n"
        "  parameters: {}\n"
        f"{inputs_yaml}"
    )


class TestOperatorInputNameForKind:
    def test_single_match_returns_identifier(self) -> None:
        """One input of the requested kind returns that parameter name."""
        with _temporary_operator(
            characterize.operators, characterize_operation, _one_container
        ) as name:
            raw = _raw_operation(name, "characterize")
            assert (
                _operator_input_name_for_kind(raw, CoreResourceKinds.DATACONTAINER)
                == "measurements"
            )

    def test_multiple_unset_inputs_of_the_same_kind_raise(self) -> None:
        """Several unset inputs of one kind cannot be chosen automatically."""
        with _temporary_operator(
            compare.operators, compare_operation, _two_containers
        ) as name:
            raw = _raw_operation(name, "compare")
            with pytest.raises(ValueError, match="multiple"):
                _operator_input_name_for_kind(raw, CoreResourceKinds.DATACONTAINER)

    def test_single_missing_input_is_chosen(self) -> None:
        """When one input of the kind is unset, that name is returned."""
        with _temporary_operator(fuse.operators, fuse_operation, _two_spaces) as name:
            raw = _raw_operation(
                name,
                "fuse",
                inputs={
                    "left": {
                        "identifier": "space-left0001",
                        "kind": "discoveryspace",
                    }
                },
            )
            assert (
                _operator_input_name_for_kind(raw, CoreResourceKinds.DISCOVERYSPACE)
                == "right"
            )


def test_create_operation_dry_run_allows_multiple_discovery_spaces(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    """A fuse manifest with two discovery spaces is accepted."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )
    operation_file = tmp_path / "operation.yaml"
    with _temporary_operator(fuse.operators, fuse_operation, _two_spaces) as name:
        _write_operation(
            operation_file,
            name,
            "fuse",
            inputs_yaml=(
                "inputs:\n"
                "  left:\n"
                "    identifier: space-left0001\n"
                "    kind: discoveryspace\n"
                "  right:\n"
                "    identifier: space-right001\n"
                "    kind: discoveryspace\n"
            ),
        )
        result = runner.invoke(
            ado,
            [
                "--override-ado-app-dir",
                str(tmp_path),
                "create",
                "operation",
                "-f",
                str(operation_file),
                "--dry-run",
            ],
        )

    assert "spaces field only supports one value" not in result.output
    assert result.exit_code == 0, result.output


def test_create_operation_dry_run_with_datacontainer(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    """--with datacontainer binds the operator's data-container parameter."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )
    operation_file = tmp_path / "operation.yaml"
    with _temporary_operator(
        characterize.operators, characterize_operation, _one_container
    ) as name:
        _write_operation(operation_file, name, "characterize")
        result = runner.invoke(
            ado,
            [
                "--override-ado-app-dir",
                str(tmp_path),
                "create",
                "operation",
                "-f",
                str(operation_file),
                "--with",
                "datacontainer=datacontainer-test0001",
                "--dry-run",
            ],
        )

    assert result.exit_code == 0, result.output


def test_create_operation_dry_run_use_latest_datacontainer(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    sql_store: SQLStore,
    data_container_resource: DataContainerResource,
) -> None:
    """--use-latest datacontainer binds the operator's data-container parameter."""
    sql_store.addResource(data_container_resource)
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )
    operation_file = tmp_path / "operation.yaml"
    with _temporary_operator(
        characterize.operators, characterize_operation, _one_container
    ) as name:
        _write_operation(operation_file, name, "characterize")
        result = runner.invoke(
            ado,
            [
                "--override-ado-app-dir",
                str(tmp_path),
                "create",
                "operation",
                "-f",
                str(operation_file),
                "--use-latest",
                "datacontainer",
                "--dry-run",
            ],
        )

    assert result.exit_code == 0, result.output
    assert data_container_resource.identifier in result.output
