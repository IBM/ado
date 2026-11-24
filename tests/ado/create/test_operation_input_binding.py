# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for binding operation resource inputs from operator signatures."""

import pathlib
from collections.abc import Callable
from contextlib import AbstractContextManager

import pytest
from typer.testing import CliRunner

from ado.cli.core.cli import app as ado
from ado.cli.resources.operation.create import (
    _descriptor_for_kind,
    _resource_inputs_for_module,
)
from ado.cli.resources.operation.template import _operation_inputs_template
from ado.core.datacontainer.resource import DataContainerResource
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.config import (
    DiscoveryOperationEnum,
    FunctionOperationInfo,
    GenericOperatorParameters,
    OperatorMetadata,
)
from ado.core.operation.operation import OperationOutput
from ado.core.resources import (
    ADOResourcePropertyDescriptor,
    CoreResourceKinds,
)
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
from tests.ado.create.conftest import raw_operation, write_operation
from tests.fixtures.shared_mysql_instance import MySqlConnectionDetails


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


def _legacy_metadata(
    required_resource_inputs: tuple[ADOResourcePropertyDescriptor, ...] = (),
) -> OperatorMetadata:
    """Return operator metadata for a class-style module configuration."""
    return OperatorMetadata(
        name="legacy-operator",
        version="1.0.0",
        configuration_model=GenericOperatorParameters,
        example_configuration=GenericOperatorParameters(),
        type=DiscoveryOperationEnum.CHARACTERIZE,
        required_resource_inputs=required_resource_inputs,
    )


class _LegacyOperatorWithoutInputs:
    """Class operator whose metadata has not recorded resource inputs."""

    @classmethod
    def operator_metadata(cls) -> OperatorMetadata:
        """Return metadata with no recorded resource inputs."""
        return _legacy_metadata()


class _LegacyOperatorWithContainer:
    """Class operator that records one data-container input."""

    @classmethod
    def operator_metadata(cls) -> OperatorMetadata:
        """Return metadata with a measurements input."""
        return _legacy_metadata(
            (
                ADOResourcePropertyDescriptor(
                    identifier="measurements",
                    kind=CoreResourceKinds.DATACONTAINER,
                ),
            )
        )


def _descriptor_for_raw(
    raw: dict,
    kind: CoreResourceKinds,
    assigned_inputs: set[str] | None = None,
) -> ADOResourcePropertyDescriptor:
    """Resolve the unqualified input descriptor for *kind* on *raw*."""
    module = raw.get("operation", {}).get("module", {})
    operator_name, descriptors = _resource_inputs_for_module(module)
    return _descriptor_for_kind(
        operator_name,
        descriptors,
        kind,
        dict(raw.get("inputs") or {}),
        assigned_inputs or set(),
    )


class TestDescriptorForKind:
    def test_single_match_returns_identifier(
        self, temporary_operator: Callable[..., AbstractContextManager[str]]
    ) -> None:
        """One input of the requested kind returns that parameter's descriptor."""
        with temporary_operator(
            characterize.operators, characterize_operation, _one_container
        ) as name:
            raw = raw_operation(name, "characterize")
            assert (
                _descriptor_for_raw(raw, CoreResourceKinds.DATACONTAINER).identifier
                == "measurements"
            )

    def test_multiple_unset_inputs_of_the_same_kind_raise(
        self, temporary_operator: Callable[..., AbstractContextManager[str]]
    ) -> None:
        """Several unset inputs of one kind cannot be chosen automatically."""
        with temporary_operator(
            compare.operators, compare_operation, _two_containers
        ) as name:
            raw = raw_operation(name, "compare")
            with pytest.raises(ValueError, match="multiple"):
                _descriptor_for_raw(raw, CoreResourceKinds.DATACONTAINER)

    def test_single_missing_input_is_chosen(
        self, temporary_operator: Callable[..., AbstractContextManager[str]]
    ) -> None:
        """When one input of the kind is unset, its descriptor is returned."""
        with temporary_operator(fuse.operators, fuse_operation, _two_spaces) as name:
            raw = raw_operation(
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
                _descriptor_for_raw(raw, CoreResourceKinds.DISCOVERYSPACE).identifier
                == "right"
            )

    def test_legacy_module_without_recorded_inputs_uses_discovery_space(self) -> None:
        """A class module with empty resource inputs binds discoverySpace."""
        raw = {
            "operation": {
                "module": {
                    "moduleName": __name__,
                    "moduleClass": "_LegacyOperatorWithoutInputs",
                }
            }
        }
        assert (
            _descriptor_for_raw(raw, CoreResourceKinds.DISCOVERYSPACE).identifier
            == "discoverySpace"
        )

    def test_legacy_module_uses_recorded_resource_inputs(self) -> None:
        """A class module binds the input recorded on its metadata."""
        raw = {
            "operation": {
                "module": {
                    "moduleName": __name__,
                    "moduleClass": "_LegacyOperatorWithContainer",
                }
            }
        }
        assert (
            _descriptor_for_raw(raw, CoreResourceKinds.DATACONTAINER).identifier
            == "measurements"
        )

    def test_script_module_uses_discovery_space(self) -> None:
        """A script module binds the discoverySpace input."""
        raw = {"operation": {"module": {"name": "my-script", "version": "0.1.0"}}}
        assert (
            _descriptor_for_raw(raw, CoreResourceKinds.DISCOVERYSPACE).identifier
            == "discoverySpace"
        )


def test_operation_inputs_template_emits_list_placeholder() -> None:
    """Templates emit a one-element list for isList resource inputs."""

    class _Operator:
        required_resource_inputs = (
            ADOResourcePropertyDescriptor(
                identifier="containers",
                kind=CoreResourceKinds.DATACONTAINER,
                isList=True,
            ),
            ADOResourcePropertyDescriptor(
                identifier="baseline",
                kind=CoreResourceKinds.DATACONTAINER,
            ),
        )

    template = _operation_inputs_template(_Operator())
    assert isinstance(template["containers"], list)
    assert len(template["containers"]) == 1
    assert template["containers"][0].kind == CoreResourceKinds.DATACONTAINER
    assert template["baseline"].kind == CoreResourceKinds.DATACONTAINER
    assert not isinstance(template["baseline"], list)


def test_create_operation_dry_run_allows_multiple_discovery_spaces(
    temporary_operator: Callable[..., AbstractContextManager[str]],
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlConnectionDetails,
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
    with temporary_operator(fuse.operators, fuse_operation, _two_spaces) as name:
        write_operation(
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
    temporary_operator: Callable[..., AbstractContextManager[str]],
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlConnectionDetails,
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
    with temporary_operator(
        characterize.operators, characterize_operation, _one_container
    ) as name:
        write_operation(operation_file, name, "characterize")
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
    temporary_operator: Callable[..., AbstractContextManager[str]],
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlConnectionDetails,
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
    with temporary_operator(
        characterize.operators, characterize_operation, _one_container
    ) as name:
        write_operation(operation_file, name, "characterize")
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


def test_with_datacontainer_file_is_not_wrapped_as_generic_failure(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlConnectionDetails,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    """A datacontainer YAML file keeps its specific error."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )
    container_file = tmp_path / "container.yaml"
    container_file.write_text("metadata: {}\n")
    operation_file = tmp_path / "operation.yaml"
    write_operation(operation_file, "random_walk", "explore")
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
            f"datacontainer={container_file}",
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "identifier only" in result.output
    assert "Failed to apply --with/--use-latest" not in result.output
