# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
"""Tests for --with and --use-latest targeting named operation inputs."""

import pathlib
from collections.abc import Callable
from contextlib import AbstractContextManager

import pytest
import typer
from testcontainers.community.mysql import MySqlContainer
from typer.testing import CliRunner

from ado.cli.commands.create import (
    parse_use_latest_options,
    parse_with_resource_options,
)
from ado.cli.core.cli import app as ado
from ado.cli.core.config import AdoConfiguration
from ado.cli.models.parameters import (
    AdoCreateCommandParameters,
    OperationInputResourceOption,
)
from ado.cli.models.types import AdoCreateSupportedResourceTypes
from ado.cli.resources.operation.create import apply_resource_flags
from ado.cli.utils.input.parsers import resource_kind_with_input_parser
from ado.core.datacontainer.resource import DataContainer, DataContainerResource
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.config import (
    DiscoveryOperationEnum,
    FunctionOperationInfo,
    GenericOperatorParameters,
)
from ado.core.operation.operation import OperationOutput
from ado.core.resources import CoreResourceKinds
from ado.metastore.project import ProjectContext
from ado.metastore.sqlstore import SQLStore
from ado.modules.operators.collections import compare, compare_operation


def _two_spaces_and_container(
    reference: DiscoverySpace,
    candidate: DiscoverySpace,
    data: DataContainerResource,
    operationInfo: FunctionOperationInfo | None = None,
    *,
    parameters: GenericOperatorParameters,
) -> OperationOutput: ...


def _raw_operation(name: str, inputs: dict | None = None) -> dict:
    """Return a raw operation configuration for a compare operator named *name*."""
    raw: dict = {
        "operation": {
            "module": {
                "operatorName": name,
                "operationType": DiscoveryOperationEnum.COMPARE.value,
            },
            "parameters": {},
        }
    }
    if inputs is not None:
        raw["inputs"] = inputs
    return raw


def _create_parameters(
    with_resources: dict | None = None,
    with_inputs: list[OperationInputResourceOption] | None = None,
    use_latest: list[CoreResourceKinds] | None = None,
) -> AdoCreateCommandParameters:
    """Build create parameters carrying --with and --use-latest values."""
    return AdoCreateCommandParameters(
        ado_configuration=AdoConfiguration(),
        dry_run=True,
        new_sample_store=False,
        override_values=[],
        resource_configuration_file=None,
        resource_type=AdoCreateSupportedResourceTypes.OPERATION,
        use_default_sample_store=False,
        use_latest=use_latest or [],
        with_resources=with_resources or {},
        with_inputs=with_inputs or [],
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_with_resource_options_splits_qualified_inputs() -> None:
    """Qualified --with keys become input options; unqualified stay kind-keyed."""
    with_resources, with_inputs = parse_with_resource_options(
        ["space:reference=space-1", "dcr:data=dc-1", "store=store-1"]
    )
    assert with_resources == {CoreResourceKinds.SAMPLESTORE: "store-1"}
    assert with_inputs == [
        OperationInputResourceOption(
            input_name="reference",
            kind=CoreResourceKinds.DISCOVERYSPACE,
            value="space-1",
        ),
        OperationInputResourceOption(
            input_name="data", kind=CoreResourceKinds.DATACONTAINER, value="dc-1"
        ),
    ]


def test_parse_with_resource_options_accepts_unqualified_datacontainer() -> None:
    """--with dcr=ID is accepted as a data container."""
    with_resources, with_inputs = parse_with_resource_options(["dcr=dc-1"])
    assert with_resources == {CoreResourceKinds.DATACONTAINER: "dc-1"}
    assert with_inputs == []


@pytest.mark.parametrize(
    "option",
    [
        "store:foo=store-1",
        "actuatorconfiguration:foo=ac-1",
        "space:not-an-identifier=space-1",
        "space:=space-1",
        "operation:foo=op-1",
    ],
)
def test_parse_with_resource_options_rejects_invalid_qualifier(option: str) -> None:
    """Qualifiers require an operator input kind and a valid input name."""
    with pytest.raises(typer.Exit):
        parse_with_resource_options([option])


def test_parse_use_latest_options_splits_qualified_inputs() -> None:
    """Qualified --use-latest values become (input, kind) pairs."""
    kinds, inputs = parse_use_latest_options(
        ["discoveryspace", "datacontainer:data", "discoveryspace:reference"]
    )
    assert kinds == [CoreResourceKinds.DISCOVERYSPACE]
    assert inputs == [
        ("data", CoreResourceKinds.DATACONTAINER),
        ("reference", CoreResourceKinds.DISCOVERYSPACE),
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("space", "discoveryspace"),
        ("dcr", "datacontainer"),
        ("dcr:data", "datacontainer:data"),
        ("space:reference", "discoveryspace:reference"),
    ],
)
def test_resource_kind_with_input_parser_resolves_shorthands(
    value: str, expected: str
) -> None:
    """The --use-latest parser resolves the kind shorthand and keeps the input."""
    assert resource_kind_with_input_parser(CoreResourceKinds)(value) == expected


@pytest.mark.parametrize("value", ["bogus", "bogus:data", "space:not-valid", "a:b:c"])
def test_resource_kind_with_input_parser_rejects_invalid(value: str) -> None:
    """The --use-latest parser rejects unknown kinds and invalid input names."""
    with pytest.raises(typer.BadParameter):
        resource_kind_with_input_parser(CoreResourceKinds)(value)


# ---------------------------------------------------------------------------
# Applying --with and --use-latest to raw operation configurations
# ---------------------------------------------------------------------------


def test_apply_with_sets_named_inputs(
    temporary_operator: Callable[..., AbstractContextManager[str]],
) -> None:
    """Qualified --with values are written to the named inputs."""
    with temporary_operator(
        compare.operators, compare_operation, _two_spaces_and_container
    ) as name:
        raw = _raw_operation(name)
        apply_resource_flags(
            raw,
            _create_parameters(
                with_inputs=[
                    OperationInputResourceOption(
                        input_name="reference",
                        kind=CoreResourceKinds.DISCOVERYSPACE,
                        value="space-ref",
                    ),
                    OperationInputResourceOption(
                        input_name="candidate",
                        kind=CoreResourceKinds.DISCOVERYSPACE,
                        value="space-cand",
                    ),
                ],
                with_resources={CoreResourceKinds.DATACONTAINER: "dc-1"},
            ),
        )
    assert raw["inputs"] == {
        "reference": {"identifier": "space-ref", "kind": "discoveryspace"},
        "candidate": {"identifier": "space-cand", "kind": "discoveryspace"},
        "data": {"identifier": "dc-1", "kind": "datacontainer"},
    }


def test_apply_with_unqualified_fills_remaining_input(
    temporary_operator: Callable[..., AbstractContextManager[str]],
) -> None:
    """An unqualified --with fills the input of its kind not set by a qualified one."""
    with temporary_operator(
        compare.operators, compare_operation, _two_spaces_and_container
    ) as name:
        raw = _raw_operation(name)
        apply_resource_flags(
            raw,
            _create_parameters(
                with_inputs=[
                    OperationInputResourceOption(
                        input_name="reference",
                        kind=CoreResourceKinds.DISCOVERYSPACE,
                        value="space-ref",
                    )
                ],
                with_resources={CoreResourceKinds.DISCOVERYSPACE: "space-cand"},
            ),
        )
    assert raw["inputs"]["reference"]["identifier"] == "space-ref"
    assert raw["inputs"]["candidate"]["identifier"] == "space-cand"


def test_apply_with_unknown_input_name_raises(
    temporary_operator: Callable[..., AbstractContextManager[str]],
) -> None:
    """A qualified --with naming an undeclared input raises."""
    with (
        temporary_operator(
            compare.operators, compare_operation, _two_spaces_and_container
        ) as name,
        pytest.raises(ValueError, match="missing_input"),
    ):
        apply_resource_flags(
            _raw_operation(name),
            _create_parameters(
                with_inputs=[
                    OperationInputResourceOption(
                        input_name="missing_input",
                        kind=CoreResourceKinds.DISCOVERYSPACE,
                        value="space-1",
                    )
                ]
            ),
        )


def test_apply_with_kind_mismatch_raises(
    temporary_operator: Callable[..., AbstractContextManager[str]],
) -> None:
    """A qualified --with whose kind differs from the input's kind raises."""
    with (
        temporary_operator(
            compare.operators, compare_operation, _two_spaces_and_container
        ) as name,
        pytest.raises(ValueError, match="reference"),
    ):
        apply_resource_flags(
            _raw_operation(name),
            _create_parameters(
                with_inputs=[
                    OperationInputResourceOption(
                        input_name="reference",
                        kind=CoreResourceKinds.DATACONTAINER,
                        value="dc-1",
                    )
                ]
            ),
        )


def test_apply_with_same_scalar_input_twice_raises(
    temporary_operator: Callable[..., AbstractContextManager[str]],
) -> None:
    """Setting one scalar input from two flags raises."""
    with (
        temporary_operator(
            compare.operators, compare_operation, _two_spaces_and_container
        ) as name,
        pytest.raises(ValueError, match="reference"),
    ):
        apply_resource_flags(
            _raw_operation(name),
            _create_parameters(
                with_inputs=[
                    OperationInputResourceOption(
                        input_name="reference",
                        kind=CoreResourceKinds.DISCOVERYSPACE,
                        value="space-1",
                    ),
                    OperationInputResourceOption(
                        input_name="reference",
                        kind=CoreResourceKinds.DISCOVERYSPACE,
                        value="space-2",
                    ),
                ]
            ),
        )


def test_apply_with_ambiguous_kind_suggests_qualifier(
    temporary_operator: Callable[..., AbstractContextManager[str]],
) -> None:
    """An unqualified --with for an ambiguous kind names the qualified syntax."""
    with (
        temporary_operator(
            compare.operators, compare_operation, _two_spaces_and_container
        ) as name,
        pytest.raises(ValueError, match=r"discoveryspace:<input>"),
    ):
        apply_resource_flags(
            _raw_operation(name),
            _create_parameters(
                with_resources={CoreResourceKinds.DISCOVERYSPACE: "space-1"}
            ),
        )


def test_apply_rejects_same_kind_on_with_and_use_latest(
    temporary_operator: Callable[..., AbstractContextManager[str]],
) -> None:
    """An unqualified kind cannot be given to both --with and --use-latest."""
    with (
        temporary_operator(
            compare.operators, compare_operation, _two_spaces_and_container
        ) as name,
        pytest.raises(ValueError, match="both --with and --use-latest"),
    ):
        apply_resource_flags(
            _raw_operation(name),
            _create_parameters(
                with_resources={CoreResourceKinds.DISCOVERYSPACE: "space-1"},
                use_latest=[CoreResourceKinds.DISCOVERYSPACE],
            ),
        )


def test_apply_unqualified_kind_when_all_already_assigned_raises(
    temporary_operator: Callable[..., AbstractContextManager[str]],
) -> None:
    """An unqualified kind fails clearly when every input of that kind is set."""
    with (
        temporary_operator(
            compare.operators, compare_operation, _two_spaces_and_container
        ) as name,
        pytest.raises(ValueError, match="were already set"),
    ):
        apply_resource_flags(
            _raw_operation(name),
            _create_parameters(
                with_inputs=[
                    OperationInputResourceOption(
                        input_name="reference",
                        kind=CoreResourceKinds.DISCOVERYSPACE,
                        value="space-ref",
                    ),
                    OperationInputResourceOption(
                        input_name="candidate",
                        kind=CoreResourceKinds.DISCOVERYSPACE,
                        value="space-cand",
                    ),
                ],
                with_resources={CoreResourceKinds.DISCOVERYSPACE: "space-extra"},
            ),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_operation_yaml(path: pathlib.Path, name: str) -> pathlib.Path:
    """Write an operation YAML for a compare operator named *name* with no inputs."""
    operation_file = path / "compare_operation.yaml"
    operation_file.write_text(
        "operation:\n"
        "  module:\n"
        f"    operatorName: {name}\n"
        "    operationType: compare\n"
        "  parameters: {}\n"
    )
    return operation_file


def test_create_operation_dry_run_with_named_inputs_and_use_latest(
    temporary_operator: Callable[..., AbstractContextManager[str]],
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    sql_store: SQLStore,
) -> None:
    """--with and --use-latest with input qualifiers combine on one operation."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )
    data_container = DataContainerResource(config=DataContainer(data={"k": "v"}))
    sql_store.addResource(data_container)

    with temporary_operator(
        compare.operators, compare_operation, _two_spaces_and_container
    ) as name:
        result = runner.invoke(
            ado,
            [
                "create",
                "operation",
                "-f",
                str(_write_operation_yaml(tmp_path, name)),
                "--with",
                "space:reference=space-ref",
                "--with",
                "space:candidate=space-cand",
                "--use-latest",
                "dcr:data",
                "--dry-run",
            ],
        )
    assert result.exit_code == 0, result.output
    assert data_container.identifier in result.output
    assert "data" in result.output


def test_create_space_rejects_input_qualifier(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    """Input qualifiers are rejected for resources other than operations."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )
    result = runner.invoke(
        ado,
        [
            "create",
            "space",
            "-f",
            "examples/optimization_test_functions/space.yaml",
            "--with",
            "space:reference=space-1",
            "--dry-run",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "only supported when creating operations" in result.output
