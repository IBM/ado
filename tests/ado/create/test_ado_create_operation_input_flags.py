# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
"""Tests for --with and --use-latest targeting named operation inputs."""

import pathlib
from collections.abc import Callable, Iterator

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


class _Params(GenericOperatorParameters):
    value: int = 1


_OPERATOR_NAME = "_test_compare_input_flags"


@pytest.fixture
def compare_dummy_operator() -> Iterator[None]:
    """Register a compare operator with two space inputs and one data container."""

    @compare_operation(
        name=_OPERATOR_NAME,
        version="0.0.1",
        configuration_model=_Params,
        example_configuration=_Params(),
        description="Test compare operator",
    )
    def compare_spaces(
        reference: DiscoverySpace,
        candidate: DiscoverySpace,
        data: DataContainerResource,
        operationInfo: FunctionOperationInfo | None = None,
        *,
        parameters: _Params,
    ) -> OperationOutput: ...

    try:
        yield
    finally:
        compare.operators.pop(_OPERATOR_NAME, None)


def _raw_operation(inputs: dict | None = None) -> dict:
    """Return a raw operation configuration for the dummy compare operator."""
    raw: dict = {
        "operation": {
            "module": {
                "operatorName": _OPERATOR_NAME,
                "operationType": DiscoveryOperationEnum.COMPARE.value,
            },
            "parameters": {"value": 1},
        }
    }
    if inputs is not None:
        raw["inputs"] = inputs
    return raw


def _create_parameters(
    with_resources: dict | None = None,
    with_inputs: list[OperationInputResourceOption] | None = None,
) -> AdoCreateCommandParameters:
    """Build create parameters carrying only --with values."""
    return AdoCreateCommandParameters(
        ado_configuration=AdoConfiguration(),
        dry_run=True,
        new_sample_store=False,
        override_values=[],
        resource_configuration_file=None,
        resource_type=AdoCreateSupportedResourceTypes.OPERATION,
        use_default_sample_store=False,
        use_latest=[],
        with_resources=with_resources or {},
        withInputs=with_inputs or [],
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
            inputName="reference",
            kind=CoreResourceKinds.DISCOVERYSPACE,
            value="space-1",
        ),
        OperationInputResourceOption(
            inputName="data", kind=CoreResourceKinds.DATACONTAINER, value="dc-1"
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


def test_apply_with_sets_named_inputs(compare_dummy_operator: None) -> None:
    """Qualified --with values are written to the named inputs."""
    raw = _raw_operation()
    apply_resource_flags(
        raw,
        _create_parameters(
            with_inputs=[
                OperationInputResourceOption(
                    inputName="reference",
                    kind=CoreResourceKinds.DISCOVERYSPACE,
                    value="space-ref",
                ),
                OperationInputResourceOption(
                    inputName="candidate",
                    kind=CoreResourceKinds.DISCOVERYSPACE,
                    value="space-cand",
                ),
            ],
            with_resources={CoreResourceKinds.DATACONTAINER: "dc-1"},
        ),
        latest_identifiers={},
    )
    assert raw["inputs"] == {
        "reference": {"identifier": "space-ref", "kind": "discoveryspace"},
        "candidate": {"identifier": "space-cand", "kind": "discoveryspace"},
        "data": {"identifier": "dc-1", "kind": "datacontainer"},
    }


def test_apply_with_unqualified_fills_remaining_input(
    compare_dummy_operator: None,
) -> None:
    """An unqualified --with fills the input of its kind not set by a qualified one."""
    raw = _raw_operation()
    apply_resource_flags(
        raw,
        _create_parameters(
            with_inputs=[
                OperationInputResourceOption(
                    inputName="reference",
                    kind=CoreResourceKinds.DISCOVERYSPACE,
                    value="space-ref",
                )
            ],
            with_resources={CoreResourceKinds.DISCOVERYSPACE: "space-cand"},
        ),
        latest_identifiers={},
    )
    assert raw["inputs"]["reference"]["identifier"] == "space-ref"
    assert raw["inputs"]["candidate"]["identifier"] == "space-cand"


def test_apply_with_unknown_input_name_raises(compare_dummy_operator: None) -> None:
    """A qualified --with naming an undeclared input raises."""
    with pytest.raises(ValueError, match="missing_input"):
        apply_resource_flags(
            _raw_operation(),
            _create_parameters(
                with_inputs=[
                    OperationInputResourceOption(
                        inputName="missing_input",
                        kind=CoreResourceKinds.DISCOVERYSPACE,
                        value="space-1",
                    )
                ]
            ),
            latest_identifiers={},
        )


def test_apply_with_kind_mismatch_raises(compare_dummy_operator: None) -> None:
    """A qualified --with whose kind differs from the input's kind raises."""
    with pytest.raises(ValueError, match="reference"):
        apply_resource_flags(
            _raw_operation(),
            _create_parameters(
                with_inputs=[
                    OperationInputResourceOption(
                        inputName="reference",
                        kind=CoreResourceKinds.DATACONTAINER,
                        value="dc-1",
                    )
                ]
            ),
            latest_identifiers={},
        )


def test_apply_with_same_scalar_input_twice_raises(
    compare_dummy_operator: None,
) -> None:
    """Setting one scalar input from two flags raises."""
    with pytest.raises(ValueError, match="reference"):
        apply_resource_flags(
            _raw_operation(),
            _create_parameters(
                with_inputs=[
                    OperationInputResourceOption(
                        inputName="reference",
                        kind=CoreResourceKinds.DISCOVERYSPACE,
                        value="space-1",
                    ),
                    OperationInputResourceOption(
                        inputName="reference",
                        kind=CoreResourceKinds.DISCOVERYSPACE,
                        value="space-2",
                    ),
                ]
            ),
            latest_identifiers={},
        )


def test_apply_with_ambiguous_kind_suggests_qualifier(
    compare_dummy_operator: None,
) -> None:
    """An unqualified --with for an ambiguous kind names the qualified syntax."""
    with pytest.raises(ValueError, match=r"discoveryspace:<input>"):
        apply_resource_flags(
            _raw_operation(),
            _create_parameters(
                with_resources={CoreResourceKinds.DISCOVERYSPACE: "space-1"}
            ),
            latest_identifiers={},
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_operation_yaml(path: pathlib.Path) -> pathlib.Path:
    """Write an operation YAML for the dummy compare operator with no inputs."""
    operation_file = path / "compare_operation.yaml"
    operation_file.write_text(
        "operation:\n"
        "  module:\n"
        f"    operatorName: {_OPERATOR_NAME}\n"
        "    operationType: compare\n"
        "  parameters:\n"
        "    value: 1\n"
    )
    return operation_file


def test_create_operation_dry_run_with_named_inputs_and_use_latest(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    sql_store: SQLStore,
    compare_dummy_operator: None,
) -> None:
    """--with and --use-latest with input qualifiers combine on one operation."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )
    data_container = DataContainerResource(config=DataContainer(data={"k": "v"}))
    sql_store.addResource(data_container)

    result = runner.invoke(
        ado,
        [
            "create",
            "operation",
            "-f",
            str(_write_operation_yaml(tmp_path)),
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
