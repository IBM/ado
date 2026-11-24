# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
"""Tests for --with and --use-latest targeting named operation inputs."""

import pathlib
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager

import pytest
import typer
from typer.testing import CliRunner

from ado.cli.commands.create import (
    parse_use_latest_options,
    parse_with_resource_options,
)
from ado.cli.core.cli import app as ado
from ado.cli.core.config import AdoConfiguration
from ado.cli.models.parameters import (
    AdoCreateCommandParameters,
    OperationInputReference,
    OperationInputResourceOption,
)
from ado.cli.models.types import AdoCreateSupportedResourceTypes
from ado.cli.resources.operation.create import apply_resource_flags
from ado.cli.utils.input.parsers import resource_kind_with_input_parser
from ado.core.datacontainer.resource import DataContainerResource
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
from tests.ado.create.conftest import raw_operation, write_operation
from tests.fixtures.shared_mysql_instance import MySqlConnectionDetails

SPACE_CONFIGURATION_FILE = pathlib.Path(
    "examples/optimization_test_functions/space.yaml"
)


def _two_spaces_and_container(
    reference: DiscoverySpace,
    candidate: DiscoverySpace,
    data: DataContainerResource,
    operationInfo: FunctionOperationInfo | None = None,
    *,
    parameters: GenericOperatorParameters,
) -> OperationOutput: ...


@pytest.fixture
def compare_operator_name(
    temporary_operator: Callable[..., AbstractContextManager[str]],
) -> Iterator[str]:
    """Register a compare operator with two space inputs and a data container."""
    with temporary_operator(
        compare.operators, compare_operation, _two_spaces_and_container
    ) as name:
        yield name


def _create_parameters(
    with_resources: dict | None = None,
    with_inputs: list[OperationInputResourceOption] | None = None,
    use_latest: list[CoreResourceKinds] | None = None,
    use_latest_inputs: list[OperationInputReference] | None = None,
) -> AdoCreateCommandParameters:
    """Build dry-run create parameters carrying --with and --use-latest values."""
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
        use_latest_inputs=use_latest_inputs or [],
    )


def _space_input(
    input_name: str, value: pathlib.Path | str
) -> OperationInputResourceOption:
    """Return a --with INPUT:discoveryspace=VALUE option."""
    return OperationInputResourceOption(
        input_name=input_name, kind=CoreResourceKinds.DISCOVERYSPACE, value=value
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "option",
    [
        OperationInputReference(
            input_name="data", kind=CoreResourceKinds.DATACONTAINER
        ),
        _space_input("reference", "space-1"),
        _space_input("reference", SPACE_CONFIGURATION_FILE),
    ],
)
def test_operation_input_option_round_trip(
    option: OperationInputReference,
) -> None:
    """Named input options survive a dump and reload with value types intact."""
    reloaded = type(option).model_validate(option.model_dump())
    assert reloaded == option
    if isinstance(option, OperationInputResourceOption):
        assert type(reloaded.value) is type(option.value)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_with_resource_options_splits_qualified_inputs() -> None:
    """Qualified --with keys become input options; unqualified stay kind-keyed."""
    with_resources, with_inputs = parse_with_resource_options(
        ["reference:space=space-1", "data:dcr=dc-1", "store=store-1"]
    )
    assert with_resources == {CoreResourceKinds.SAMPLESTORE: "store-1"}
    assert with_inputs == [
        _space_input("reference", "space-1"),
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
        "foo:store=store-1",
        "foo:actuatorconfiguration=ac-1",
        "not-an-identifier:space=space-1",
        ":space=space-1",
        "foo:operation=op-1",
    ],
)
def test_parse_with_resource_options_rejects_invalid_qualifier(option: str) -> None:
    """Qualifiers require an operator input kind and a valid input name."""
    with pytest.raises(typer.Exit):
        parse_with_resource_options([option])


def test_parse_use_latest_options_splits_qualified_inputs() -> None:
    """Qualified --use-latest values become named input references."""
    kinds, inputs = parse_use_latest_options(
        ["discoveryspace", "data:datacontainer", "reference:discoveryspace"]
    )
    assert kinds == [CoreResourceKinds.DISCOVERYSPACE]
    assert inputs == [
        OperationInputReference(
            input_name="data", kind=CoreResourceKinds.DATACONTAINER
        ),
        OperationInputReference(
            input_name="reference", kind=CoreResourceKinds.DISCOVERYSPACE
        ),
    ]


def test_parse_use_latest_options_rejects_non_input_kind() -> None:
    """A qualified --use-latest kind must be an operator input kind."""
    with pytest.raises(typer.Exit):
        parse_use_latest_options(["store:samplestore"])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("space", "discoveryspace"),
        ("dcr", "datacontainer"),
        ("data:dcr", "data:datacontainer"),
        ("reference:space", "reference:discoveryspace"),
    ],
)
def test_resource_kind_with_input_parser_resolves_shorthands(
    value: str, expected: str
) -> None:
    """The --use-latest parser resolves the kind shorthand and keeps the input."""
    assert resource_kind_with_input_parser(CoreResourceKinds)(value) == expected


@pytest.mark.parametrize("value", ["bogus", "data:bogus", "not-valid:space", "a:b:c"])
def test_resource_kind_with_input_parser_rejects_invalid(value: str) -> None:
    """The --use-latest parser rejects unknown kinds and invalid input names."""
    with pytest.raises(typer.BadParameter):
        resource_kind_with_input_parser(CoreResourceKinds)(value)


# ---------------------------------------------------------------------------
# Applying --with and --use-latest to raw operation configurations
# ---------------------------------------------------------------------------


def test_apply_with_sets_named_inputs(compare_operator_name: str) -> None:
    """Qualified --with values are written to the named inputs."""
    raw = raw_operation(compare_operator_name, DiscoveryOperationEnum.COMPARE.value)
    apply_resource_flags(
        raw,
        _create_parameters(
            with_inputs=[
                _space_input("reference", "space-ref"),
                _space_input("candidate", "space-cand"),
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
    compare_operator_name: str,
) -> None:
    """An unqualified --with fills the input of its kind not set by a qualified one."""
    raw = raw_operation(compare_operator_name, DiscoveryOperationEnum.COMPARE.value)
    apply_resource_flags(
        raw,
        _create_parameters(
            with_inputs=[_space_input("reference", "space-ref")],
            with_resources={CoreResourceKinds.DISCOVERYSPACE: "space-cand"},
        ),
    )
    assert raw["inputs"]["reference"]["identifier"] == "space-ref"
    assert raw["inputs"]["candidate"]["identifier"] == "space-cand"


def test_apply_unqualified_kind_skips_list_set_by_qualified_flag(
    summarize_operator_name: str,
) -> None:
    """An unqualified kind skips a list input already set by a qualified flag."""
    raw = raw_operation(summarize_operator_name, DiscoveryOperationEnum.SUMMARIZE.value)
    apply_resource_flags(
        raw,
        _create_parameters(
            with_inputs=[
                OperationInputResourceOption(
                    input_name="containers",
                    kind=CoreResourceKinds.DATACONTAINER,
                    value="datacontainer-list",
                )
            ],
            with_resources={CoreResourceKinds.DATACONTAINER: "datacontainer-base"},
        ),
    )
    assert raw["inputs"]["containers"] == [
        {"identifier": "datacontainer-list", "kind": "datacontainer"}
    ]
    assert raw["inputs"]["baseline"]["identifier"] == "datacontainer-base"


def test_apply_with_file_in_dry_run_leaves_input_unset(
    compare_operator_name: str,
) -> None:
    """A --with file validated in a dry run claims its input without writing it."""
    raw = raw_operation(compare_operator_name, DiscoveryOperationEnum.COMPARE.value)
    apply_resource_flags(
        raw,
        _create_parameters(
            with_inputs=[_space_input("reference", SPACE_CONFIGURATION_FILE)],
            with_resources={CoreResourceKinds.DISCOVERYSPACE: "space-cand"},
        ),
    )
    assert "reference" not in raw["inputs"]
    assert raw["inputs"]["candidate"]["identifier"] == "space-cand"


@pytest.mark.parametrize(
    ("build_parameters", "error"),
    [
        (
            lambda space_file: _create_parameters(
                with_inputs=[_space_input("missing_input", space_file)]
            ),
            "missing_input",
        ),
        (
            lambda space_file: _create_parameters(
                with_resources={CoreResourceKinds.DISCOVERYSPACE: space_file}
            ),
            r"<input>:discoveryspace",
        ),
    ],
    ids=["unknown-input", "ambiguous-kind"],
)
def test_apply_validates_inputs_before_creating_resources(
    compare_operator_name: str,
    tmp_path: pathlib.Path,
    build_parameters: Callable[[pathlib.Path], AdoCreateCommandParameters],
    error: str,
) -> None:
    """Input errors are raised before a --with file is turned into a resource.

    The space file is invalid, so creating it first would exit with a space
    validation error instead of raising the input error.
    """
    invalid_space_file = tmp_path / "invalid_space.yaml"
    invalid_space_file.write_text("not: a space\n")
    with pytest.raises(ValueError, match=error):
        apply_resource_flags(
            raw_operation(compare_operator_name, DiscoveryOperationEnum.COMPARE.value),
            build_parameters(invalid_space_file),
        )


def test_apply_with_unknown_input_name_raises(compare_operator_name: str) -> None:
    """A qualified --with naming an undeclared input raises."""
    with pytest.raises(ValueError, match="missing_input"):
        apply_resource_flags(
            raw_operation(compare_operator_name, DiscoveryOperationEnum.COMPARE.value),
            _create_parameters(with_inputs=[_space_input("missing_input", "space-1")]),
        )


def test_apply_with_kind_mismatch_raises(compare_operator_name: str) -> None:
    """A qualified --with whose kind differs from the input's kind raises."""
    with pytest.raises(ValueError, match="reference"):
        apply_resource_flags(
            raw_operation(compare_operator_name, DiscoveryOperationEnum.COMPARE.value),
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
    compare_operator_name: str,
) -> None:
    """Setting one scalar input from two flags raises."""
    with pytest.raises(ValueError, match="reference"):
        apply_resource_flags(
            raw_operation(compare_operator_name, DiscoveryOperationEnum.COMPARE.value),
            _create_parameters(
                with_inputs=[
                    _space_input("reference", "space-1"),
                    _space_input("reference", "space-2"),
                ]
            ),
        )


def test_apply_with_ambiguous_kind_suggests_qualifier(
    compare_operator_name: str,
) -> None:
    """An unqualified --with for an ambiguous kind names the qualified syntax."""
    with pytest.raises(ValueError, match=r"<input>:discoveryspace"):
        apply_resource_flags(
            raw_operation(compare_operator_name, DiscoveryOperationEnum.COMPARE.value),
            _create_parameters(
                with_resources={CoreResourceKinds.DISCOVERYSPACE: "space-1"}
            ),
        )


def test_apply_rejects_same_kind_on_with_and_use_latest(
    compare_operator_name: str,
) -> None:
    """An unqualified kind cannot be given to both --with and --use-latest."""
    with pytest.raises(ValueError, match="both --with and --use-latest"):
        apply_resource_flags(
            raw_operation(compare_operator_name, DiscoveryOperationEnum.COMPARE.value),
            _create_parameters(
                with_resources={CoreResourceKinds.DISCOVERYSPACE: "space-1"},
                use_latest=[CoreResourceKinds.DISCOVERYSPACE],
            ),
        )


@pytest.mark.parametrize(
    ("with_resources", "use_latest"),
    [
        ({}, [CoreResourceKinds.SAMPLESTORE]),
        ({}, [CoreResourceKinds.OPERATION]),
        ({CoreResourceKinds.SAMPLESTORE: "store-1"}, []),
    ],
    ids=["use-latest-samplestore", "use-latest-operation", "with-samplestore"],
)
def test_apply_rejects_kinds_operations_cannot_use(
    compare_operator_name: str,
    with_resources: dict,
    use_latest: list[CoreResourceKinds],
) -> None:
    """Kinds that are neither operation inputs nor actuator configurations raise."""
    with pytest.raises(ValueError, match="cannot be used when creating an operation"):
        apply_resource_flags(
            raw_operation(compare_operator_name, DiscoveryOperationEnum.COMPARE.value),
            _create_parameters(with_resources=with_resources, use_latest=use_latest),
        )


def test_apply_unqualified_kind_when_all_already_assigned_raises(
    compare_operator_name: str,
) -> None:
    """An unqualified kind fails clearly when every input of that kind is set."""
    with pytest.raises(ValueError, match="were already set"):
        apply_resource_flags(
            raw_operation(compare_operator_name, DiscoveryOperationEnum.COMPARE.value),
            _create_parameters(
                with_inputs=[
                    _space_input("reference", "space-ref"),
                    _space_input("candidate", "space-cand"),
                ],
                with_resources={CoreResourceKinds.DISCOVERYSPACE: "space-extra"},
            ),
        )


# ---------------------------------------------------------------------------
# List inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("existing_containers", "expected_identifiers"),
    [
        (None, ["datacontainer-aaa"]),
        (
            [{"identifier": "datacontainer-file", "kind": "datacontainer"}],
            ["datacontainer-file", "datacontainer-aaa"],
        ),
        (
            {"identifier": "datacontainer-file", "kind": "datacontainer"},
            ["datacontainer-file", "datacontainer-aaa"],
        ),
    ],
    ids=["unset", "list", "single-mapping"],
)
def test_apply_unqualified_kind_appends_to_list_input(
    summarize_operator_name: str,
    existing_containers: list | dict | None,
    expected_identifiers: list[str],
) -> None:
    """An unqualified --with binds to the list input and appends to its value."""
    inputs: dict = {
        "baseline": {"identifier": "datacontainer-baseline", "kind": "datacontainer"}
    }
    if existing_containers is not None:
        inputs["containers"] = existing_containers
    raw = raw_operation(
        summarize_operator_name, DiscoveryOperationEnum.SUMMARIZE.value, inputs
    )
    apply_resource_flags(
        raw,
        _create_parameters(
            with_resources={CoreResourceKinds.DATACONTAINER: "datacontainer-aaa"}
        ),
    )
    assert [
        container["identifier"] for container in raw["inputs"]["containers"]
    ] == expected_identifiers
    assert raw["inputs"]["baseline"]["identifier"] == "datacontainer-baseline"


def test_apply_named_list_input_appends_each_value(
    summarize_operator_name: str,
) -> None:
    """Repeated --with for a list input append after the file's entries."""
    raw = raw_operation(
        summarize_operator_name,
        DiscoveryOperationEnum.SUMMARIZE.value,
        {
            "containers": [
                {"identifier": "datacontainer-file", "kind": "datacontainer"}
            ],
            "baseline": {
                "identifier": "datacontainer-baseline",
                "kind": "datacontainer",
            },
        },
    )
    apply_resource_flags(
        raw,
        _create_parameters(
            with_inputs=[
                OperationInputResourceOption(
                    input_name="containers",
                    kind=CoreResourceKinds.DATACONTAINER,
                    value=identifier,
                )
                for identifier in ("datacontainer-aaa", "datacontainer-bbb")
            ]
        ),
    )
    assert raw["inputs"]["containers"] == [
        {"identifier": "datacontainer-file", "kind": "datacontainer"},
        {"identifier": "datacontainer-aaa", "kind": "datacontainer"},
        {"identifier": "datacontainer-bbb", "kind": "datacontainer"},
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_create_operation_dry_run_with_named_inputs_and_use_latest(
    compare_operator_name: str,
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlConnectionDetails,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    sql_store: SQLStore,
    data_container_resource: DataContainerResource,
) -> None:
    """--with and --use-latest with input qualifiers combine on one operation."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )
    sql_store.addResource(data_container_resource)
    operation_file = tmp_path / "compare_operation.yaml"
    write_operation(
        operation_file, compare_operator_name, DiscoveryOperationEnum.COMPARE.value
    )

    result = runner.invoke(
        ado,
        [
            "create",
            "operation",
            "-f",
            str(operation_file),
            "--with",
            "reference:space=space-ref",
            "--with",
            "candidate:space=space-cand",
            "--use-latest",
            "data:dcr",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert data_container_resource.identifier in result.output


def test_create_operation_reports_unresolvable_operator_module(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlConnectionDetails,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    """A module that cannot be imported is reported as a flag error, not a traceback."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )
    operation_file = tmp_path / "operation.yaml"
    operation_file.write_text(
        "operation:\n"
        "  module:\n"
        "    moduleName: module_that_does_not_exist\n"
        "    moduleClass: Operator\n"
        "  parameters: {}\n"
    )

    result = runner.invoke(
        ado,
        [
            "create",
            "operation",
            "-f",
            str(operation_file),
            "--with",
            "space=space-1",
            "--dry-run",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "Failed to apply --with/--use-latest" in result.output


def test_create_space_rejects_input_qualifier(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlConnectionDetails,
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
    for flag, value in (
        ("--with", "reference:space=space-1"),
        ("--use-latest", "reference:space"),
    ):
        result = runner.invoke(
            ado,
            [
                "create",
                "space",
                "-f",
                str(SPACE_CONFIGURATION_FILE),
                flag,
                value,
                "--dry-run",
            ],
        )
        assert result.exit_code == 1, result.output
        assert "only supported when creating operations" in result.output
