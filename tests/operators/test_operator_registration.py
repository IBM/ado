# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for registering an operator and calling its wrapper."""

import pathlib
from collections.abc import Callable
from contextlib import AbstractContextManager

import pytest

from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.config import FunctionOperationInfo, GenericOperatorParameters
from ado.core.operation.operation import OperationOutput
from ado.metastore.project import ProjectContext
from ado.modules.operators.collections import (
    characterize,
    characterize_operation,
    compare,
    compare_operation,
)
from ado.utilities.location import SQLiteStoreConfiguration


def _stub(*args: object, **kwargs: object) -> OperationOutput:
    """Return an empty operation output."""
    return OperationOutput()


def _two_spaces(
    left: DiscoverySpace,
    right: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
    *,
    parameters: GenericOperatorParameters,
) -> OperationOutput:
    """Compare two discovery spaces."""
    return _stub(left, right, operationInfo, parameters=parameters)


def _one_space(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
    *,
    parameters: GenericOperatorParameters,
) -> OperationOutput:
    """Characterize one discovery space."""
    return _stub(discoverySpace, operationInfo, parameters=parameters)


def test_compare_registration_records_resource_inputs(
    temporary_operator: Callable[..., AbstractContextManager[str]],
) -> None:
    """Registration stores the deduced inputs and rejects a bad call."""
    with temporary_operator(compare.operators, compare_operation, _two_spaces) as name:
        metadata = compare.operators[name]
        assert [
            descriptor.identifier for descriptor in metadata.required_resource_inputs
        ] == ["left", "right"]
        function = metadata.function
        assert function is not None
        with pytest.raises(TypeError, match="left"):
            function(parameters=GenericOperatorParameters())
        with pytest.raises(TypeError, match="unexpected"):
            function(
                left=object(),
                right=object(),
                extra=True,
                parameters=GenericOperatorParameters(),
            )


def test_wrapper_resolves_project_context_from_discovery_space(
    tmp_path: pathlib.Path,
    make_stub_discovery_space: Callable,
    temporary_operator: Callable[..., AbstractContextManager[str]],
) -> None:
    """A discovery space supplies project context when operation info omits it."""
    context = ProjectContext(
        project="carried",
        metadataStore=SQLiteStoreConfiguration(path=str(tmp_path / "carried.db")),
    )
    space = make_stub_discovery_space(context)
    space._identifier = "space-carried01"  # noqa: SLF001
    with temporary_operator(
        characterize.operators, characterize_operation, _one_space
    ) as name:
        function = characterize.operators[name].function
        assert function is not None
        with pytest.raises(ValueError, match="carried"):
            function(space, parameters=GenericOperatorParameters())
