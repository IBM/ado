# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
"""Tests for operator resource-input conversion helpers."""

from unittest.mock import MagicMock

import pytest

from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.config import (
    DiscoveryOperationEnum,
    DiscoveryOperationResourceConfiguration,
)
from ado.core.operation.context import resolve_operation_project_context
from ado.core.operation.inputs import (
    discovery_spaces_from_inputs,
    iter_resolved_inputs,
    iter_resource_references,
    project_contexts_from_inputs,
    resource_references_from_resolved_inputs,
    resource_references_to_rich_types,
)
from ado.core.resources import ADOResourceReference, CoreResourceKinds
from ado.metastore.project import ProjectContext
from ado.utilities.location import SQLiteStoreConfiguration
from ado.utilities.pydantic import ignore_plugin_validation_context


class _Space(DiscoverySpace):
    """Discovery space stand-in that only carries a project context."""

    def __init__(
        self, project_context: ProjectContext, identifier: str = "space-test"
    ) -> None:
        """Skip parent initialization."""
        self._project_context = project_context
        self._identifier = identifier


def test_iter_resource_references_wraps_a_scalar() -> None:
    """A single reference is yielded as a one-element sequence."""
    reference = ADOResourceReference(
        identifier="space-a", kind=CoreResourceKinds.DISCOVERYSPACE
    )
    assert iter_resource_references(reference) == [reference]


def test_iter_resource_references_returns_a_list() -> None:
    """A list of references is returned unchanged."""
    references = [
        ADOResourceReference(
            identifier="space-a", kind=CoreResourceKinds.DISCOVERYSPACE
        ),
        ADOResourceReference(
            identifier="space-b", kind=CoreResourceKinds.DISCOVERYSPACE
        ),
    ]
    assert iter_resource_references(references) == references


def test_iter_resolved_inputs_flattens_lists() -> None:
    """Resolved list inputs are iterated item by item."""
    context = ProjectContext()
    first = _Space(context, "space-a")
    second = _Space(context, "space-b")
    assert list(iter_resolved_inputs([first, second])) == [first, second]
    assert list(iter_resolved_inputs(first)) == [first]


def test_discovery_spaces_from_inputs_flattens_list_params() -> None:
    """Discovery spaces nested in list parameters are collected."""
    context = ProjectContext()
    first = _Space(context, "space-a")
    second = _Space(context, "space-b")
    third = _Space(context, "space-c")
    spaces = discovery_spaces_from_inputs({"batch": [first, second], "other": third})
    assert spaces == [first, second, third]


def test_project_contexts_from_list_of_spaces() -> None:
    """Each space in a list input contributes a project context."""
    context = ProjectContext()
    carried = project_contexts_from_inputs(
        {"spaces": [_Space(context, "space-a"), _Space(context, "space-b")]}
    )
    assert [name for name, _ in carried] == ["spaces", "spaces"]
    assert all(ctx == context for _, ctx in carried)

    resolved = resolve_operation_project_context(
        None, {"spaces": [_Space(context, "space-a"), _Space(context, "space-b")]}
    )
    assert resolved == context


def test_project_contexts_from_list_rejects_disagreeing_spaces() -> None:
    """Spaces in one list that belong to different projects are rejected."""
    first = ProjectContext()
    second = ProjectContext(
        project="other",
        metadataStore=SQLiteStoreConfiguration(path="other.db"),
    )
    with pytest.raises(ValueError, match="disagree on project context"):
        resolve_operation_project_context(
            None,
            {"spaces": [_Space(first, "space-a"), _Space(second, "space-b")]},
        )


def test_resource_references_from_resolved_inputs_preserves_lists() -> None:
    """Resolved list inputs dump back to lists of references."""
    context = ProjectContext()
    first = _Space(context, "space-a")
    second = _Space(context, "space-b")
    third = _Space(context, "space-c")
    dumped = resource_references_from_resolved_inputs(
        {"batch": [first, second], "other": third}
    )
    assert dumped["batch"] == [first.reference, second.reference]
    assert dumped["other"] == third.reference


def test_resource_references_to_rich_types_resolves_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A list of references is resolved to a list of rich objects."""
    from ado.core.operation import inputs as inputs_module

    resolved_ids: list[str] = []

    def _fake_resolve(reference: ADOResourceReference, metastore: object) -> _Space:
        resolved_ids.append(reference.identifier)
        return _Space(ProjectContext(), reference.identifier)

    monkeypatch.setitem(
        inputs_module._RESOLVERS,
        CoreResourceKinds.DISCOVERYSPACE,
        _fake_resolve,
    )
    result = resource_references_to_rich_types(
        {
            "spaces": [
                ADOResourceReference(
                    identifier="space-a", kind=CoreResourceKinds.DISCOVERYSPACE
                ),
                ADOResourceReference(
                    identifier="space-b", kind=CoreResourceKinds.DISCOVERYSPACE
                ),
            ],
            "other": ADOResourceReference(
                identifier="space-c", kind=CoreResourceKinds.DISCOVERYSPACE
            ),
        },
        metastore=MagicMock(),
    )
    assert resolved_ids == ["space-a", "space-b", "space-c"]
    assert isinstance(result["spaces"], list)
    assert [space.uri for space in result["spaces"]] == ["space-a", "space-b"]
    assert result["other"].uri == "space-c"


def test_operation_inputs_accept_yaml_lists() -> None:
    """Operation YAML may use a list of references for one input name."""
    configuration = DiscoveryOperationResourceConfiguration.model_validate(
        {
            "operation": {
                "module": {
                    "operatorName": "unused",
                    "operationType": DiscoveryOperationEnum.EXPLORE.value,
                }
            },
            "inputs": {
                "containers": [
                    {
                        "identifier": "datacontainer-aaa",
                        "kind": "datacontainer",
                    },
                    {
                        "identifier": "datacontainer-bbb",
                        "kind": "datacontainer",
                    },
                ],
                "baseline": {
                    "identifier": "datacontainer-baseline",
                    "kind": "datacontainer",
                },
            },
        },
        context=ignore_plugin_validation_context,
    )
    assert isinstance(configuration.inputs["containers"], list)
    assert len(configuration.inputs["containers"]) == 2
    assert configuration.inputs["containers"][0].identifier == "datacontainer-aaa"
    assert configuration.inputs["baseline"].identifier == "datacontainer-baseline"


def test_spaces_computed_field_flattens_list_inputs() -> None:
    """The spaces property includes discoveryspace ids nested in list inputs."""
    configuration = DiscoveryOperationResourceConfiguration.model_validate(
        {
            "operation": {
                "module": {
                    "operatorName": "unused",
                    "operationType": DiscoveryOperationEnum.EXPLORE.value,
                }
            },
            "inputs": {
                "spaces": [
                    {"identifier": "space-aaa", "kind": "discoveryspace"},
                    {"identifier": "space-bbb", "kind": "discoveryspace"},
                ],
                "other": {"identifier": "space-ccc", "kind": "discoveryspace"},
            },
        },
        context=ignore_plugin_validation_context,
    )
    assert configuration.spaces == ["space-aaa", "space-bbb", "space-ccc"]
