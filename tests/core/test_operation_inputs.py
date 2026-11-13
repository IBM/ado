# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
"""Tests for operator resource-input conversion helpers."""

from collections.abc import Callable

import pytest

from ado.core.datacontainer.resource import DataContainer, DataContainerResource
from ado.core.operation.config import (
    DiscoveryOperationEnum,
    DiscoveryOperationResourceConfiguration,
)
from ado.core.operation.inputs import (
    discovery_spaces_from_inputs,
    project_contexts_from_inputs,
    resolve_operation_project_context,
    resource_references_from_resolved_inputs,
    resource_references_to_rich_types,
    verify_inputs_exist_in_metastore,
)
from ado.metastore.base import ResourcesDoNotExistError
from ado.metastore.project import ProjectContext
from ado.metastore.sqlstore import SQLStore
from ado.utilities.location import SQLiteStoreConfiguration
from ado.utilities.pydantic import ignore_plugin_validation_context


def _data_container_resource(value: str) -> DataContainerResource:
    """Return a data container resource holding one key-value pair."""
    return DataContainerResource(config=DataContainer(data={"key": {"value": value}}))


def test_discovery_spaces_from_inputs_flattens_list_params(
    make_stub_discovery_space: Callable,
) -> None:
    """Discovery spaces nested in list parameters are collected."""
    first = make_stub_discovery_space(identifier="space-a")
    second = make_stub_discovery_space(identifier="space-b")
    third = make_stub_discovery_space(identifier="space-c")
    spaces = discovery_spaces_from_inputs({"batch": [first, second], "other": third})
    assert spaces == [first, second, third]


def test_project_contexts_from_list_of_spaces(
    make_stub_discovery_space: Callable,
) -> None:
    """Each space in a list input contributes a project context."""
    context = ProjectContext()
    inputs = {
        "spaces": [
            make_stub_discovery_space(context, "space-a"),
            make_stub_discovery_space(context, "space-b"),
        ]
    }
    carried = project_contexts_from_inputs(inputs)
    assert [name for name, _ in carried] == ["spaces", "spaces"]
    assert all(ctx == context for _, ctx in carried)

    assert resolve_operation_project_context(None, inputs) == context


def test_project_contexts_from_list_rejects_disagreeing_spaces(
    make_stub_discovery_space: Callable,
) -> None:
    """Spaces in one list that belong to different projects are rejected."""
    first = ProjectContext()
    second = ProjectContext(
        project="other",
        metadataStore=SQLiteStoreConfiguration(path="other.db"),
    )
    with pytest.raises(ValueError, match="disagree on project context"):
        resolve_operation_project_context(
            None,
            {
                "spaces": [
                    make_stub_discovery_space(first, "space-a"),
                    make_stub_discovery_space(second, "space-b"),
                ]
            },
        )


def test_resource_references_from_resolved_inputs_preserves_lists(
    make_stub_discovery_space: Callable,
) -> None:
    """Resolved list inputs dump back to lists of references."""
    first = make_stub_discovery_space(identifier="space-a")
    second = make_stub_discovery_space(identifier="space-b")
    third = make_stub_discovery_space(identifier="space-c")
    dumped = resource_references_from_resolved_inputs(
        {"batch": [first, second], "other": third}
    )
    assert dumped["batch"] == [first.reference, second.reference]
    assert dumped["other"] == third.reference


def test_resource_references_to_rich_types_resolves_lists(
    sql_store: SQLStore,
    create_resources: Callable,
) -> None:
    """A list of references is resolved to a list of stored resources."""
    listed = [_data_container_resource("one"), _data_container_resource("two")]
    scalar = _data_container_resource("three")
    create_resources([*listed, scalar])

    result = resource_references_to_rich_types(
        {
            "containers": [resource.reference for resource in listed],
            "baseline": scalar.reference,
        },
        metastore=sql_store,
    )

    assert isinstance(result["containers"], list)
    assert [resource.identifier for resource in result["containers"]] == [
        resource.identifier for resource in listed
    ]
    assert result["baseline"].identifier == scalar.identifier


def test_verify_inputs_exist_in_metastore_checks_list_members(
    sql_store: SQLStore,
    create_resources: Callable,
) -> None:
    """A list member missing from the metastore is reported."""
    stored = _data_container_resource("one")
    missing = _data_container_resource("two")
    create_resources([stored])

    with pytest.raises(ResourcesDoNotExistError, match=missing.identifier):
        verify_inputs_exist_in_metastore(
            {"containers": [stored, missing], "baseline": stored}, sql_store
        )


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
