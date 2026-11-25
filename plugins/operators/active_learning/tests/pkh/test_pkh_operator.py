# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Regression tests for PKH measurement and completed dataset publication."""

import sys
from collections.abc import Callable
from typing import Literal

import pytest
from active_learning.pkh import operator
from active_learning.pkh.data import (
    candidate_coordinates,
    candidate_entities,
    feature_frame,
    validate_target_output,
)
from active_learning.pkh.parameters import PKHParameters

from ado.core.datacontainer.resource import DataContainerResource
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.operation import OperationOutput
from ado.core.operation.resource import (
    OperationExitStateEnum,
    OperationResource,
    OperationResourceStatus,
)
from ado.schema.entity import Entity
from ado.schema.observed_property import ObservedProperty


@pytest.mark.parametrize(
    ("walk_state", "expected_state", "expected_message"),
    [
        ("fail", OperationExitStateEnum.FAIL, "Child walk finished"),
        ("success", OperationExitStateEnum.ERROR, "budget is incomplete"),
    ],
    ids=["sampling-failure", "publication-failure"],
)
def test_failures_return_child_operation_and_cause(
    categorical_space: DiscoverySpace,
    walk_state: str,
    expected_state: OperationExitStateEnum,
    expected_message: str,
) -> None:
    """PKH retains the child and reports the stage that failed."""
    walk = OperationOutput(
        operation=OperationResource(
            operationType="explore",
            operatorIdentifier="random_walk",
            config={
                "inputs": {"discoverySpace": categorical_space.reference},
                "operation": {},
            },
        ),
        exitStatus=OperationResourceStatus(
            event="finished", exit_state=walk_state, message="Child walk finished"
        ),
    )
    parameters = PKHParameters(targetOutput="self", numberEntities=1)
    target = validate_target_output(parameters.targetOutput, categorical_space)
    result = operator._output_after_sampling(
        walk, categorical_space, parameters, target
    )
    assert result.resources == [walk.operation]
    assert result.exitStatus.exit_state == expected_state
    assert expected_message in result.exitStatus.message


@pytest.mark.parametrize("child_result", ["missing", "sibling", "target"])
def test_publication_uses_the_child_walks_measurement(
    categorical_space: DiscoverySpace,
    record_pkh_measurement: Callable[[Entity, str, ObservedProperty, int], None],
    child_result: Literal["missing", "sibling", "target"],
) -> None:
    """Check that a concurrent measurement cannot supply or replace the child's target."""
    walk = OperationOutput(
        operation=OperationResource(
            operationType="explore",
            operatorIdentifier="random_walk",
            metadata={"entities_submitted": 1},
            config={
                "inputs": {"discoverySpace": categorical_space.reference},
                "operation": {},
            },
        ),
    )
    entity = categorical_space.entitySpace.entity_for_point({"code": None})
    target, _, sibling = categorical_space.measurementSpace.observedProperties
    record_pkh_measurement(entity, "another-walk", target, 999)
    if child_result != "missing":
        record_pkh_measurement(
            entity,
            walk.operation.identifier,
            target if child_result == "target" else sibling,
            17,
        )

    result = operator._output_after_sampling(
        walk,
        categorical_space,
        PKHParameters(targetOutput="self", numberEntities=1),
        target,
    )
    assert result.resources[0] == walk.operation
    if child_result == "target":
        assert result.exitStatus.exit_state == OperationExitStateEnum.SUCCESS
        table = result.resources[1].config.tabularData["predictions"].data
        row = table["entityIdentifier"].index(entity.identifier)
        assert table["self"][row] == 17
        assert table["is_predicted"][row] is False
    else:
        assert result.exitStatus.exit_state == OperationExitStateEnum.ERROR
        expected = (
            "missing measurement results"
            if child_result == "missing"
            else "Missing finite target measurement"
        )
        assert expected in result.exitStatus.message
        assert result.resources == [walk.operation]


def test_published_mixed_coordinates_and_targets(
    categorical_space: DiscoverySpace,
) -> None:
    """Keep distinct model inputs and exact observations alongside a prediction."""
    candidates = candidate_entities(categorical_space)
    frame = feature_frame(categorical_space, candidate_coordinates(candidates))
    assert frame["code"][0] != frame["code"][1]
    target = validate_target_output("self", categorical_space)
    parameters = PKHParameters(targetOutput="self", numberEntities=1, nEstimators=3)
    sampled = {candidates[1].identifier: 2**53 + 1}
    if sys.version_info >= (3, 14):
        with pytest.raises(ValueError, match=r"Python 3\.11 through 3\.13"):
            operator._complete_dataset(categorical_space, parameters, target, sampled)
        return

    dataset, _ = operator._complete_dataset(
        categorical_space, parameters, target, sampled
    )
    stored = DataContainerResource.model_validate_json(dataset.model_dump_json())
    rows = stored.config.tabularData["predictions"].data
    binary_code = {"valueType": "BLOB_VALUE_TYPE", "value": "Y29kZQ=="}
    assert rows["code"] == [1, "1", binary_code, 2**53 + 1, None]
    assert rows["self"][:4] == [0, 2**53 + 1, 2, 3]
    assert rows["is_predicted"] == [False, False, False, False, True]


def test_prediction_budget_counts_only_the_selected_target(
    categorical_space: DiscoverySpace,
) -> None:
    """Exclude the identically named target from another actuator at both stages."""
    parameters = PKHParameters(targetOutput="selected-selected-self", numberEntities=1)
    with pytest.raises(ValueError, match=r"at least 4.*at most 2"):
        operator.pkh.__wrapped__(categorical_space, parameters=parameters)

    target = validate_target_output(parameters.targetOutput, categorical_space)
    with pytest.raises(ValueError, match=r"at least 4.*collected 1"):
        operator._complete_dataset(
            categorical_space, parameters, target, {"stored-number": 10}
        )
    other_lab = categorical_space.entitySpace.entity_for_point({"code": None})
    with pytest.raises(ValueError, match="Missing finite target measurements"):
        operator._complete_dataset(
            categorical_space, parameters, target, {other_lab.identifier: None}
        )
