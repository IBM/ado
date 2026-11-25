# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Regression tests for PKH measurement and completed dataset publication."""

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from active_learning.pkh import operator
from active_learning.pkh.data import measured_labels, validate_target_output
from active_learning.pkh.parameters import PKHParameters

from ado.core.datacontainer.resource import DataContainerResource
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.operation import OperationOutput
from ado.core.operation.resource import OperationResource, OperationResourceStatus
from ado.schema.domain import PropertyDomain
from ado.schema.measurementspace import MeasurementSpace, MeasurementSpaceConfiguration
from ado.schema.property import ConstitutiveProperty


def test_rejects_multiple_experiments_before_sampling(
    empty_pkh_space: DiscoverySpace,
) -> None:
    """PKH validates the experiment count before starting a measurement run."""
    experiment = empty_pkh_space.measurementSpace.experiments[0]
    space = DiscoverySpace(
        entitySpace=empty_pkh_space.entitySpace,
        measurementSpace=MeasurementSpace(
            MeasurementSpaceConfiguration(
                experiments=[
                    experiment,
                    experiment.model_copy(update={"identifier": "other"}),
                ]
            )
        ),
        sample_store=empty_pkh_space.sample_store,
        project_context=empty_pkh_space.project_context,
    )
    parameters = PKHParameters(targetOutput="self", numberEntities=1)
    with pytest.raises(ValueError, match="exactly 1 experiment"):
        operator.pkh.__wrapped__(space, parameters=parameters)


@pytest.mark.parametrize(
    ("target", "error"),
    [
        ("is_predicted", "output column names conflict"),
        ("selected-self", "must identify exactly one observed property"),
    ],
)
def test_rejects_unusable_target_names(
    empty_pkh_space: DiscoverySpace, target: str, error: str
) -> None:
    """PKH requires one identifiable target with its own output column."""
    with pytest.raises(ValueError, match=error):
        validate_target_output(target, empty_pkh_space)


def test_failed_sampling_preserves_status_and_skips_publication(
    empty_pkh_space: DiscoverySpace,
) -> None:
    """A failed walk retains its status and is the only resource PKH returns."""
    walk = OperationOutput(
        operation=OperationResource(
            operationType="explore",
            operatorIdentifier="random_walk",
            config={
                "inputs": {"discoverySpace": empty_pkh_space.reference},
                "operation": {},
            },
        ),
        exitStatus=OperationResourceStatus(
            event="finished", exit_state="fail", message="Measurement run failed"
        ),
    )
    parameters = PKHParameters(targetOutput="self", numberEntities=1)
    target = validate_target_output(parameters.targetOutput, empty_pkh_space)
    result = operator._output_after_sampling(walk, empty_pkh_space, parameters, target)
    assert result.exitStatus == walk.exitStatus
    assert result.resources == [walk.operation]


def test_completed_dataset_preserves_mixed_categories(
    categorical_space: DiscoverySpace,
    tmp_path: Path,
) -> None:
    """ADO can round-trip each candidate, its provenance, and its finite target."""
    target = validate_target_output("self", categorical_space)
    original_labels = measured_labels(categorical_space, target)
    sampled = categorical_space.entitySpace.entity_for_point(
        {"kind": "b", "count": 2**53 + 1}
    )
    parameters = PKHParameters(
        targetOutput="self",
        numberEntities=1,
        nEstimators=3,
        outputDirectory=str(tmp_path / "previous_model"),
    )
    if sys.version_info >= (3, 14):
        with pytest.raises(ValueError, match=r"Python 3\.11 through 3\.13"):
            operator._complete_dataset(
                categorical_space, parameters, target, {sampled.identifier}
            )
        return
    resource, mmd_squared = operator._complete_dataset(
        categorical_space, parameters, target, {sampled.identifier}
    )
    stored = DataContainerResource.model_validate_json(resource.model_dump_json())
    assert stored.config.metadata.labels["predictor"] == "RandomForest_FULL"
    table = stored.config.tabularData["predictions"].dataframe()
    assert len(table) == 20
    assert table.entityIdentifier.is_unique
    assert (
        table.loc[~table.is_predicted].set_index("entityIdentifier")["self"].to_dict()
        == original_labels
    )
    unmeasured = table.loc[table.is_predicted]
    assert unmeasured.kind.tolist() == ["c"] * 5 + ["d"] * 5
    raw_counts = [0.5, 2**53, 2**53 + 1, None, b"code"] * 2
    assert unmeasured["count"].tolist() == [str(value) for value in raw_counts]
    assert np.isfinite(unmeasured["self"]).all()
    assert np.isfinite(mmd_squared)
    assert measured_labels(categorical_space, target) == original_labels
    assert not list(tmp_path.glob("previous_model*"))


@pytest.mark.parametrize(
    ("codes", "expected_labels", "expected_mmd"),
    [
        ([2**53], [1], None),
        ([2**53, 2**53 + 1], [1, 2], 0.0),
    ],
)
def test_fully_measured_small_space_publishes_existing_targets(
    categorical_space: DiscoverySpace,
    make_pkh_space: Callable[[list[ConstitutiveProperty]], DiscoverySpace],
    codes: list[int],
    expected_labels: list[int],
    expected_mmd: float | None,
) -> None:
    """PKH publishes all measured targets even on grids too small to fit AutoGluon."""
    kind, count = categorical_space.entitySpace.constitutiveProperties
    space = make_pkh_space(
        [
            kind.model_copy(update={"propertyDomain": PropertyDomain(values=["b"])}),
            count.model_copy(update={"propertyDomain": PropertyDomain(values=codes)}),
        ]
    )
    parameters = PKHParameters(
        targetOutput="self", numberEntities=1, finalPredictor="autogluon"
    )
    target = validate_target_output(parameters.targetOutput, space)
    resource, mmd_squared = operator._complete_dataset(space, parameters, target, set())
    table = resource.config.tabularData["predictions"].dataframe()
    assert table["self"].tolist() == expected_labels
    assert not table.is_predicted.any()
    assert "predictor" not in (resource.config.metadata.labels or {})
    assert mmd_squared == expected_mmd


def test_incomplete_measurements_cannot_publish_a_dataset(
    categorical_space: DiscoverySpace,
) -> None:
    """PKH checks newly measured entities before publishing their targets."""
    parameters = PKHParameters(targetOutput="self", numberEntities=1)
    target = validate_target_output(parameters.targetOutput, categorical_space)
    missing = categorical_space.entitySpace.entity_for_point(
        {"kind": "d", "count": 2**53 + 1}
    )
    with pytest.raises(ValueError, match="Missing finite target measurements"):
        operator._complete_dataset(
            categorical_space, parameters, target, {missing.identifier}
        )
    with pytest.raises(ValueError, match="budget is incomplete"):
        operator._complete_dataset(categorical_space, parameters, target, set())


@pytest.mark.parametrize("predictor", ["random_forest", "autogluon"])
def test_checks_attainable_and_collected_label_counts(
    categorical_space: DiscoverySpace,
    predictor: str,
) -> None:
    """Each final predictor needs enough labels if unmeasured candidates remain."""
    parameters = PKHParameters(
        targetOutput="selected-selected-self",
        numberEntities=1,
        finalPredictor=predictor,
    )
    with pytest.raises(ValueError, match=r"at least 4.*at most 2"):
        operator.pkh.__wrapped__(categorical_space, parameters=parameters)
    sampled = categorical_space.entitySpace.entity_for_point({"kind": 1, "count": 0.5})
    target = validate_target_output(parameters.targetOutput, categorical_space)
    with pytest.raises(ValueError, match=r"at least 4.*collected 1"):
        operator._complete_dataset(
            categorical_space, parameters, target, {sampled.identifier}
        )
