# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Regression tests for PKH measurements and output protection."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from active_learning.operator import PredictorExport
from active_learning.pkh.data import validate_target_output
from active_learning.pkh.operator import pkh
from active_learning.pkh.parameters import PKHParameters
from active_learning.pkh.sampler import PKHSampleSelector

from ado.core.discoveryspace.space import DiscoverySpace
from ado.schema.measurementspace import MeasurementSpace, MeasurementSpaceConfiguration


def test_rejects_multiple_experiments_before_reserving_output(
    categorical_space: DiscoverySpace, tmp_path: Path
) -> None:
    """An invalid space must fail before PKH claims an output directory."""
    experiment = categorical_space.measurementSpace.experiments[0]
    space = DiscoverySpace(
        entitySpace=categorical_space.entitySpace,
        measurementSpace=MeasurementSpace(
            MeasurementSpaceConfiguration(
                experiments=[
                    experiment,
                    experiment.model_copy(update={"identifier": "other"}),
                ]
            )
        ),
        sample_store=categorical_space.sample_store,
        project_context=categorical_space.project_context,
    )
    (tmp_path / "model_finalized").mkdir()
    parameters = PKHParameters(
        targetOutput="response",
        numberEntities=1,
        outputDirectory=str(tmp_path / "model"),
    )
    with pytest.raises(ValueError, match="exactly 1 experiment"):
        pkh.__wrapped__(space, parameters=parameters)


def test_rejects_target_that_would_overwrite_prediction_flag(
    categorical_space: DiscoverySpace,
) -> None:
    """A measured target must have its own output column."""
    with pytest.raises(ValueError, match="output column names conflict"):
        validate_target_output("is_predicted", categorical_space)


def test_existing_model_is_preserved(
    categorical_space: DiscoverySpace, tmp_path: Path
) -> None:
    """A rerun must leave the previous predictor intact."""
    directory = tmp_path / "model_finalized"
    directory.mkdir()
    model = directory / "model.pkl"
    model.write_bytes(b"previous model")
    parameters = PKHParameters(
        targetOutput="response",
        numberEntities=1,
        outputDirectory=str(tmp_path / "model"),
    )
    with pytest.raises(FileExistsError):
        pkh.__wrapped__(categorical_space, parameters=parameters)
    assert model.read_bytes() == b"previous model"


def test_incomplete_measurements_cannot_publish_a_model(
    categorical_space: DiscoverySpace, tmp_path: Path
) -> None:
    """Prior labels cannot make an incomplete operation appear successful."""
    directory = tmp_path / "model_finalized"
    directory.mkdir()
    parameters = PKHParameters(targetOutput="response", numberEntities=1)
    target = validate_target_output(parameters.targetOutput, categorical_space)
    missing = categorical_space.entitySpace.entity_for_point({"kind": "d", "count": 1})
    with pytest.raises(ValueError, match="Missing finite target measurements"):
        PredictorExport(
            categorical_space,
            target,
            parameters,
            directory,
            {missing.identifier},
            PKHSampleSelector,
        )
    with pytest.raises(ValueError, match="budget is incomplete"):
        PredictorExport(
            categorical_space, target, parameters, directory, set(), PKHSampleSelector
        )
    assert not list(directory.iterdir())


@pytest.mark.asyncio
async def test_sampler_requires_current_target_before_next_entity(
    categorical_space: DiscoverySpace,
) -> None:
    """Only entities without target labels remain eligible for sampling."""

    async def get_space() -> DiscoverySpace:
        """Return the space through the asynchronous manager interface."""
        return categorical_space

    manager = SimpleNamespace(discoverySpace=SimpleNamespace(remote=get_space))
    sampler = PKHSampleSelector(
        PKHParameters(targetOutput="response", numberEntities=2)
    )
    stream = await sampler.remoteEntityIterator(manager)
    [first] = await anext(stream)
    assert first.valueForConstitutivePropertyIdentifier("kind").value == "d"
    with pytest.raises(ValueError, match="Missing finite target measurement"):
        await anext(stream)
