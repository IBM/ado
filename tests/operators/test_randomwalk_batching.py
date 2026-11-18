# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from ado.modules.operators.randomwalk import EntitySubmissionTracker
from ado.schema.domain import PropertyDomain, VariableTypeEnum
from ado.schema.experiment import Experiment, ParameterizedExperiment
from ado.schema.measurementspace import MeasurementSpace, MeasurementSpaceConfiguration
from ado.schema.property import (
    AbstractPropertyDescriptor,
    ConstitutiveProperty,
    ConstitutivePropertyDescriptor,
)
from ado.schema.property_value import ConstitutivePropertyValue

_ACTUATOR = "test_actuator"


def _experiment(identifier: str, *requires: Experiment) -> Experiment:
    """Build an experiment that may require observed properties from requires.

    Args:
        identifier: Experiment identifier.
        requires: Experiments whose first observed property is a required input.

    Returns:
        An experiment with one target property.
    """

    return Experiment(
        actuatorIdentifier=_ACTUATOR,
        identifier=identifier,
        targetProperties=[
            AbstractPropertyDescriptor(identifier=f"{identifier}_output")
        ],
        requiredProperties=tuple(
            requirement.observedProperties[0] for requirement in requires
        ),
        version="1.0.0",
    )


def _space(*experiments: Experiment) -> MeasurementSpace:
    """Build a measurement space containing experiments.

    Args:
        experiments: Experiments in the space.

    Returns:
        A measurement space for those experiments.
    """

    return MeasurementSpace(
        configuration=MeasurementSpaceConfiguration(experiments=list(experiments))
    )


def _parameterized_provider(value: str) -> ParameterizedExperiment:
    """Build a parameterization of the shared provider experiment.

    Args:
        value: Non-default value of the ``level`` optional property.

    Returns:
        A parameterized provider experiment.
    """

    level = ConstitutiveProperty(
        identifier="level",
        propertyDomain=PropertyDomain(
            variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
            values=["low", "mid", "high"],
        ),
    )
    return ParameterizedExperiment(
        actuatorIdentifier=_ACTUATOR,
        identifier="provider",
        targetProperties=[AbstractPropertyDescriptor(identifier="bound")],
        optionalProperties=(level,),
        defaultParameterization=(
            ConstitutivePropertyValue(
                value="low",
                property=ConstitutivePropertyDescriptor(identifier="level"),
            ),
        ),
        parameterization=[
            ConstitutivePropertyValue(
                value=value,
                property=ConstitutivePropertyDescriptor(identifier="level"),
            )
        ],
        version="1.0.0",
    )


def test_independent_experiments_free_the_slot_once_submitted() -> None:
    """Three submitted independent experiments free a batch-size-1 slot before any finish."""

    experiments = [_experiment("a"), _experiment("b"), _experiment("c")]
    space = _space(*experiments)
    tracker = EntitySubmissionTracker()

    tracker.note_submitted(0, experiments[0].reference)
    tracker.note_submitted(0, experiments[1].reference)
    assert not tracker.has_room(batch_size=1, measurement_space=space)

    tracker.note_submitted(0, experiments[2].reference)
    assert tracker.has_room(batch_size=1, measurement_space=space)


def test_single_dependent_holds_the_slot_until_it_is_submitted() -> None:
    """A dependent of the only submitted experiment keeps the slot until it is submitted."""

    prerequisite = _experiment("a")
    dependent = _experiment("b", prerequisite)
    space = _space(prerequisite, dependent)
    tracker = EntitySubmissionTracker()

    tracker.note_submitted(0, prerequisite.reference)
    assert not tracker.has_room(batch_size=1, measurement_space=space)

    tracker.note_finished(0, prerequisite.reference)
    tracker.note_queued(0, dependent.reference)
    assert not tracker.has_room(batch_size=1, measurement_space=space)

    tracker.note_submitted(0, dependent.reference)
    assert tracker.has_room(batch_size=1, measurement_space=space)


def test_dependent_blocked_on_inflight_prerequisite_holds_the_slot() -> None:
    """Finishing an unrelated experiment leaves the slot occupied while the prerequisite is in flight."""

    finished = _experiment("a")
    prerequisite = _experiment("b")
    dependent = _experiment("c", prerequisite)
    space = _space(finished, prerequisite, dependent)
    tracker = EntitySubmissionTracker()

    tracker.note_submitted(0, finished.reference)
    tracker.note_submitted(0, prerequisite.reference)
    tracker.note_finished(0, finished.reference)
    assert not tracker.has_room(batch_size=1, measurement_space=space)

    tracker.note_finished(0, prerequisite.reference)
    tracker.note_queued(0, dependent.reference)
    assert not tracker.has_room(batch_size=1, measurement_space=space)

    tracker.note_submitted(0, dependent.reference)
    assert tracker.has_room(batch_size=1, measurement_space=space)


def test_dependent_on_two_providers_holds_the_slot_until_both_finish() -> None:
    """A dependent of two experiments keeps the slot when only one provider has finished."""

    first = _experiment("a")
    second = _experiment("b")
    dependent = _experiment("c", first, second)
    space = _space(first, second, dependent)
    tracker = EntitySubmissionTracker()

    tracker.note_submitted(0, first.reference)
    tracker.note_submitted(0, second.reference)
    tracker.note_finished(0, first.reference)
    assert not tracker.has_room(batch_size=1, measurement_space=space)

    tracker.note_finished(0, second.reference)
    tracker.note_queued(0, dependent.reference)
    assert not tracker.has_room(batch_size=1, measurement_space=space)

    tracker.note_submitted(0, dependent.reference)
    assert tracker.has_room(batch_size=1, measurement_space=space)


def test_chain_holds_the_slot_until_the_last_experiment_is_submitted() -> None:
    """A chain stays open from the first submission until the last experiment is submitted."""

    first = _experiment("a")
    second = _experiment("b", first)
    third = _experiment("c", second)
    space = _space(first, second, third)
    tracker = EntitySubmissionTracker()

    tracker.note_submitted(0, first.reference)
    assert not tracker.has_room(batch_size=1, measurement_space=space)

    tracker.note_finished(0, first.reference)
    tracker.note_queued(0, second.reference)
    assert not tracker.has_room(batch_size=1, measurement_space=space)

    tracker.note_submitted(0, second.reference)
    assert not tracker.has_room(batch_size=1, measurement_space=space)

    tracker.note_finished(0, second.reference)
    tracker.note_queued(0, third.reference)
    assert not tracker.has_room(batch_size=1, measurement_space=space)

    tracker.note_submitted(0, third.reference)
    assert tracker.has_room(batch_size=1, measurement_space=space)


def test_diamond_holds_the_slot_while_the_join_is_blocked() -> None:
    """The join experiment keeps the slot after both branches are in flight."""

    source = _experiment("a")
    extra = _experiment("e")
    left = _experiment("b", source)
    right = _experiment("c", source)
    join = _experiment("d", left, right)
    space = _space(source, extra, left, right, join)
    tracker = EntitySubmissionTracker()

    tracker.note_submitted(0, source.reference)
    tracker.note_submitted(0, extra.reference)
    tracker.note_finished(0, source.reference)
    tracker.note_submitted(0, left.reference)
    tracker.note_submitted(0, right.reference)
    tracker.note_finished(0, extra.reference)
    assert not tracker.has_room(batch_size=1, measurement_space=space)

    tracker.note_finished(0, left.reference)
    tracker.note_finished(0, right.reference)
    tracker.note_queued(0, join.reference)
    assert not tracker.has_room(batch_size=1, measurement_space=space)

    tracker.note_submitted(0, join.reference)
    assert tracker.has_room(batch_size=1, measurement_space=space)


def test_terminal_failure_frees_the_slot_when_the_dependent_cannot_run() -> None:
    """A failed prerequisite that never queues its dependent frees the slot."""

    prerequisite = _experiment("b")
    dependent = _experiment("c", prerequisite)
    space = _space(prerequisite, dependent)
    tracker = EntitySubmissionTracker()

    tracker.note_submitted(0, prerequisite.reference)
    tracker.note_finished(0, prerequisite.reference)
    assert tracker.has_room(batch_size=1, measurement_space=space)


def test_batch_size_two_frees_one_slot_when_one_entity_is_fully_submitted() -> None:
    """Two blocked entities fill the batch, and submitting one dependent frees one slot."""

    finished = _experiment("a")
    prerequisite = _experiment("b")
    dependent = _experiment("c", prerequisite)
    space = _space(finished, prerequisite, dependent)
    tracker = EntitySubmissionTracker()

    for request_index in (0, 1):
        tracker.note_submitted(request_index, finished.reference)
        tracker.note_submitted(request_index, prerequisite.reference)
    assert not tracker.has_room(batch_size=2, measurement_space=space)

    tracker.note_submitted(0, dependent.reference)
    assert tracker.has_room(batch_size=2, measurement_space=space)


def test_second_parameterization_keeps_the_dependent_reachable_until_both_finish() -> (
    None
):
    """A dependent stays reachable until every parameterization of its provider has finished.

    Submitting the dependent frees the slot while another parameterization is in flight.
    """

    mid = _parameterized_provider("mid")
    high = _parameterized_provider("high")
    provider = Experiment(
        actuatorIdentifier=_ACTUATOR,
        identifier="provider",
        targetProperties=[AbstractPropertyDescriptor(identifier="bound")],
        optionalProperties=mid.optionalProperties,
        defaultParameterization=mid.defaultParameterization,
        version="1.0.0",
    )
    dependent = _experiment("analyze", provider)
    space = _space(mid, high, dependent)

    waiting_on_both = EntitySubmissionTracker()
    waiting_on_both.note_submitted(0, mid.reference)
    waiting_on_both.note_submitted(0, high.reference)
    waiting_on_both.note_finished(0, mid.reference)
    assert not waiting_on_both.has_room(batch_size=1, measurement_space=space)
    waiting_on_both.note_finished(0, high.reference)
    assert waiting_on_both.has_room(batch_size=1, measurement_space=space)

    submitted_early = EntitySubmissionTracker()
    submitted_early.note_submitted(0, mid.reference)
    submitted_early.note_submitted(0, high.reference)
    submitted_early.note_finished(0, mid.reference)
    submitted_early.note_submitted(0, dependent.reference)
    assert submitted_early.has_room(batch_size=1, measurement_space=space)


def test_record_submission_with_no_request_finishes_the_experiment() -> None:
    """An experiment that creates no measurement request leaves the slot free."""

    experiment = _experiment("a")
    space = _space(experiment)
    tracker = EntitySubmissionTracker()

    tracker.record_submission(
        request_index=0,
        experiment_reference=experiment.reference,
        request_ids=[],
    )
    assert tracker.has_room(batch_size=1, measurement_space=space)
