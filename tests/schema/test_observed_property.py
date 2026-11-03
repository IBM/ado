# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pytest

from ado.schema.experiment import Experiment
from ado.schema.observed_property import ObservedProperty
from ado.schema.property import AbstractPropertyDescriptor
from ado.schema.property_value import ConstitutivePropertyValue
from ado.schema.reference import ExperimentReference


def test_observed_property_equal_reversed_parameterization(
    mock_parameterizable_experiment: Experiment,
    customParameterization: list[ConstitutivePropertyValue],
) -> None:
    """Parameterization order does not affect equality or hashing."""
    target = AbstractPropertyDescriptor(
        identifier=mock_parameterizable_experiment.observedProperties[
            0
        ].targetProperty.identifier
    )
    ref_forward = ExperimentReference(
        actuatorIdentifier=mock_parameterizable_experiment.actuatorIdentifier,
        experimentIdentifier=mock_parameterizable_experiment.identifier,
        parameterization=customParameterization,
    )
    ref_reversed = ExperimentReference(
        actuatorIdentifier=mock_parameterizable_experiment.actuatorIdentifier,
        experimentIdentifier=mock_parameterizable_experiment.identifier,
        parameterization=list(reversed(customParameterization)),
    )
    op_forward = ObservedProperty(
        targetProperty=target, experimentReference=ref_forward
    )
    op_reversed = ObservedProperty(
        targetProperty=target, experimentReference=ref_reversed
    )

    assert op_forward == op_reversed
    assert hash(op_forward) == hash(op_reversed)
    assert len({op_forward, op_reversed}) == 1


def test_observed_property_not_equal_different_target(
    mock_parameterizable_experiment: Experiment,
    customParameterization: list[ConstitutivePropertyValue],
) -> None:
    """ObservedProperties with different target property identifiers are not equal."""
    ref = ExperimentReference(
        actuatorIdentifier=mock_parameterizable_experiment.actuatorIdentifier,
        experimentIdentifier=mock_parameterizable_experiment.identifier,
        parameterization=customParameterization,
    )
    op1 = ObservedProperty(
        targetProperty=AbstractPropertyDescriptor(identifier="prop_a"),
        experimentReference=ref,
    )
    op2 = ObservedProperty(
        targetProperty=AbstractPropertyDescriptor(identifier="prop_b"),
        experimentReference=ref,
    )

    assert op1 != op2


def test_observed_property_not_equal_different_parameterization(
    mock_parameterizable_experiment: Experiment,
    customParameterization: list[ConstitutivePropertyValue],
) -> None:
    """ObservedProperties with different parameterizations are not equal."""
    target = AbstractPropertyDescriptor(
        identifier=mock_parameterizable_experiment.observedProperties[
            0
        ].targetProperty.identifier
    )
    op_full = ObservedProperty(
        targetProperty=target,
        experimentReference=ExperimentReference(
            actuatorIdentifier=mock_parameterizable_experiment.actuatorIdentifier,
            experimentIdentifier=mock_parameterizable_experiment.identifier,
            parameterization=customParameterization,
        ),
    )
    op_partial = ObservedProperty(
        targetProperty=target,
        experimentReference=ExperimentReference(
            actuatorIdentifier=mock_parameterizable_experiment.actuatorIdentifier,
            experimentIdentifier=mock_parameterizable_experiment.identifier,
            parameterization=customParameterization[:1],
        ),
    )

    assert op_full != op_partial


def test_observed_property_not_equal_parameterized_vs_unparameterized(
    mock_parameterizable_experiment: Experiment,
) -> None:
    """An ObservedProperty with parameterization differs from one without."""
    target = AbstractPropertyDescriptor(
        identifier=mock_parameterizable_experiment.observedProperties[
            0
        ].targetProperty.identifier
    )
    op_parameterized = ObservedProperty(
        targetProperty=target,
        experimentReference=ExperimentReference(
            actuatorIdentifier=mock_parameterizable_experiment.actuatorIdentifier,
            experimentIdentifier=mock_parameterizable_experiment.identifier,
            parameterization=[
                mock_parameterizable_experiment.defaultParameterization[0]
            ],
        ),
    )
    op_base = ObservedProperty(
        targetProperty=target,
        experimentReference=ExperimentReference(
            actuatorIdentifier=mock_parameterizable_experiment.actuatorIdentifier,
            experimentIdentifier=mock_parameterizable_experiment.identifier,
        ),
    )

    assert op_parameterized != op_base


@pytest.mark.parametrize("other", ["not-an-observed-property", 42, None])
def test_observed_property_not_equal_non_observed_property(
    observed_property: ObservedProperty,
    other: object,
) -> None:
    """An ObservedProperty is never equal to a non-ObservedProperty object."""
    assert observed_property != other


def test_observed_property_hashable(
    observed_property: ObservedProperty,
) -> None:
    """An ObservedProperty can be used as a dictionary key."""
    d = {observed_property: "some_key"}
    assert d
