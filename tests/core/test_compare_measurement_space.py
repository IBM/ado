# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
"""Tests for DiscoverySpaceConfiguration.compare_measurement_space()."""

from ado.core.discoveryspace.config import (
    DiscoverySpaceConfiguration,
    MeasurementSpaceComparisonResult,
    ParameterizationMatch,
    SpaceHierarchy,
    VersionMatch,
)
from ado.schema.experiment import Experiment
from ado.schema.measurementspace import MeasurementSpaceConfiguration
from ado.schema.property_value import ConstitutivePropertyValue
from ado.schema.reference import ExperimentReference


def test_measurement_space_comparison_result_lifecycle() -> None:
    """Test full create -> dump -> validate lifecycle for MeasurementSpaceComparisonResult."""
    result = MeasurementSpaceComparisonResult(
        base_experiment_match=SpaceHierarchy.EQUAL_SPACE,
        version_match=VersionMatch.IDENTICAL,
        parameterization_match=ParameterizationMatch.IDENTICAL,
    )
    dumped = result.model_dump()
    loaded = MeasurementSpaceComparisonResult.model_validate(dumped)
    assert loaded == result

    result_disjoint = MeasurementSpaceComparisonResult(
        base_experiment_match=SpaceHierarchy.DISJOINT,
    )
    dumped_disjoint = result_disjoint.model_dump()
    loaded_disjoint = MeasurementSpaceComparisonResult.model_validate(dumped_disjoint)
    assert loaded_disjoint == result_disjoint
    assert loaded_disjoint.version_match is None
    assert loaded_disjoint.parameterization_match is None


def test_compare_identical_spaces(experiment_reference: ExperimentReference) -> None:
    """Identical experiment sets with exact versions and no parameterization."""
    space1 = DiscoverySpaceConfiguration(experiments=[experiment_reference])
    space2 = DiscoverySpaceConfiguration(experiments=[experiment_reference])

    comparison_result = space1.compare_measurement_space(space2)
    assert comparison_result.base_experiment_match == SpaceHierarchy.EQUAL_SPACE
    assert comparison_result.version_match == VersionMatch.IDENTICAL
    assert (
        comparison_result.parameterization_match == ParameterizationMatch.NOT_APPLICABLE
    )


def test_compare_superspace(
    parameterized_references: list[ExperimentReference],
) -> None:
    """Candidate has all experiments of reference plus extra."""
    candidate = DiscoverySpaceConfiguration(experiments=parameterized_references)
    reference = DiscoverySpaceConfiguration(experiments=parameterized_references[:2])

    comparison_result = candidate.compare_measurement_space(reference)
    assert comparison_result.base_experiment_match == SpaceHierarchy.SUPER_SPACE
    assert comparison_result.version_match == VersionMatch.IDENTICAL
    assert comparison_result.parameterization_match == ParameterizationMatch.IDENTICAL


def test_compare_subspace(
    parameterized_references: list[ExperimentReference],
) -> None:
    """Candidate has strict subset of reference experiments."""
    candidate = DiscoverySpaceConfiguration(experiments=[parameterized_references[0]])
    reference = DiscoverySpaceConfiguration(experiments=parameterized_references[:2])

    comparison_result = candidate.compare_measurement_space(reference)
    assert comparison_result.base_experiment_match == SpaceHierarchy.SUB_SPACE
    assert comparison_result.version_match == VersionMatch.IDENTICAL
    assert comparison_result.parameterization_match == ParameterizationMatch.IDENTICAL


def test_compare_overlapping(
    parameterized_references: list[ExperimentReference],
) -> None:
    """Candidate and reference share some experiments but neither is subset of other."""
    candidate = DiscoverySpaceConfiguration(
        experiments=[parameterized_references[0], parameterized_references[2]]
    )
    reference = DiscoverySpaceConfiguration(experiments=parameterized_references[:2])

    comparison_result = candidate.compare_measurement_space(reference)
    assert comparison_result.base_experiment_match == SpaceHierarchy.OVERLAPPING
    assert comparison_result.version_match == VersionMatch.IDENTICAL
    assert comparison_result.parameterization_match == ParameterizationMatch.IDENTICAL


def test_compare_disjoint(
    parameterized_references: list[ExperimentReference],
) -> None:
    """Candidate and reference share no experiments."""
    candidate = DiscoverySpaceConfiguration(experiments=[parameterized_references[0]])
    reference = DiscoverySpaceConfiguration(experiments=[parameterized_references[1]])

    comparison_result = candidate.compare_measurement_space(reference)
    assert comparison_result.base_experiment_match == SpaceHierarchy.DISJOINT
    assert comparison_result.version_match is None
    assert comparison_result.parameterization_match is None


def test_compare_empty_or_none_experiments(
    experiment_reference: ExperimentReference,
) -> None:
    """Empty or None experiments returns DISJOINT."""
    space_with_exp = DiscoverySpaceConfiguration(experiments=[experiment_reference])
    space_empty = DiscoverySpaceConfiguration(experiments=[])
    space_none = DiscoverySpaceConfiguration(experiments=None)

    assert (
        space_empty.compare_measurement_space(space_with_exp).base_experiment_match
        == SpaceHierarchy.DISJOINT
    )
    assert (
        space_with_exp.compare_measurement_space(space_empty).base_experiment_match
        == SpaceHierarchy.DISJOINT
    )
    assert (
        space_none.compare_measurement_space(space_with_exp).base_experiment_match
        == SpaceHierarchy.DISJOINT
    )
    assert (
        space_none.compare_measurement_space(space_empty).base_experiment_match
        == SpaceHierarchy.DISJOINT
    )


def test_version_match_exact_and_major_only(
    experiment_reference: ExperimentReference,
) -> None:
    """Minor/patch version differences result in MAJOR_ONLY."""
    candidate = DiscoverySpaceConfiguration(
        experiments=[
            experiment_reference.model_copy(update={"experimentVersion": "1.2.0"})
        ]
    )
    reference = DiscoverySpaceConfiguration(
        experiments=[
            experiment_reference.model_copy(update={"experimentVersion": "1.0.0"})
        ]
    )

    comparison_result = candidate.compare_measurement_space(reference)
    assert comparison_result.base_experiment_match == SpaceHierarchy.EQUAL_SPACE
    assert comparison_result.version_match == VersionMatch.MAJOR_ONLY


def test_different_major_version_not_paired(
    experiment_reference: ExperimentReference,
) -> None:
    """Experiments with different major versions are not paired (DISJOINT)."""
    candidate = DiscoverySpaceConfiguration(
        experiments=[
            experiment_reference.model_copy(update={"experimentVersion": "2.0.0"})
        ]
    )
    reference = DiscoverySpaceConfiguration(
        experiments=[
            experiment_reference.model_copy(update={"experimentVersion": "1.0.0"})
        ]
    )

    comparison_result = candidate.compare_measurement_space(reference)
    assert comparison_result.base_experiment_match == SpaceHierarchy.DISJOINT
    assert comparison_result.version_match is None
    assert comparison_result.parameterization_match is None


def test_unversioned_experiments_paired(
    experiment_reference: ExperimentReference,
) -> None:
    """Unversioned experiments on both sides are paired and match IDENTICAL."""
    candidate = DiscoverySpaceConfiguration(experiments=[experiment_reference])
    reference = DiscoverySpaceConfiguration(experiments=[experiment_reference])

    comparison_result = candidate.compare_measurement_space(reference)
    assert comparison_result.base_experiment_match == SpaceHierarchy.EQUAL_SPACE
    assert comparison_result.version_match == VersionMatch.IDENTICAL


def test_unversioned_and_versioned_not_paired(
    experiment_reference: ExperimentReference,
) -> None:
    """Unversioned and versioned experiments have different pairing keys and are not paired."""
    candidate = DiscoverySpaceConfiguration(experiments=[experiment_reference])
    reference = DiscoverySpaceConfiguration(
        experiments=[
            experiment_reference.model_copy(update={"experimentVersion": "1.0.0"})
        ]
    )

    comparison_result = candidate.compare_measurement_space(reference)
    assert comparison_result.base_experiment_match == SpaceHierarchy.DISJOINT


def test_different_actuators_not_paired(
    experiment_reference: ExperimentReference,
) -> None:
    """Same experiment id with different actuator identifiers are not paired."""
    candidate = DiscoverySpaceConfiguration(
        experiments=[
            experiment_reference.model_copy(
                update={"actuatorIdentifier": "act1", "experimentVersion": "1.0.0"}
            )
        ]
    )
    reference = DiscoverySpaceConfiguration(
        experiments=[
            experiment_reference.model_copy(
                update={"actuatorIdentifier": "act2", "experimentVersion": "1.0.0"}
            )
        ]
    )

    comparison_result = candidate.compare_measurement_space(reference)
    assert comparison_result.base_experiment_match == SpaceHierarchy.DISJOINT


def test_parameterization_match_identical(
    experiment_reference: ExperimentReference,
    customParameterization: list[ConstitutivePropertyValue],
) -> None:
    """Identical parameterizations report IDENTICAL."""
    candidate_ref = experiment_reference.model_copy(
        update={
            "experimentVersion": "1.0.0",
            "parameterization": customParameterization,
        }
    )
    reference_ref = experiment_reference.model_copy(
        update={
            "experimentVersion": "1.0.0",
            "parameterization": customParameterization,
        }
    )

    candidate = DiscoverySpaceConfiguration(experiments=[candidate_ref])
    reference = DiscoverySpaceConfiguration(experiments=[reference_ref])

    comparison_result = candidate.compare_measurement_space(reference)
    assert comparison_result.base_experiment_match == SpaceHierarchy.EQUAL_SPACE
    assert comparison_result.parameterization_match == ParameterizationMatch.IDENTICAL


def test_parameterization_match_reordered_is_identical(
    experiment_reference: ExperimentReference,
    customParameterization: list[ConstitutivePropertyValue],
) -> None:
    """Parameterizations with same items in different order report IDENTICAL."""
    candidate_ref = ExperimentReference(
        experimentIdentifier=experiment_reference.experimentIdentifier,
        actuatorIdentifier=experiment_reference.actuatorIdentifier,
        experimentVersion="1.0.0",
        parameterization=customParameterization,
    )
    reference_ref = ExperimentReference(
        experimentIdentifier=experiment_reference.experimentIdentifier,
        actuatorIdentifier=experiment_reference.actuatorIdentifier,
        experimentVersion="1.0.0",
        parameterization=list(reversed(customParameterization)),
    )

    candidate = DiscoverySpaceConfiguration(experiments=[candidate_ref])
    reference = DiscoverySpaceConfiguration(experiments=[reference_ref])

    comparison_result = candidate.compare_measurement_space(reference)
    assert comparison_result.base_experiment_match == SpaceHierarchy.EQUAL_SPACE
    assert comparison_result.parameterization_match == ParameterizationMatch.IDENTICAL


def test_parameterization_match_equal_values_of_different_type_is_identical(
    experiment_reference: ExperimentReference,
    customParameterization: list[ConstitutivePropertyValue],
) -> None:
    """Test that parameterizations with equal values of different types report IDENTICAL."""
    int_valued = [
        v.model_copy(update={"value": -1})
        if v.property.identifier == "test_opt2"
        else v
        for v in customParameterization
    ]
    float_valued = [
        v.model_copy(update={"value": -1.0})
        if v.property.identifier == "test_opt2"
        else v
        for v in customParameterization
    ]
    candidate_ref = ExperimentReference(
        experimentIdentifier=experiment_reference.experimentIdentifier,
        actuatorIdentifier=experiment_reference.actuatorIdentifier,
        experimentVersion="1.0.0",
        parameterization=int_valued,
    )
    reference_ref = ExperimentReference(
        experimentIdentifier=experiment_reference.experimentIdentifier,
        actuatorIdentifier=experiment_reference.actuatorIdentifier,
        experimentVersion="1.0.0",
        parameterization=float_valued,
    )

    candidate = DiscoverySpaceConfiguration(experiments=[candidate_ref])
    reference = DiscoverySpaceConfiguration(experiments=[reference_ref])

    comparison_result = candidate.compare_measurement_space(reference)
    assert comparison_result.parameterization_match == ParameterizationMatch.IDENTICAL


def test_parameterization_match_different(
    experiment_reference: ExperimentReference,
    customParameterization: list[ConstitutivePropertyValue],
    defaultParameterization: list[ConstitutivePropertyValue],
) -> None:
    """Different parameterization values report DIFFERENT."""
    candidate_ref = experiment_reference.model_copy(
        update={
            "experimentVersion": "1.0.0",
            "parameterization": customParameterization,
        }
    )
    reference_ref = experiment_reference.model_copy(
        update={
            "experimentVersion": "1.0.0",
            "parameterization": defaultParameterization,
        }
    )

    candidate = DiscoverySpaceConfiguration(experiments=[candidate_ref])
    reference = DiscoverySpaceConfiguration(experiments=[reference_ref])

    comparison_result = candidate.compare_measurement_space(reference)
    assert comparison_result.base_experiment_match == SpaceHierarchy.EQUAL_SPACE
    assert comparison_result.parameterization_match == ParameterizationMatch.DIFFERENT


def test_parameterization_match_one_side_only(
    experiment_reference: ExperimentReference,
    customParameterization: list[ConstitutivePropertyValue],
) -> None:
    """When only one side has parameterization, report DIFFERENT."""
    candidate_ref = experiment_reference.model_copy(
        update={
            "experimentVersion": "1.0.0",
            "parameterization": customParameterization,
        }
    )
    reference_ref = experiment_reference.model_copy(
        update={"experimentVersion": "1.0.0", "parameterization": None}
    )

    candidate = DiscoverySpaceConfiguration(experiments=[candidate_ref])
    reference = DiscoverySpaceConfiguration(experiments=[reference_ref])

    comparison_result = candidate.compare_measurement_space(reference)
    assert comparison_result.base_experiment_match == SpaceHierarchy.EQUAL_SPACE
    assert comparison_result.parameterization_match == ParameterizationMatch.DIFFERENT


def test_aggregation_least_powerful_version(
    experiment_reference: ExperimentReference,
) -> None:
    """One MAJOR_ONLY makes the entire version match MAJOR_ONLY."""
    ref_a = experiment_reference.model_copy(
        update={"experimentIdentifier": "exp1", "experimentVersion": "1.0.0"}
    )
    ref_b = experiment_reference.model_copy(
        update={"experimentIdentifier": "exp2", "experimentVersion": "1.0.0"}
    )
    cand_b = experiment_reference.model_copy(
        update={"experimentIdentifier": "exp2", "experimentVersion": "1.1.0"}
    )

    candidate = DiscoverySpaceConfiguration(experiments=[ref_a, cand_b])
    reference = DiscoverySpaceConfiguration(experiments=[ref_a, ref_b])

    comparison_result = candidate.compare_measurement_space(reference)
    assert comparison_result.base_experiment_match == SpaceHierarchy.EQUAL_SPACE
    assert comparison_result.version_match == VersionMatch.MAJOR_ONLY


def test_aggregation_least_powerful_parameterization(
    experiment_reference: ExperimentReference,
    customParameterization: list[ConstitutivePropertyValue],
    defaultParameterization: list[ConstitutivePropertyValue],
) -> None:
    """One DIFFERENT makes parameterization match DIFFERENT; IDENTICAL + N/A yields IDENTICAL."""
    cand_a = experiment_reference.model_copy(
        update={
            "experimentIdentifier": "exp1",
            "experimentVersion": "1.0.0",
            "parameterization": customParameterization,
        }
    )
    cand_b = experiment_reference.model_copy(
        update={
            "experimentIdentifier": "exp2",
            "experimentVersion": "1.0.0",
            "parameterization": customParameterization,
        }
    )
    ref_a = experiment_reference.model_copy(
        update={
            "experimentIdentifier": "exp1",
            "experimentVersion": "1.0.0",
            "parameterization": customParameterization,
        }
    )
    ref_b = experiment_reference.model_copy(
        update={
            "experimentIdentifier": "exp2",
            "experimentVersion": "1.0.0",
            "parameterization": defaultParameterization,
        }
    )

    # exp1 matches (IDENTICAL), exp2 mismatches (DIFFERENT) -> DIFFERENT
    candidate = DiscoverySpaceConfiguration(experiments=[cand_a, cand_b])
    reference = DiscoverySpaceConfiguration(experiments=[ref_a, ref_b])

    comparison_result = candidate.compare_measurement_space(reference)
    assert comparison_result.parameterization_match == ParameterizationMatch.DIFFERENT

    # exp1 matches (IDENTICAL), exp3 unparameterized (N/A) -> IDENTICAL
    cand_c = experiment_reference.model_copy(
        update={"experimentIdentifier": "exp3", "experimentVersion": "1.0.0"}
    )
    ref_c = experiment_reference.model_copy(
        update={"experimentIdentifier": "exp3", "experimentVersion": "1.0.0"}
    )

    candidate_mixed = DiscoverySpaceConfiguration(experiments=[cand_a, cand_c])
    reference_mixed = DiscoverySpaceConfiguration(experiments=[ref_a, ref_c])

    comparison_result_mixed = candidate_mixed.compare_measurement_space(reference_mixed)
    assert (
        comparison_result_mixed.parameterization_match
        == ParameterizationMatch.IDENTICAL
    )


def test_measurement_space_configuration_form(
    mock_parameterizable_experiment: Experiment,
) -> None:
    """compare_measurement_space works when experiments is MeasurementSpaceConfiguration."""
    candidate_measurement_space = MeasurementSpaceConfiguration(
        experiments=[mock_parameterizable_experiment]
    )
    candidate = DiscoverySpaceConfiguration(experiments=candidate_measurement_space)
    reference = DiscoverySpaceConfiguration(
        experiments=[mock_parameterizable_experiment.reference]
    )

    comparison_result = candidate.compare_measurement_space(reference)
    assert comparison_result.base_experiment_match == SpaceHierarchy.EQUAL_SPACE
    assert comparison_result.version_match == VersionMatch.IDENTICAL
    assert (
        comparison_result.parameterization_match == ParameterizationMatch.NOT_APPLICABLE
    )
