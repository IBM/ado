# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for experiment interface compatibility checks in orchestrator.schema.experiment."""

from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum
from orchestrator.schema.experiment import (
    Experiment,
    ExperimentInterfaceIssueKind,
    ParameterizedExperiment,
    check_experiment_interface_compatible,
)
from orchestrator.schema.property import (
    AbstractPropertyDescriptor,
    ConstitutiveProperty,
    ConstitutivePropertyDescriptor,
)
from orchestrator.schema.property_value import ConstitutivePropertyValue


def test_compatible_experiments_return_no_issues(
    mock_parameterizable_experiment: Experiment,
) -> None:
    """Identical expected and provided experiments are compatible."""
    issues = check_experiment_interface_compatible(
        expected_experiment=mock_parameterizable_experiment,
        provided_experiment=mock_parameterizable_experiment,
    )
    assert issues == []


def test_missing_required_constitutive_input_in_provided(
    mock_parameterizable_experiment: Experiment,
    requiredProperties: list[ConstitutiveProperty],
) -> None:
    """Expected required inputs must exist in the provided experiment."""
    provided_experiment = mock_parameterizable_experiment.model_copy(
        update={
            "requiredProperties": tuple(requiredProperties[1:]),
        }
    )
    issues = check_experiment_interface_compatible(
        expected_experiment=mock_parameterizable_experiment,
        provided_experiment=provided_experiment,
    )
    assert any(
        issue.kind
        == ExperimentInterfaceIssueKind.MISSING_REQUIRED_CONSTITUTIVE_IN_PROVIDED
        and issue.identifier == "test_req1"
        for issue in issues
    )


def test_extra_required_constitutive_input_in_provided(
    mock_parameterizable_experiment: Experiment,
    requiredProperties: list[ConstitutiveProperty],
) -> None:
    """Provided-only required inputs must also be required in the expected experiment."""
    extra_required = ConstitutiveProperty(identifier="provided_only_req")
    expected_experiment = mock_parameterizable_experiment.model_copy(
        update={
            "requiredProperties": tuple(requiredProperties),
        }
    )
    provided_experiment = mock_parameterizable_experiment.model_copy(
        update={
            "requiredProperties": (*requiredProperties, extra_required),
        }
    )
    issues = check_experiment_interface_compatible(
        expected_experiment=expected_experiment,
        provided_experiment=provided_experiment,
    )
    assert any(
        issue.kind
        == ExperimentInterfaceIssueKind.EXTRA_REQUIRED_CONSTITUTIVE_IN_PROVIDED
        and issue.identifier == "provided_only_req"
        for issue in issues
    )


def test_missing_required_observed_input_in_provided(
    mock_parameterizable_experiment_with_required_observed: Experiment,
    mock_parameterizable_experiment: Experiment,
) -> None:
    """Expected required observed inputs must exist in the provided experiment."""
    observed_input = mock_parameterizable_experiment.observedProperties[0]
    provided_experiment = mock_parameterizable_experiment_with_required_observed.model_copy(
        update={
            "requiredProperties": tuple(
                mock_parameterizable_experiment_with_required_observed.requiredConstitutiveProperties
            ),
        }
    )
    expected_experiment = mock_parameterizable_experiment_with_required_observed
    issues = check_experiment_interface_compatible(
        expected_experiment=expected_experiment,
        provided_experiment=provided_experiment,
    )
    assert any(
        issue.kind == ExperimentInterfaceIssueKind.MISSING_REQUIRED_OBSERVED_IN_PROVIDED
        and issue.identifier == observed_input.identifier
        for issue in issues
    )


def test_extra_required_observed_input_in_provided(
    mock_parameterizable_experiment_with_required_observed: Experiment,
    mock_parameterizable_experiment: Experiment,
) -> None:
    """Provided-only required observed inputs must also be required in the expected."""
    observed_input = mock_parameterizable_experiment.observedProperties[0]
    expected_experiment = mock_parameterizable_experiment.model_copy(
        update={
            "targetProperties": [
                AbstractPropertyDescriptor(identifier="measurable_three")
            ],
        }
    )
    provided_experiment = mock_parameterizable_experiment_with_required_observed
    issues = check_experiment_interface_compatible(
        expected_experiment=expected_experiment,
        provided_experiment=provided_experiment,
    )
    assert any(
        issue.kind == ExperimentInterfaceIssueKind.EXTRA_REQUIRED_OBSERVED_IN_PROVIDED
        and issue.identifier == observed_input.identifier
        for issue in issues
    )


def test_incompatible_constitutive_domain(
    mock_parameterizable_experiment: Experiment,
    requiredProperties: list[ConstitutiveProperty],
) -> None:
    """Expected property domains must be subdomains of provided property domains."""
    narrowed_provided_required = requiredProperties[0].model_copy(
        update={
            "propertyDomain": PropertyDomain(
                variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
                values=["X"],
            )
        }
    )
    provided_experiment = mock_parameterizable_experiment.model_copy(
        update={
            "requiredProperties": (
                narrowed_provided_required,
                *requiredProperties[1:],
            ),
        }
    )
    issues = check_experiment_interface_compatible(
        expected_experiment=mock_parameterizable_experiment,
        provided_experiment=provided_experiment,
    )
    assert any(
        issue.kind == ExperimentInterfaceIssueKind.DOMAIN_NOT_COMPATIBLE
        and issue.identifier == "test_req1"
        for issue in issues
    )


def test_parameterized_optional_missing_from_provided(
    mock_parameterizable_experiment: Experiment,
    customParameterization: list[ConstitutivePropertyValue],
) -> None:
    """Parameterized optional inputs must be optional in the provided experiment."""
    expected_experiment = ParameterizedExperiment(
        parameterization=customParameterization,
        **mock_parameterizable_experiment.model_dump(),
    )
    provided_experiment = mock_parameterizable_experiment.model_copy(
        update={
            "optionalProperties": tuple(
                prop
                for prop in mock_parameterizable_experiment.optionalProperties
                if prop.identifier != customParameterization[0].property.identifier
            ),
            "defaultParameterization": tuple(
                value
                for value in mock_parameterizable_experiment.defaultParameterization
                if value.property.identifier
                != customParameterization[0].property.identifier
            ),
        }
    )
    issues = check_experiment_interface_compatible(
        expected_experiment=expected_experiment,
        provided_experiment=provided_experiment,
    )
    assert any(
        issue.kind
        == ExperimentInterfaceIssueKind.PARAMETERIZED_OPTIONAL_NOT_IN_PROVIDED
        and issue.identifier == "test_opt1"
        for issue in issues
    )


def test_parameterized_value_outside_provided_domain(
    mock_parameterizable_experiment: Experiment,
) -> None:
    """Parameterized values must fall within the provided optional property domain."""
    parameterization = [
        ConstitutivePropertyValue(
            value="C",
            property=ConstitutivePropertyDescriptor(identifier="test_opt1"),
        )
    ]
    expected_experiment = ParameterizedExperiment(
        parameterization=parameterization,
        **mock_parameterizable_experiment.model_dump(),
    )
    narrowed_optional = mock_parameterizable_experiment.optionalProperties[
        0
    ].model_copy(
        update={
            "propertyDomain": PropertyDomain(
                variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
                values=["A", "B"],
            )
        }
    )
    provided_experiment = mock_parameterizable_experiment.model_copy(
        update={
            "optionalProperties": (
                narrowed_optional,
                *mock_parameterizable_experiment.optionalProperties[1:],
            ),
        }
    )
    issues = check_experiment_interface_compatible(
        expected_experiment=expected_experiment,
        provided_experiment=provided_experiment,
    )
    assert any(
        issue.kind == ExperimentInterfaceIssueKind.PARAMETERIZED_VALUE_OUT_OF_DOMAIN
        and issue.identifier == "test_opt1"
        and issue.value == "C"
        for issue in issues
    )


def test_non_parameterized_optional_default_mismatch(
    mock_parameterizable_experiment: Experiment,
) -> None:
    """Non-parameterized optional defaults must match between expected and provided."""
    expected_experiment = mock_parameterizable_experiment.model_copy(
        update={
            "defaultParameterization": (
                ConstitutivePropertyValue(
                    value="A",
                    property=ConstitutivePropertyDescriptor(identifier="test_opt1"),
                ),
                *mock_parameterizable_experiment.defaultParameterization[1:],
            ),
        }
    )
    issues = check_experiment_interface_compatible(
        expected_experiment=expected_experiment,
        provided_experiment=mock_parameterizable_experiment,
    )
    assert any(
        issue.kind == ExperimentInterfaceIssueKind.OPTIONAL_DEFAULT_MISMATCH
        and issue.identifier == "test_opt1"
        and issue.expectedDefault == "A"
        and issue.providedDefault == "B"
        for issue in issues
    )


def test_missing_target_output_in_provided(
    mock_parameterizable_experiment: Experiment,
) -> None:
    """Expected outputs must be produced by the provided experiment."""
    provided_experiment = mock_parameterizable_experiment.model_copy(
        update={
            "targetProperties": [],
        }
    )
    issues = check_experiment_interface_compatible(
        expected_experiment=mock_parameterizable_experiment,
        provided_experiment=provided_experiment,
    )
    assert any(
        issue.kind == ExperimentInterfaceIssueKind.OUTPUT_NOT_IN_PROVIDED
        and issue.identifier == "measurable_one"
        for issue in issues
    )


def test_provided_only_optional_properties_are_compatible(
    mock_parameterizable_experiment: Experiment,
) -> None:
    """Provided-only optional parameters do not break interface compatibility."""
    provided_experiment = mock_parameterizable_experiment.model_copy(
        update={
            "optionalProperties": (
                *mock_parameterizable_experiment.optionalProperties,
                ConstitutiveProperty(identifier="new_provided_optional"),
            ),
            "defaultParameterization": (
                *mock_parameterizable_experiment.defaultParameterization,
                ConstitutivePropertyValue(
                    value=1,
                    property=ConstitutivePropertyDescriptor(
                        identifier="new_provided_optional"
                    ),
                ),
            ),
        }
    )
    issues = check_experiment_interface_compatible(
        expected_experiment=mock_parameterizable_experiment,
        provided_experiment=provided_experiment,
    )
    assert issues == []


def test_multiple_interface_issues_are_collected(
    mock_parameterizable_experiment: Experiment,
    requiredProperties: list[ConstitutiveProperty],
) -> None:
    """All interface mismatches are reported for a single experiment."""
    provided_experiment = mock_parameterizable_experiment.model_copy(
        update={
            "requiredProperties": tuple(requiredProperties[1:]),
            "targetProperties": [],
        }
    )
    issues = check_experiment_interface_compatible(
        expected_experiment=mock_parameterizable_experiment,
        provided_experiment=provided_experiment,
    )
    assert len(issues) >= 2
    assert any(
        issue.kind
        == ExperimentInterfaceIssueKind.MISSING_REQUIRED_CONSTITUTIVE_IN_PROVIDED
        for issue in issues
    )
    assert any(
        issue.kind == ExperimentInterfaceIssueKind.OUTPUT_NOT_IN_PROVIDED
        for issue in issues
    )
