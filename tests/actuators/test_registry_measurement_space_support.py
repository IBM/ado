# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for measurement-space experiment support checks in ActuatorRegistry."""

from ado.modules.actuators.registry import (
    ActuatorRegistry,
    format_measurement_space_interface_issue,
)
from ado.schema.experiment import (
    Experiment,
    check_experiment_interface_compatible,
)
from ado.schema.measurementspace import (
    MeasurementSpace,
    MeasurementSpaceConfiguration,
)
from ado.schema.property import (
    AbstractPropertyDescriptor,
    ConstitutiveProperty,
    ConstitutivePropertyDescriptor,
)
from ado.schema.property_value import ConstitutivePropertyValue


def test_format_measurement_space_interface_issue(
    mock_parameterizable_experiment: Experiment,
) -> None:
    """Registry formatting adds measurement-space and actuator catalog context."""
    issues = check_experiment_interface_compatible(
        expected_experiment=mock_parameterizable_experiment,
        provided_experiment=mock_parameterizable_experiment.model_copy(
            update={"targetProperties": []}
        ),
    )
    formatted = format_measurement_space_interface_issue(
        mock_parameterizable_experiment, issues[0]
    )
    assert formatted.startswith("ExperimentInterfaceMismatchError:")
    assert "measurement-space experiment" in formatted
    assert "actuator catalog" in formatted


def test_check_measurement_space_supported_reports_interface_mismatch(
    global_registry: ActuatorRegistry,
    mock_parameterizable_experiment: Experiment,
) -> None:
    """Registry check reports interface mismatches for embedded measurement-space experiments."""
    expected_experiment = mock_parameterizable_experiment.model_copy(
        update={
            "targetProperties": [
                AbstractPropertyDescriptor(identifier="missing_output"),
            ],
        }
    )
    measurement_space = MeasurementSpace(
        configuration=MeasurementSpaceConfiguration(experiments=[expected_experiment])
    )
    issues = global_registry.checkMeasurementSpaceSupported(measurement_space)
    assert any(
        "ExperimentInterfaceMismatchError" in issue
        and "missing_output" in issue
        and "measurement-space experiment" in issue
        for issue in issues
    )


def test_check_measurement_space_supported_skips_interface_on_lookup_failure(
    global_registry: ActuatorRegistry,
    mock_parameterizable_experiment: Experiment,
) -> None:
    """Interface checks are skipped when the experiment cannot be resolved."""
    expected_experiment = mock_parameterizable_experiment.model_copy(
        update={"identifier": "nonexistent_experiment_for_support_check"}
    )
    measurement_space = MeasurementSpace(
        configuration=MeasurementSpaceConfiguration(experiments=[expected_experiment])
    )
    issues = global_registry.checkMeasurementSpaceSupported(measurement_space)
    assert any("UnknownExperimentError" in issue for issue in issues)
    assert not any("ExperimentInterfaceMismatchError" in issue for issue in issues)


def test_check_measurement_space_supported_compatible_measurement_space(
    global_registry: ActuatorRegistry,
    mock_parameterizable_experiment: Experiment,
) -> None:
    """A measurement space matching the catalog produces no issues."""
    measurement_space = MeasurementSpace(
        configuration=MeasurementSpaceConfiguration(
            experiments=[mock_parameterizable_experiment]
        )
    )
    issues = global_registry.checkMeasurementSpaceSupported(measurement_space)
    assert issues == []


def test_check_measurement_space_supported_allows_provided_optional_extension(
    global_registry: ActuatorRegistry,
    mock_parameterizable_experiment: Experiment,
) -> None:
    """Provided-only optional parameters remain compatible."""
    catalog = global_registry.catalogForActuatorIdentifier("mock")
    major_version_identifier = mock_parameterizable_experiment.major_version_identifier
    original_experiment = catalog._experiments[major_version_identifier]
    extended_provided_experiment = mock_parameterizable_experiment.model_copy(
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
    try:
        catalog._experiments[major_version_identifier] = extended_provided_experiment

        measurement_space = MeasurementSpace(
            configuration=MeasurementSpaceConfiguration(
                experiments=[mock_parameterizable_experiment]
            )
        )
        issues = global_registry.checkMeasurementSpaceSupported(measurement_space)
        assert issues == []
    finally:
        catalog._experiments[major_version_identifier] = original_experiment


def test_check_measurement_space_supported_unknown_experiment_error_prefix(
    global_registry: ActuatorRegistry,
) -> None:
    """Unknown experiment errors retain the existing prefix."""
    measurement_space = MeasurementSpace(
        configuration=MeasurementSpaceConfiguration(
            experiments=[
                Experiment(
                    actuatorIdentifier="mock",
                    identifier="definitely_missing_experiment",
                    targetProperties=[AbstractPropertyDescriptor(identifier="output")],
                )
            ]
        )
    )
    issues = global_registry.checkMeasurementSpaceSupported(measurement_space)
    assert len(issues) == 1
    assert issues[0].startswith("UnknownExperimentError:")
