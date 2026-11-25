# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Unit tests for CLI property resolution and DataFrame enrichment."""

import pandas as pd

from ado.cli.core.config import AdoConfiguration
from ado.cli.models.parameters import AdoShowMeasurementsCommandParameters
from ado.cli.models.types import (
    AdoShowMeasurementsSupportedEntityTypes,
    AdoShowMeasurementsSupportedOutputFormats,
    AdoShowMeasurementsSupportedPropertyFormats,
)
from ado.cli.utils.resources.properties import (
    add_properties_to_dataframe,
    normalize_include_properties_option,
    resolve_experiment_properties,
    resolve_properties_for_experiment,
    resolve_requested_property_identifiers,
)
from ado.metastore.project import ProjectContext
from ado.schema.domain import PropertyDomain, VariableTypeEnum
from ado.schema.experiment import Experiment
from ado.schema.property import ConstitutiveProperty, ConstitutivePropertyDescriptor
from ado.schema.property_value import ConstitutivePropertyValue
from ado.schema.reference import ExperimentReference
from ado.utilities.location import SQLiteStoreConfiguration


def test_normalize_include_properties_option_empty() -> None:
    """Empty or None input produces an empty list."""
    assert normalize_include_properties_option(None) == []
    assert normalize_include_properties_option([]) == []


def test_normalize_include_properties_option_comma_separated_and_repetition() -> None:
    """Comma-separated values and multiple arguments are split, stripped, and deduplicated."""
    normalized_include_properties = normalize_include_properties_option(
        ["constitutive, parameterized", "optional,constitutive", "my_prop", "my_prop"]
    )
    assert normalized_include_properties == [
        "constitutive",
        "parameterized",
        "optional",
        "my_prop",
    ]


def test_normalize_include_properties_option_keywords_lowercased() -> None:
    """Known keywords are lowercased, custom identifiers retain case."""
    normalized_include_properties = normalize_include_properties_option(
        ["CONSTITUTIVE", "All", "OptIONAL", "CustomProp"]
    )
    assert normalized_include_properties == [
        "constitutive",
        "all",
        "optional",
        "CustomProp",
    ]


def test_resolve_properties_for_experiment_unparameterized() -> None:
    """An experiment with default optional properties returns them in optional_properties and empty parameterized."""
    optional_property = ConstitutiveProperty(
        identifier="max_retries",
        propertyDomain=PropertyDomain(
            variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
            domainRange=[1, 10],
            interval=1,
        ),
    )
    default_value = ConstitutivePropertyValue(
        property=ConstitutivePropertyDescriptor(identifier="max_retries"),
        value=3,
    )
    experiment = Experiment(
        identifier="test_exp",
        actuatorIdentifier="test_actuator",
        targetProperties=(),
        optionalProperties=(optional_property,),
        defaultParameterization=(default_value,),
    )

    (
        optional_properties,
        parameterized_properties,
    ) = resolve_properties_for_experiment(experiment)
    assert "max_retries" in optional_properties
    assert optional_properties["max_retries"].value == 3
    assert len(parameterized_properties) == 0


def test_resolve_properties_for_experiment_parameterized_reference() -> None:
    """An ExperimentReference with parameterization returns custom values."""
    custom_property_value = ConstitutivePropertyValue(
        property=ConstitutivePropertyDescriptor(identifier="learning_rate"),
        value=0.01,
    )
    experiment_reference = ExperimentReference(
        actuatorIdentifier="actuator",
        experimentIdentifier="exp",
        parameterization=[custom_property_value],
    )

    (
        optional_properties,
        parameterized_properties,
    ) = resolve_properties_for_experiment(experiment_reference)
    assert "learning_rate" in optional_properties
    assert optional_properties["learning_rate"].value == 0.01
    assert "learning_rate" in parameterized_properties
    assert parameterized_properties["learning_rate"].value == 0.01


def test_resolve_experiment_properties_multiple() -> None:
    """Multiple experiments combine their optional and parameterized properties."""
    property_value_1 = ConstitutivePropertyValue(
        property=ConstitutivePropertyDescriptor(identifier="p1"),
        value="a",
    )
    property_value_2 = ConstitutivePropertyValue(
        property=ConstitutivePropertyDescriptor(identifier="p2"),
        value="b",
    )
    experiment_reference_1 = ExperimentReference(
        actuatorIdentifier="act1",
        experimentIdentifier="exp1",
        parameterization=[property_value_1],
    )
    experiment_reference_2 = ExperimentReference(
        actuatorIdentifier="act2",
        experimentIdentifier="exp2",
        parameterization=[property_value_2],
    )

    (
        optional_properties,
        parameterized_properties,
    ) = resolve_experiment_properties([experiment_reference_1, experiment_reference_2])
    assert optional_properties["p1"].value == "a"
    assert optional_properties["p2"].value == "b"
    assert parameterized_properties["p1"].value == "a"
    assert parameterized_properties["p2"].value == "b"


def test_resolve_requested_property_identifiers() -> None:
    """Resolve property requests to an ordered list of property identifiers based on keywords and names."""
    constitutive_identifiers = ["dim1", "dim2"]
    parameterized_identifiers = ["param1"]
    optional_identifiers = ["param1", "opt2"]

    # Test 'constitutive'
    assert resolve_requested_property_identifiers(
        ["constitutive"],
        constitutive_identifiers,
        parameterized_identifiers,
        optional_identifiers,
    ) == ["dim1", "dim2"]

    # Test 'parameterized'
    assert resolve_requested_property_identifiers(
        ["parameterized"],
        constitutive_identifiers,
        parameterized_identifiers,
        optional_identifiers,
    ) == ["param1"]

    # Test 'optional'
    assert resolve_requested_property_identifiers(
        ["optional"],
        constitutive_identifiers,
        parameterized_identifiers,
        optional_identifiers,
    ) == ["param1", "opt2"]

    # Test 'all'
    assert resolve_requested_property_identifiers(
        ["all"],
        constitutive_identifiers,
        parameterized_identifiers,
        optional_identifiers,
    ) == ["dim1", "dim2", "param1", "opt2"]

    # Test specific identifier and combined
    assert resolve_requested_property_identifiers(
        ["param1", "custom_named", "constitutive"],
        constitutive_identifiers,
        parameterized_identifiers,
        optional_identifiers,
    ) == ["param1", "custom_named", "dim1", "dim2"]


def test_add_properties_to_dataframe() -> None:
    """Test adding experiment properties as columns to a measurements DataFrame."""
    property_value = ConstitutivePropertyValue(
        property=ConstitutivePropertyDescriptor(identifier="learning_rate"),
        value=0.01,
    )
    experiment_reference = ExperimentReference(
        actuatorIdentifier="actuator",
        experimentIdentifier="exp",
        parameterization=[property_value],
    )

    measurements_df = pd.DataFrame(
        [
            {
                "identifier": "e1",
                "generatorid": "g1",
                "experiment_id": str(experiment_reference),
                "dim1": 10,
                "accuracy": 0.95,
            }
        ]
    )

    result_df = add_properties_to_dataframe(
        df=measurements_df,
        include_properties=["parameterized"],
        experiments=[experiment_reference],
        constitutive_property_identifiers=["dim1"],
    )

    assert "learning_rate" in result_df.columns
    assert result_df["learning_rate"].iloc[0] == 0.01
    # Existing columns keep their order and the requested properties are appended
    expected_order = [
        "identifier",
        "generatorid",
        "experiment_id",
        "dim1",
        "accuracy",
        "learning_rate",
    ]
    assert list(result_df.columns) == expected_order


def test_show_measurements_command_parameters_lifecycle() -> None:
    """Test full pydantic lifecycle for AdoShowMeasurementsCommandParameters: create -> dump -> create from dump."""
    parameters = AdoShowMeasurementsCommandParameters(
        ado_configuration=AdoConfiguration(
            project_context=ProjectContext(
                project="test_proj",
                metadataStore=SQLiteStoreConfiguration(path="test_proj.db"),
            )
        ),
        aggregation_method=None,
        from_experiment=["exp1"],
        include_properties=["parameterized", "optional"],
        measurements_output_format=AdoShowMeasurementsSupportedOutputFormats.TABLE,
        measurements_property_format=AdoShowMeasurementsSupportedPropertyFormats.TARGET,
        measurements_type=AdoShowMeasurementsSupportedEntityTypes.MEASURED,
        no_trunc=True,
        output_file=None,
        properties=["acc"],
        resource_configuration=None,
        resource_id="space-123",
    )

    dumped_parameters = parameters.model_dump()
    assert dumped_parameters["include_properties"] == ["parameterized", "optional"]

    reconstructed_parameters = AdoShowMeasurementsCommandParameters.model_validate(
        dumped_parameters
    )
    assert reconstructed_parameters.include_properties == [
        "parameterized",
        "optional",
    ]
    assert reconstructed_parameters.resource_id == "space-123"
