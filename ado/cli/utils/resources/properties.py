# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Shared property resolver and --include-properties normalisation utilities for CLI commands."""

import typing
from collections.abc import Sequence

if typing.TYPE_CHECKING:
    import pandas as pd

from ado.modules.actuators.registry import ActuatorRegistry
from ado.schema.experiment import Experiment, ParameterizedExperiment
from ado.schema.measurementspace import (
    MeasurementSpace,
    MeasurementSpaceConfiguration,
)
from ado.schema.property_value import ConstitutivePropertyValue
from ado.schema.reference import ExperimentReference

KEYWORD_CONSTITUTIVE = "constitutive"
KEYWORD_PARAMETERIZED = "parameterized"
KEYWORD_OPTIONAL = "optional"
KEYWORD_ALL = "all"
ALL_KEYWORDS: frozenset[str] = frozenset(
    {
        KEYWORD_CONSTITUTIVE,
        KEYWORD_PARAMETERIZED,
        KEYWORD_OPTIONAL,
        KEYWORD_ALL,
    }
)


def normalize_include_properties_option(
    include_properties: list[str] | None,
) -> list[str]:
    """Normalise the values of the --include-properties / -p CLI option.

    Handles comma-separated values, whitespace stripping, and deduplication
    while preserving insertion order. Known keywords are lowercased.

    Args:
        include_properties: List of strings passed to --include-properties / -p.

    Returns:
        List of normalised and deduplicated requested properties.
    """
    if not include_properties:
        return []

    normalized_properties: list[str] = []
    for property_argument in include_properties:
        for raw_property in property_argument.split(","):
            requested_property = raw_property.strip()
            if not requested_property:
                continue
            if requested_property.lower() in ALL_KEYWORDS:
                requested_property = requested_property.lower()
            if requested_property not in normalized_properties:
                normalized_properties.append(requested_property)
    return normalized_properties


def resolve_properties_for_experiment(
    experiment: Experiment | ParameterizedExperiment | ExperimentReference,
    registry: ActuatorRegistry | None = None,
) -> tuple[dict[str, ConstitutivePropertyValue], dict[str, ConstitutivePropertyValue]]:
    """Resolve optional and parameterized properties for a single experiment or reference.

    Args:
        experiment: An Experiment, ParameterizedExperiment, or ExperimentReference.
        registry: Optional ActuatorRegistry to look up experiment definitions. If None,
            the global registry is used.

    Returns:
        A tuple of (optional_properties, parameterized_properties) mappings from
        property identifier to ConstitutivePropertyValue.
    """
    optional_properties: dict[str, ConstitutivePropertyValue] = {}
    parameterized_properties: dict[str, ConstitutivePropertyValue] = {}

    # Defaults come from the experiment definition. A reference does not carry
    # them, so its definition is looked up in the registry.
    experiment_definition: Experiment | ExperimentReference = experiment
    if isinstance(experiment, ExperimentReference):
        actuator_registry = registry or ActuatorRegistry.globalRegistry()
        try:
            experiment_definition = actuator_registry.experimentForReference(
                experiment, match_on="any"
            )
        except Exception:
            experiment_definition = experiment

    if isinstance(experiment_definition, Experiment):
        for default_property_value in experiment_definition.defaultParameterization:
            optional_properties[default_property_value.property.identifier] = (
                default_property_value
            )

    # Custom values always come from the experiment that was passed in and
    # override the defaults.
    if isinstance(experiment, (ParameterizedExperiment, ExperimentReference)):
        for custom_property_value in experiment.parameterization or []:
            parameterized_properties[custom_property_value.property.identifier] = (
                custom_property_value
            )
            optional_properties[custom_property_value.property.identifier] = (
                custom_property_value
            )

    return optional_properties, parameterized_properties


def resolve_experiment_properties(
    experiments: (
        MeasurementSpaceConfiguration
        | MeasurementSpace
        | Sequence[Experiment | ParameterizedExperiment | ExperimentReference]
        | None
    ),
    registry: ActuatorRegistry | None = None,
) -> tuple[dict[str, ConstitutivePropertyValue], dict[str, ConstitutivePropertyValue]]:
    """Resolve optional and parameterized properties across multiple experiments.

    Args:
        experiments: A MeasurementSpaceConfiguration, MeasurementSpace, or sequence
            of Experiment / ParameterizedExperiment / ExperimentReference objects.
        registry: Optional ActuatorRegistry to look up experiment definitions.

    Returns:
        A tuple of (optional_properties, parameterized_properties) mappings from
        property identifier to ConstitutivePropertyValue.
    """
    optional_properties: dict[str, ConstitutivePropertyValue] = {}
    parameterized_properties: dict[str, ConstitutivePropertyValue] = {}

    experiment_list: Sequence[
        Experiment | ParameterizedExperiment | ExperimentReference
    ] = (
        experiments.experiments
        if isinstance(experiments, (MeasurementSpaceConfiguration, MeasurementSpace))
        else experiments or []
    )

    for experiment in experiment_list:
        (
            experiment_optional_properties,
            experiment_parameterized_properties,
        ) = resolve_properties_for_experiment(experiment, registry=registry)
        optional_properties.update(experiment_optional_properties)
        parameterized_properties.update(experiment_parameterized_properties)

    return optional_properties, parameterized_properties


def resolve_requested_property_identifiers(
    include_properties: list[str],
    constitutive_identifiers: typing.Iterable[str],
    parameterized_identifiers: typing.Iterable[str],
    optional_identifiers: typing.Iterable[str],
) -> list[str]:
    """Resolve requested properties to an ordered list of property identifiers.

    Args:
        include_properties: Raw list of strings from the --include-properties / -p CLI option.
        constitutive_identifiers: Available constitutive property identifiers.
        parameterized_identifiers: Available parameterized property identifiers.
        optional_identifiers: Available optional property identifiers.

    Returns:
        Ordered, deduplicated list of resolved property identifiers.
    """
    constitutive_identifiers_list = list(constitutive_identifiers)
    parameterized_identifiers_list = list(parameterized_identifiers)
    optional_identifiers_list = list(optional_identifiers)

    all_available_identifiers = list(
        dict.fromkeys(
            constitutive_identifiers_list
            + optional_identifiers_list
            + parameterized_identifiers_list
        )
    )

    normalized_include_properties = normalize_include_properties_option(
        include_properties
    )
    result_identifiers: list[str] = []

    for requested_property in normalized_include_properties:
        match requested_property:
            case "constitutive":
                matched_identifiers = constitutive_identifiers_list
            case "parameterized":
                matched_identifiers = parameterized_identifiers_list
            case "optional":
                matched_identifiers = optional_identifiers_list
            case "all":
                matched_identifiers = all_available_identifiers
            case _:
                matched_identifiers = [requested_property]

        result_identifiers.extend(matched_identifiers)

    return list(dict.fromkeys(result_identifiers))


def add_properties_to_dataframe(
    df: "pd.DataFrame",
    include_properties: list[str] | None,
    experiments: (
        MeasurementSpaceConfiguration
        | MeasurementSpace
        | Sequence[Experiment | ParameterizedExperiment | ExperimentReference]
        | None
    ),
    constitutive_property_identifiers: list[str] | None = None,
) -> "pd.DataFrame":
    """Add requested experiment properties as columns to a measurements DataFrame.

    Args:
        df: The input pandas measurements DataFrame.
        include_properties: Values of the --include-properties / -p CLI option.
        experiments: Experiments associated with the space, operation, or samplestore.
        constitutive_property_identifiers: Optional list of constitutive property identifiers.

    Returns:
        The input DataFrame with the requested property columns appended.
    """
    if df.empty or include_properties is None:
        return df

    experiment_list: Sequence[
        Experiment | ParameterizedExperiment | ExperimentReference
    ] = (
        experiments.experiments
        if isinstance(experiments, (MeasurementSpaceConfiguration, MeasurementSpace))
        else experiments or []
    )

    constitutive_identifiers = list(constitutive_property_identifiers or [])
    (
        optional_properties,
        parameterized_properties,
    ) = resolve_experiment_properties(experiment_list)

    # Optional properties of each experiment, keyed by the string form of its
    # reference, which is the value stored in the "experiment_id" column.
    properties_by_experiment_id: dict[str, dict[str, ConstitutivePropertyValue]] = {}
    for experiment in experiment_list:
        experiment_properties, _ = resolve_properties_for_experiment(experiment)
        experiment_reference = (
            experiment
            if isinstance(experiment, ExperimentReference)
            else experiment.reference
        )
        properties_by_experiment_id[str(experiment_reference)] = experiment_properties

    requested_identifiers = resolve_requested_property_identifiers(
        include_properties=include_properties,
        constitutive_identifiers=constitutive_identifiers,
        parameterized_identifiers=parameterized_properties.keys(),
        optional_identifiers=optional_properties.keys(),
    )

    experiment_ids = (
        df["experiment_id"] if "experiment_id" in df.columns else [None] * len(df)
    )
    row_property_mappings = [
        properties_by_experiment_id.get(str(experiment_id), optional_properties)
        for experiment_id in experiment_ids
    ]

    for property_identifier in requested_identifiers:
        # Constitutive properties already present in the DataFrame are not overwritten.
        if (
            property_identifier in constitutive_identifiers
            and property_identifier in df.columns
        ):
            continue
        df[property_identifier] = [
            property_mapping[property_identifier].value
            if property_identifier in property_mapping
            else None
            for property_mapping in row_property_mappings
        ]

    return df
