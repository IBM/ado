# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typing

import typer
import yaml
from rich.status import Status

from orchestrator.cli.models.parameters import AdoShowEntitiesCommandParameters
from orchestrator.cli.models.types import (
    AdoShowEntitiesSupportedEntityTypes,
    AdoShowEntitiesSupportedPropertyFormats,
)
from orchestrator.cli.utils.generic.wrappers import get_sql_store
from orchestrator.cli.utils.output.dataframes import df_to_output
from orchestrator.cli.utils.output.prints import (
    ADO_SPINNER_INITIALIZING_DISCOVERY_SPACE,
    ADO_SPINNER_QUERYING_DB,
    ERROR,
    HINT,
    INFO,
    WARN,
    console_print,
    cyan,
    magenta,
)
from orchestrator.core.discoveryspace.config import DiscoverySpaceConfiguration
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.metastore.base import ResourceDoesNotExistError

if typing.TYPE_CHECKING:
    import pandas as pd

    from orchestrator.schema.entity import Entity


def show_discovery_space_entities(parameters: AdoShowEntitiesCommandParameters) -> None:

    import pandas as pd

    supported_entity_types = [
        AdoShowEntitiesSupportedEntityTypes.MATCHING,
        AdoShowEntitiesSupportedEntityTypes.MEASURED,
        AdoShowEntitiesSupportedEntityTypes.MISSING,
        AdoShowEntitiesSupportedEntityTypes.UNMEASURED,
    ]

    if parameters.entities_type not in supported_entity_types:
        supported_types_str = [t.value for t in supported_entity_types]
        raise typer.BadParameter(
            f"type must be one of {supported_types_str} for ado show-entities space",
        )

    sql = get_sql_store(project_context=parameters.ado_configuration.project_context)
    space_identifier = parameters.resource_id

    if parameters.resource_id:

        with Status(ADO_SPINNER_QUERYING_DB) as status:
            space_resource = sql.getResource(
                identifier=parameters.resource_id, kind=CoreResourceKinds.DISCOVERYSPACE
            )
            if not space_resource:
                status.stop()
                raise ResourceDoesNotExistError(
                    resource_id=parameters.resource_id,
                    kind=CoreResourceKinds.DISCOVERYSPACE,
                )

        configuration = space_resource.config

    else:

        try:
            configuration = DiscoverySpaceConfiguration.model_validate(
                yaml.safe_load(parameters.resource_configuration.read_text())
            )
        except ValueError as e:
            console_print(
                f"{ERROR}The space configuration provided is not valid:\n{e}",
                stderr=True,
            )
            raise typer.Exit(1) from e
        except OSError as e:
            console_print(
                f"{ERROR}There was a problem while reading the space configuration provided:\n\t{e}",
                stderr=True,
            )
            raise typer.Exit(1) from e

        if parameters.entities_type != AdoShowEntitiesSupportedEntityTypes.MATCHING:
            console_print(
                f"{WARN}The {cyan(f'--include {parameters.entities_type.value}')} option is not supported when "
                "passing a resource configuration file.\n\tThe only supported option is "
                f"is {cyan(f'--include {AdoShowEntitiesSupportedEntityTypes.MATCHING.value}')}.\n"
                f"{INFO}The {cyan(AdoShowEntitiesSupportedEntityTypes.MATCHING.value)} option will be used.",
                stderr=True,
            )
            parameters.entities_type = AdoShowEntitiesSupportedEntityTypes.MATCHING

        # AP: we set the resource ID to the file name to make it easier
        # for the prints
        parameters.resource_id = f"from_file_{parameters.resource_configuration.stem}"

    with Status(ADO_SPINNER_INITIALIZING_DISCOVERY_SPACE) as status:

        space = DiscoverySpace.from_configuration(
            conf=configuration,
            project_context=parameters.ado_configuration.project_context,
            identifier=space_identifier,
        )

        output_df: pd.DataFrame = pd.DataFrame()
        status.update(f"Finding {parameters.entities_type.value} entities")

        if parameters.entities_type == AdoShowEntitiesSupportedEntityTypes.MATCHING:
            output_df = space.matchingEntitiesTable(
                property_type=parameters.entities_property_format.value,
                aggregationMethod=parameters.aggregation_method,
            )
        elif parameters.entities_type == AdoShowEntitiesSupportedEntityTypes.MEASURED:
            output_df = space.measuredEntitiesTable(
                property_type=parameters.entities_property_format.value,
                aggregationMethod=parameters.aggregation_method,
            )
        elif parameters.entities_type == AdoShowEntitiesSupportedEntityTypes.UNMEASURED:
            unsampled_entities = unmeasured_entities_from_space(space)
            output_df = entities_to_dataframe(unsampled_entities)
        elif parameters.entities_type == AdoShowEntitiesSupportedEntityTypes.MISSING:
            missing_entities = missing_entities_from_space(space)
            output_df = entities_to_dataframe(missing_entities)

    if output_df.empty:
        console_print(
            f"{INFO}Nothing was returned for "
            f"[i]entity type {magenta(parameters.entities_type.value)}[/i] and "
            f"[i]property format {magenta(parameters.entities_property_format.value)}[/i] "
            f"in [i]space {magenta(parameters.resource_id)}[/i].",
            stderr=True,
        )
        return

    if (
        parameters.properties
        and parameters.entities_type != AdoShowEntitiesSupportedEntityTypes.MISSING
    ):
        df_column_set = set(output_df.columns)
        properties_set = set(parameters.properties)
        available_properties = (
            space.measurementSpace.targetProperties
            if parameters.entities_property_format
            == AdoShowEntitiesSupportedPropertyFormats.TARGET
            else space.measurementSpace.observedProperties
        )
        available_properties_formatted = "-\t" + "\n-\t".join(
            p.identifier for p in available_properties
        )
        if not properties_set.issubset(df_column_set):
            console_print(
                f"{ERROR}{properties_set.difference(df_column_set)} are not in the available properties.\n"
                f"{HINT}Available ones for the {parameters.entities_property_format.value} format are:\n"
                f"{available_properties_formatted}",
                stderr=True,
            )
            raise typer.Exit(1)

        # AP: always include identifier
        parameters.properties.insert(0, "identifier")
        output_df = output_df[parameters.properties]

    file_name = f"{parameters.resource_id}_description_{parameters.entities_type.value}_{parameters.entities_property_format.value}.{parameters.entities_output_format.value}"
    df_to_output(
        df=output_df,
        output_format=parameters.entities_output_format.value,
        file_name=file_name,
    )


def unmeasured_entities_from_space(space: DiscoverySpace) -> list["Entity"]:
    from orchestrator.core.discoveryspace.samplers import (
        ExplicitEntitySpaceGridSampleGenerator,
        WalkModeEnum,
    )

    if not space.entitySpace.isDiscreteSpace:
        console_print(
            f"{ERROR}The entity space has at least one continuous dimension (infinite points). "
            "Unmeasured points cannot be calculated.",
            stderr=True,
        )
        raise typer.Exit(1)

    sampler = ExplicitEntitySpaceGridSampleGenerator(WalkModeEnum.SEQUENTIAL)

    iterator = sampler.entitySpaceIterator(space.entitySpace, batchsize=1)

    measured_entities = [
        e for e in space.sampledEntities() if len(e.observedPropertyValues) > 0
    ]

    return [entity[0] for entity in iterator if entity[0] not in measured_entities]


def missing_entities_from_space(space: DiscoverySpace) -> list["Entity"]:
    from orchestrator.core.discoveryspace.samplers import (
        ExplicitEntitySpaceGridSampleGenerator,
        WalkModeEnum,
    )

    if not space.entitySpace.isDiscreteSpace:
        console_print(
            f"{ERROR}The entity space has at least one continuous dimension (infinite points). "
            "Missing points cannot be calculated.",
            stderr=True,
        )
        raise typer.Exit(1)

    sampler = ExplicitEntitySpaceGridSampleGenerator(WalkModeEnum.SEQUENTIAL)

    # AP 22/05/2024: The entitySpaceIterator returns entities in batches,
    # even if by default the batch size is 1. This means that we need to
    # select the first (and only) item.
    iterator = sampler.entitySpaceIterator(space.entitySpace, batchsize=1)
    matching_entities = [
        e for e in space.matchingEntities() if len(e.observedPropertyValues) > 0
    ]
    return [entity[0] for entity in iterator if entity[0] not in matching_entities]


def entities_to_dataframe(
    entities: typing.Iterable["Entity"],
    constitutive_properties_only: bool = True,
) -> "pd.DataFrame":
    import pandas as pd

    if constitutive_properties_only:
        return pd.DataFrame(
            [
                {p.property.identifier: p.value for p in e.constitutive_property_values}
                for e in entities
            ]
        )
    return pd.DataFrame(
        [{p.property.identifier: p.value for p in e.propertyValues} for e in entities]
    )
