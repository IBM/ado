# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import typing

import pydantic
import typer
import yaml
from rich.status import Status

from orchestrator.cli.models.parameters import AdoGetCommandParameters
from orchestrator.cli.models.types import (
    AdoGetSupportedOutputFormats,
)
from orchestrator.cli.utils.generic.wrappers import get_sql_store
from orchestrator.cli.utils.output.prints import (
    ADO_INFO_EMPTY_DATAFRAME,
    ADO_SPINNER_GETTING_OUTPUT_READY,
    ADO_SPINNER_QUERYING_DB,
    ERROR,
    console_print,
)
from orchestrator.cli.utils.queries.parser import prepare_query_filters_for_db
from orchestrator.cli.utils.resources.formatters import (
    format_default_ado_get_multiple_resources,
    format_resource_for_ado_get_custom_format,
)
from orchestrator.core import DiscoverySpaceResource
from orchestrator.core.discoveryspace.config import (
    DiscoverySpaceConfiguration,
    SpaceHierarchy,
)
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.schema.entityspace import EntitySpaceRepresentation
from orchestrator.schema.point import SpacePoint

if typing.TYPE_CHECKING:
    import pandas as pd


def get_discovery_space(parameters: AdoGetCommandParameters) -> None:
    from orchestrator.cli.utils.resources.handlers import (
        handle_ado_get_default_format,
        handle_ado_get_special_formats,
    )

    if parameters.matching_point:

        matching_spaces = _find_spaces_matching_point(parameters)
        if parameters.output_format == AdoGetSupportedOutputFormats.DEFAULT:
            output_df = format_default_ado_get_multiple_resources(
                resources=_discovery_space_resource_list_to_ado_get_default_dataframe(
                    resources=matching_spaces, parameters=parameters
                ),
                resource_kind=CoreResourceKinds.DISCOVERYSPACE,
            )

            if output_df.empty:
                console_print(ADO_INFO_EMPTY_DATAFRAME, stderr=True)
            else:
                console_print(output_df)

        else:
            console_print(
                format_resource_for_ado_get_custom_format(
                    to_print=matching_spaces, parameters=parameters
                )
            )

        return

    if parameters.matching_space or parameters.matching_space_id:

        matching_spaces = _find_spaces_matching_space(parameters)
        if parameters.output_format == AdoGetSupportedOutputFormats.DEFAULT:
            output_df = format_default_ado_get_multiple_resources(
                resources=matching_spaces,
                resource_kind=CoreResourceKinds.DISCOVERYSPACE,
            )

            if output_df.empty:
                console_print(ADO_INFO_EMPTY_DATAFRAME, stderr=True)
            else:
                console_print(output_df)

        return

    if parameters.output_format == AdoGetSupportedOutputFormats.DEFAULT:
        handle_ado_get_default_format(
            parameters=parameters,
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
        )
    else:
        handle_ado_get_special_formats(
            parameters=parameters,
            resource_type=CoreResourceKinds.DISCOVERYSPACE,
        )


def _find_spaces_matching_point(
    parameters: AdoGetCommandParameters,
) -> list[DiscoverySpaceResource]:

    if (
        not parameters.matching_point.exists()
        or not parameters.matching_point.is_file()
    ):
        console_print(
            f"{ERROR}{parameters.matching_point.resolve()} does not exist or is not a file.",
            stderr=True,
        )
        raise typer.Exit(1)

    try:
        target_point = SpacePoint.model_validate(
            yaml.safe_load(parameters.matching_point.read_text())
        )
    except pydantic.ValidationError:
        raise

    experiment_selectors = []
    for reference in target_point.experiments or []:
        experiment_selectors.extend(
            prepare_query_filters_for_db(
                {
                    "config.experiments": {
                        "experiments": {"identifier": reference.experimentIdentifier}
                    }
                }
            )
        )

    sql_store = get_sql_store(
        project_context=parameters.ado_configuration.project_context
    )

    with Status(ADO_SPINNER_QUERYING_DB):
        spaces_matching_experiments: list[DiscoverySpaceResource] = list(
            sql_store.getResourcesOfKind(
                kind=CoreResourceKinds.DISCOVERYSPACE.value,
                field_selectors=experiment_selectors,
            ).values()
        )

    return [
        space
        for space in spaces_matching_experiments
        if EntitySpaceRepresentation(space.config.entitySpace).isPointInSpace(
            point=target_point.entity, allow_partial_matches=True
        )
    ]


def _find_spaces_matching_space(
    parameters: AdoGetCommandParameters,
) -> "pd.DataFrame":

    sql_store = get_sql_store(
        project_context=parameters.ado_configuration.project_context
    )

    if parameters.matching_space:
        if (
            not parameters.matching_space.exists()
            or not parameters.matching_space.is_file()
        ):
            console_print(
                f"{ERROR}{parameters.matching_space.resolve()} does not exist or is not a file.",
                stderr=True,
            )
            raise typer.Exit(1)

        with Status("Validating input") as spinner:
            try:
                space_configuration = DiscoverySpaceConfiguration.model_validate(
                    yaml.safe_load(parameters.matching_space.read_text())
                )
            except pydantic.ValidationError as error:
                spinner.stop()
                console_print(
                    f"{ERROR}The space configuration provided was not valid:\n{error}",
                    stderr=True,
                )
                raise typer.Exit(1) from error
    else:

        with Status(ADO_SPINNER_QUERYING_DB):
            space_resource: DiscoverySpaceResource = sql_store.getResource(
                identifier=parameters.matching_space_id,
                kind=CoreResourceKinds.DISCOVERYSPACE,
                raise_error_if_no_resource=True,
            )
            space_configuration = space_resource.config

    # Pre-filtering on experiment identifiers allows us to check less things by hand
    target_base_experiment_identifiers = {
        experiment.experimentIdentifier
        for experiment in space_configuration.convert_experiments_to_reference_list().experiments
    }
    with Status("Preparing query") as spinner:
        experiment_selectors = []
        for identifier in target_base_experiment_identifiers:
            experiment_selectors.extend(
                prepare_query_filters_for_db(
                    {"config.experiments": {"experiments": {"identifier": identifier}}}
                )
            )

        spinner.update(ADO_SPINNER_QUERYING_DB)
        spaces_matching_experiments: list[DiscoverySpaceResource] = list(
            sql_store.getResourcesOfKind(
                kind=CoreResourceKinds.DISCOVERYSPACE.value,
                field_selectors=experiment_selectors,
            ).values()
        )

        # The DB query only fetches spaces that have *at least*
        # the experiments that we ask for. If we want an exact
        # match, we then have to manually filter out the spaces
        # that have additional experiments
        spinner.update(ADO_SPINNER_GETTING_OUTPUT_READY)
        filtered_spaces_matching_experiments = []
        for space in spaces_matching_experiments:
            current_base_experiment_identifiers = {
                experiment.experimentIdentifier
                for experiment in space.config.convert_experiments_to_reference_list().experiments
            }

            if (
                current_base_experiment_identifiers
                == target_base_experiment_identifiers
            ):
                filtered_spaces_matching_experiments.append(space)

        spaces_matching_experiments = filtered_spaces_matching_experiments

        from datetime import datetime, timezone

        import pandas as pd

        result_df = pd.DataFrame(
            [
                {
                    "IDENTIFIER": space_resource.identifier,
                    "NAME": space_resource.config.metadata.name,
                    "DESCRIPTION": space_resource.config.metadata.labels,
                    "AGE": datetime.now(timezone.utc) - space_resource.created,
                    "RELATION_TO_INPUT_SPACE": space_resource.config.compare_space_hierarchy(
                        reference_space=space_configuration,
                    ).value,
                }
                for space_resource in spaces_matching_experiments
            ]
        )

    if result_df.empty:
        return result_df

    result_df = result_df[
        result_df["RELATION_TO_INPUT_SPACE"] != SpaceHierarchy.UNDEFINED.value
    ]
    result_df = result_df.sort_values(by="AGE", ascending=False)
    result_df = result_df.reset_index(drop=True)

    return (
        result_df
        if parameters.show_details
        else result_df[["IDENTIFIER", "NAME", "AGE", "RELATION_TO_INPUT_SPACE"]]
    )


def _discovery_space_resource_list_to_ado_get_default_dataframe(
    resources: list[DiscoverySpaceResource],
    parameters: AdoGetCommandParameters,
) -> "pd.DataFrame":
    from datetime import datetime, timezone

    import pandas as pd

    result_df = pd.DataFrame(
        [
            {
                "IDENTIFIER": space_resource.identifier,
                "NAME": space_resource.config.metadata.name,
                "DESCRIPTION": space_resource.config.metadata.labels,
                "AGE": datetime.now(timezone.utc) - space_resource.created,
            }
            for space_resource in resources
        ]
    )
    result_df = result_df.sort_values(by="AGE", ascending=False)
    result_df = result_df.reset_index(drop=True)

    return (
        result_df
        if parameters.show_details
        else result_df[["IDENTIFIER", "NAME", "AGE"]]
    )
