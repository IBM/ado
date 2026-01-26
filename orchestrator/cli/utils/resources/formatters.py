# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import datetime
import json
import math
import typing

import pydantic
import typer
import yaml

from orchestrator.cli.models.types import AdoGetSupportedOutputFormats
from orchestrator.cli.utils.generic.constants import (
    SECONDS_IN_A_DAY,
    SECONDS_IN_A_MINUTE,
    SECONDS_IN_AN_HOUR,
)
from orchestrator.cli.utils.jsonpath.filters import remove_fields_from_dictionary
from orchestrator.cli.utils.output.prints import (
    ADO_GET_CONFIG_ONLY_WHEN_SINGLE_RESOURCE,
    ERROR,
    WARN,
    console_print,
)
from orchestrator.cli.utils.pydantic.constants import (
    event_importance_order,
    minimize_output_context,
)
from orchestrator.core import (
    ADOResource,
    CoreResourceKinds,
    DiscoverySpaceResource,
    OperationResource,
)
from orchestrator.core.metadata import ConfigurationMetadata
from orchestrator.core.operation.resource import (
    OperationResourceEventEnum,
    OperationResourceStatus,
)
from orchestrator.core.resources import ADOResourceEventEnum, ADOResourceStatus
from orchestrator.utilities.output import (
    printable_pydantic_model,
)

if typing.TYPE_CHECKING:
    import pandas as pd

    from orchestrator.cli.models.parameters import AdoGetCommandParameters


def format_default_ado_get_single_resource(
    resource: ADOResource, show_details: bool
) -> "pd.DataFrame":
    import json

    import pandas as pd

    columns = (
        ["IDENTIFIER", "NAME", "AGE"]
        if not show_details
        else ["IDENTIFIER", "NAME", "DESCRIPTION", "LABELS", "AGE"]
    )

    if isinstance(resource, OperationResource):
        columns.insert(-1, "STATUS")
        columns.insert(-1, "EXIT_STATE")

    if not resource:
        return pd.DataFrame(columns=columns)

    metadata = resource.config.metadata or ConfigurationMetadata()
    output = {
        "IDENTIFIER": resource.identifier,
        "NAME": f'"{metadata.name}"' if metadata.name else None,
        "AGE": timedelta_to_string(
            time_since_timestamp(resource.created).total_seconds()
        ),
    }

    if show_details:
        output["DESCRIPTION"] = (
            f'"{metadata.description}"' if metadata.description else None
        )
        output["LABELS"] = json.dumps(metadata.labels) if metadata.labels else None

    if isinstance(resource, OperationResource):
        status_update = most_important_status_update(resource.status)
        output["STATUS"] = status_update.event.value
        output["EXIT_STATE"] = (
            status_update.exit_state.value
            if isinstance(status_update.event, OperationResourceEventEnum)
            and status_update.exit_state is not None
            else "N/A"
        )

    # AP: if we don't set the index manually, pandas will complain with
    #   ValueError: If using all scalar values, you must pass an index
    # We also use the columns array to reorder the columns
    return pd.DataFrame(output, index=[0])[columns]


def format_default_ado_get_multiple_resources(
    resources: "pd.DataFrame", resource_kind: CoreResourceKinds
) -> "pd.DataFrame":
    if resources.empty:
        return resources

    # AP 13-12-2024:
    # Currently only Operations support status updates.
    # We try to keep it flexible.
    status_model = pydantic.RootModel[list[ADOResourceStatus]]
    if resource_kind == CoreResourceKinds.OPERATION:
        status_model = pydantic.RootModel[list[OperationResourceStatus]]

    # AP 13-12-2024:
    # The exit state column should be there just for operations
    # we do some trickery to ensure we put it before age
    columns = list(resources.columns)

    if resource_kind == CoreResourceKinds.OPERATION:
        columns.insert(-1, "EXIT_STATE")

        resources["STATUS"] = resources["STATUS"].apply(
            lambda x: most_important_status_update(
                status_model.model_validate(json.loads(x)).root if x else None
            )
        )

        resources["EXIT_STATE"] = resources["STATUS"].apply(
            lambda x: (
                x.exit_state.value
                if isinstance(x.event, OperationResourceEventEnum)
                and x.exit_state is not None
                else "N/A"
            )
        )

    if "STATUS" in resources.columns:
        resources["STATUS"] = resources["STATUS"].apply(lambda x: x.event.value)

    # AP: the default formatting of timedelta objects is too verbose
    # we convert it to
    if "AGE" in resources.columns:

        resources["AGE"] = resources["AGE"].apply(
            lambda x: timedelta_to_string(x.total_seconds())
        )

    return resources[columns]


def format_resource_for_ado_get_custom_format(
    to_print: (
        ADOResource
        | list[ADOResource]
        | pydantic.BaseModel
        | list[pydantic.BaseModel]
        | dict
    ),
    parameters: "AdoGetCommandParameters",
) -> str:
    match parameters.output_format:
        case AdoGetSupportedOutputFormats.CONFIG:
            return _config_formatter_for_ado_resource(
                to_print=to_print, parameters=parameters
            )
        case AdoGetSupportedOutputFormats.YAML:
            return _yaml_formatter_for_ado_resource(
                to_print=to_print, parameters=parameters
            )
        case AdoGetSupportedOutputFormats.JSON:
            return _json_formatter_for_ado_resource(
                to_print=to_print, parameters=parameters
            )
        case AdoGetSupportedOutputFormats.RAW:
            return _raw_formatter_for_ado_resource(
                to_print=to_print, parameters=parameters
            )
        case _:
            raise ValueError(
                f"Output format {parameters.output_format.value} is not supported."
            )


def _config_formatter_for_ado_resource(
    to_print: (
        ADOResource
        | list[ADOResource]
        | pydantic.BaseModel
        | list[pydantic.BaseModel]
        | dict
    ),
    parameters: "AdoGetCommandParameters",
) -> str:

    if isinstance(to_print, list):
        console_print(f"{ERROR}{ADO_GET_CONFIG_ONLY_WHEN_SINGLE_RESOURCE}", stderr=True)
        raise typer.Exit(1)

    if not hasattr(to_print, "config"):
        console_print(
            f"{ERROR}The resource requested does not have a config field.", stderr=True
        )
        raise typer.Exit(1)

    if parameters.minimize_output:
        serialization_context = minimize_output_context
        serialization_target = _minimize_ado_resource_representation(to_print).config
    else:
        serialization_context = None
        serialization_target = to_print.config

    # To handle lists correctly, we need to select all items of the list
    if parameters.exclude_fields and not parameters.resource_id:
        parameters.exclude_fields = [
            f"[*].{field_exclusion}" for field_exclusion in parameters.exclude_fields
        ]

    # AP: 28/07/2025:
    # We can't simply use model_dump because we would end up with errors like:
    #    RepresenterError: ('cannot represent an object', <ADOResourceEventEnum.CREATED: 'created'>)
    # when calling yaml.safe_dump
    dict_representation = yaml.safe_load(
        printable_pydantic_model(serialization_target).model_dump_json(
            exclude_none=parameters.exclude_none,
            exclude_unset=parameters.exclude_unset,
            exclude_defaults=parameters.exclude_default,
            context=serialization_context,
        )
    )

    if parameters.exclude_fields:
        dict_representation = remove_fields_from_dictionary(
            dict_representation, parameters.exclude_fields
        )

    return yaml.safe_dump(dict_representation)


def _yaml_formatter_for_ado_resource(
    to_print: (
        ADOResource
        | list[ADOResource]
        | pydantic.BaseModel
        | list[pydantic.BaseModel]
        | dict
    ),
    parameters: "AdoGetCommandParameters",
) -> str:

    if parameters.minimize_output:
        serialization_context = minimize_output_context
        serialization_target = _minimize_ado_resource_representation(to_print)
    else:
        serialization_context = None
        serialization_target = to_print

    # To handle lists correctly, we need to select all items of the list
    if parameters.exclude_fields and not parameters.resource_id:
        parameters.exclude_fields = [
            f"[*].{field_exclusion}" for field_exclusion in parameters.exclude_fields
        ]

    # AP: 28/07/2025:
    # We can't simply use model_dump because we would end up with errors like:
    #    RepresenterError: ('cannot represent an object', <ADOResourceEventEnum.CREATED: 'created'>)
    # when calling yaml.safe_dump
    dict_representation = yaml.safe_load(
        printable_pydantic_model(serialization_target).model_dump_json(
            exclude_none=parameters.exclude_none,
            exclude_unset=parameters.exclude_unset,
            exclude_defaults=parameters.exclude_default,
            context=serialization_context,
        )
    )

    if parameters.exclude_fields:
        dict_representation = remove_fields_from_dictionary(
            dict_representation, parameters.exclude_fields
        )

    return yaml.safe_dump(dict_representation)


def _json_formatter_for_ado_resource(
    to_print: (
        ADOResource | list[ADOResource] | pydantic.BaseModel | list[pydantic.BaseModel]
    ),
    parameters: "AdoGetCommandParameters",
) -> str:

    if parameters.minimize_output:
        serialization_context = minimize_output_context
        serialization_target = _minimize_ado_resource_representation(to_print)
    else:
        serialization_context = None
        serialization_target = to_print

    # When exclude_fields is False, we know our data is valid.
    # We don't need to do any processing other than
    # using printable_pydantic_model to handle lists.
    if not parameters.exclude_fields:
        return printable_pydantic_model(serialization_target).model_dump_json(
            indent=2,
            exclude_none=parameters.exclude_none,
            exclude_unset=parameters.exclude_unset,
            exclude_defaults=parameters.exclude_default,
            context=serialization_context,
        )

    # Here we need to remove some fields and this might
    # mean creating a model that's invalid.

    # To handle lists correctly, we need to select all items of the list
    if not parameters.resource_id:
        parameters.exclude_fields = [
            f"[*].{field_exclusion}" for field_exclusion in parameters.exclude_fields
        ]

    printable_model = printable_pydantic_model(serialization_target)
    filtered_representation = remove_fields_from_dictionary(
        input_dictionary=printable_model.model_dump(
            exclude_none=parameters.exclude_none,
            exclude_unset=parameters.exclude_unset,
            exclude_defaults=parameters.exclude_default,
            context=serialization_context,
        ),
        fields_to_remove=parameters.exclude_fields,
    )

    # AP: 28/07/2025:
    # pydantic's json serializer can handle more data types
    # so we need to construct a model on-the-fly.
    # We can't use `model_validate` as we might have removed
    # a required field.
    if isinstance(filtered_representation, list):
        model = printable_model.model_construct(filtered_representation)
    else:
        model = printable_model.model_construct(**filtered_representation)

    # AP: 30/09/2025
    # We set warnings="none" as otherwise we'd print a ton of:
    # PydanticSerializationUnexpectedValue(Expected `SOME_MODEL` -
    # serialized value may not be as expected [input_value={...}, input_type=dict])
    return model.model_dump_json(
        indent=2,
        warnings="none",
        exclude_none=parameters.exclude_none,
        exclude_unset=parameters.exclude_unset,
        exclude_defaults=parameters.exclude_default,
        context=serialization_context,
    )


def _raw_formatter_for_ado_resource(
    to_print: (
        ADOResource
        | list[ADOResource]
        | pydantic.BaseModel
        | list[pydantic.BaseModel]
        | dict
    ),
    parameters: "AdoGetCommandParameters",
) -> str:
    import pprint

    if parameters.minimize_output:
        console_print(
            f"{WARN}Minimizing output is not supported for the raw output type.",
            stderr=True,
        )

    return pprint.pformat(to_print)


def _minimize_ado_resource_representation(
    to_print: (
        ADOResource | list[ADOResource] | pydantic.BaseModel | list[pydantic.BaseModel]
    ),
) -> ADOResource | pydantic.BaseModel:
    if isinstance(to_print, list):
        console_print(
            f"{ERROR}The minimal output format can only be used "
            f"when a resource identifier is provided.",
            stderr=True,
        )
        raise typer.Exit(1)

    if isinstance(to_print, DiscoverySpaceResource):
        to_print.config = to_print.config.convert_experiments_to_reference_list()

    return to_print


def time_since_timestamp(ts: datetime.datetime) -> datetime.timedelta:
    # AP: there are still some datetimes that are not timezone aware
    return (
        datetime.datetime.now() - ts
        if not ts.tzinfo
        else datetime.datetime.now(tz=datetime.timezone.utc) - ts
    )


def most_important_status_update(
    statuses: list[OperationResourceStatus],
) -> OperationResourceStatus:

    if not statuses:
        return OperationResourceStatus(event=ADOResourceEventEnum.ADDED)

    status_updates = [s.event for s in statuses]
    for important_event in event_importance_order:
        if important_event in status_updates:
            idx = status_updates.index(important_event)
            return statuses[idx]

    return OperationResourceStatus(event=ADOResourceEventEnum.ADDED)


def timedelta_to_string(total_seconds: float) -> str:
    if math.isnan(total_seconds):
        return "NaT"
    if total_seconds < SECONDS_IN_A_MINUTE:
        return f"{round(total_seconds)}s"
    if total_seconds < SECONDS_IN_AN_HOUR:
        minutes, seconds = divmod(total_seconds, SECONDS_IN_A_MINUTE)
        return f"{int(minutes)}m{round(seconds)}s"
    if total_seconds < SECONDS_IN_A_DAY:
        hours, remainder = divmod(total_seconds, SECONDS_IN_AN_HOUR)
        minutes = remainder / SECONDS_IN_A_MINUTE
        return f"{int(hours)}h{round(minutes)}m"
    days, remainder = divmod(total_seconds, SECONDS_IN_A_DAY)
    hours = remainder / SECONDS_IN_AN_HOUR
    return f"{int(days)}d{round(hours)}h"
