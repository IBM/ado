# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import enum

import typer
from rich.status import Status

from orchestrator.cli.models.parameters import AdoShowRequestsCommandParameters
from orchestrator.cli.utils.output.dataframes import df_to_output
from orchestrator.cli.utils.output.prints import (
    ADO_SPINNER_QUERYING_DB,
    ERROR,
    console_print,
)
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.metastore.base import (
    NoRelatedResourcesError,
    ResourceDoesNotExistError,
)
from orchestrator.schema.request import ReplayedMeasurement
from orchestrator.schema.result import ValidMeasurementResult


def show_operation_requests(parameters: AdoShowRequestsCommandParameters) -> None:

    class _COLUMN(enum.Enum):
        REQUEST_ID = "Request ID"
        REQUEST_INDEX = "Request Index"
        REQUEST_TYPE = "Request type"
        TIMESTAMP = "Timestamp"
        EXPERIMENT_ID = "Experiment ID"
        ENTITY_IDS = "Entity IDs"
        STATUS = "Status"
        MEASUREMENTS = "Measurements"
        VALID_MEASUREMENTS = "Valid Measurements"
        INVALID_MEASUREMENTS = "Invalid Measurements"
        METADATA = "Metadata"

    hidable_fields = {
        **dict.fromkeys(["request id", "id"], _COLUMN.REQUEST_ID.value),
        **dict.fromkeys(["request index", "idx"], _COLUMN.REQUEST_INDEX.value),
        **dict.fromkeys(["type", "request type"], _COLUMN.REQUEST_TYPE.value),
        **dict.fromkeys(["timestamp"], _COLUMN.TIMESTAMP.value),
        **dict.fromkeys(["experiment id"], _COLUMN.EXPERIMENT_ID.value),
        **dict.fromkeys(["entity", "entities", "entity ids"], _COLUMN.ENTITY_IDS.value),
        **dict.fromkeys(["status"], _COLUMN.STATUS.value),
        **dict.fromkeys(["measurements"], _COLUMN.MEASUREMENTS.value),
        **dict.fromkeys(
            ["valid", "valid measurements"], _COLUMN.VALID_MEASUREMENTS.value
        ),
        **dict.fromkeys(
            ["invalid", "invalid measurements"], _COLUMN.INVALID_MEASUREMENTS.value
        ),
        **dict.fromkeys(["meta", "metadata"], _COLUMN.METADATA.value),
    }

    if parameters.hide_fields:
        for idx, field in enumerate(parameters.hide_fields):
            if field.lower() not in hidable_fields:
                console_print(
                    f"{ERROR}You can only hide the following fields (case insensitive): "
                    f"{list(hidable_fields.keys())}",
                    stderr=True,
                )
                raise typer.Exit(1)
            parameters.hide_fields[idx] = hidable_fields[field.lower()]

    file_name = f"measurement_requests_for_operation_{parameters.resource_id}.{parameters.output_format.value}"

    with Status(ADO_SPINNER_QUERYING_DB) as status:
        try:
            space = DiscoverySpace.from_operation_id(
                operation_id=parameters.resource_id,
                project_context=parameters.ado_configuration.project_context,
            )
        except (ResourceDoesNotExistError, NoRelatedResourcesError):
            status.stop()
            raise

        status.update("Fetching measurements")
        measurement_requests = space.measurement_requests_for_operation(
            operation_id=parameters.resource_id,
        )

    rows = [
        {
            _COLUMN.REQUEST_ID.value: request.requestid,
            _COLUMN.REQUEST_INDEX.value: request.requestIndex,
            _COLUMN.REQUEST_TYPE.value: (
                "replayed" if isinstance(request, ReplayedMeasurement) else "measured"
            ),
            _COLUMN.TIMESTAMP.value: request.timestamp,
            _COLUMN.EXPERIMENT_ID.value: request.experimentReference,
            _COLUMN.ENTITY_IDS.value: [
                entity.identifier for entity in request.entities
            ],
            _COLUMN.STATUS.value: request.status.value,
            _COLUMN.MEASUREMENTS.value: len(request.measurements),
            _COLUMN.VALID_MEASUREMENTS.value: len(
                [
                    r
                    for r in request.measurements
                    if isinstance(r, ValidMeasurementResult)
                ]
            ),
            _COLUMN.INVALID_MEASUREMENTS.value: len(
                [
                    r
                    for r in request.measurements
                    if not isinstance(r, ValidMeasurementResult)
                ]
            ),
            _COLUMN.METADATA.value: request.metadata,
        }
        for request in measurement_requests
    ]

    import pandas as pd

    df = pd.DataFrame(rows)
    if parameters.hide_fields:
        df = df.drop(parameters.hide_fields, axis="columns")

    df_to_output(
        df=df, output_format=parameters.output_format.value, file_name=file_name
    )
