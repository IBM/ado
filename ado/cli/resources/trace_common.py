# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Shared column definitions, hidable-field maps, and row builders for all
``ado show trace`` handlers.
"""

import enum

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------


class REQUEST_COLUMN(enum.Enum):
    """Column names for request-level trace view."""

    REQUEST_ID = "Request ID"
    OPERATION_ID = "Operation ID"
    SPACE_ID = "Space ID"
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


class RESULT_COLUMN(enum.Enum):
    """Column names for result-level (unrolled-entities) trace view."""

    REQUEST_ID = "Request ID"
    OPERATION_ID = "Operation ID"
    SPACE_ID = "Space ID"
    REQUEST_INDEX = "Request Index"
    REQUEST_TYPE = "Request type"
    TIMESTAMP = "Timestamp"
    EXPERIMENT_ID = "Experiment ID"
    RESULT_INDEX = "Result Index"
    RESULT_UID = "Result UID"
    ENTITY_ID = "Entity ID"
    VALID = "Valid"
    NUMBER_OF_PROPERTIES = "Number of Properties"
    INVALID_REASON = "Invalid Reason"
    REQUEST_METADATA = "Request Metadata"
    RESULT_METADATA = "Result Metadata"


# ---------------------------------------------------------------------------
# Hidable field maps
# ---------------------------------------------------------------------------

REQUEST_HIDABLE_FIELDS: dict[str, str] = {
    **dict.fromkeys(["request id", "id"], REQUEST_COLUMN.REQUEST_ID.value),
    **dict.fromkeys(["operation id", "operation"], REQUEST_COLUMN.OPERATION_ID.value),
    **dict.fromkeys(["space id", "space"], REQUEST_COLUMN.SPACE_ID.value),
    **dict.fromkeys(["request index"], REQUEST_COLUMN.REQUEST_INDEX.value),
    **dict.fromkeys(["type", "request type"], REQUEST_COLUMN.REQUEST_TYPE.value),
    **dict.fromkeys(["timestamp", "time"], REQUEST_COLUMN.TIMESTAMP.value),
    **dict.fromkeys(
        ["experiment id", "experiment"], REQUEST_COLUMN.EXPERIMENT_ID.value
    ),
    **dict.fromkeys(
        ["entity", "entities", "entity ids"], REQUEST_COLUMN.ENTITY_IDS.value
    ),
    **dict.fromkeys(["status"], REQUEST_COLUMN.STATUS.value),
    **dict.fromkeys(["measurements"], REQUEST_COLUMN.MEASUREMENTS.value),
    **dict.fromkeys(
        ["valid", "valid measurements"], REQUEST_COLUMN.VALID_MEASUREMENTS.value
    ),
    **dict.fromkeys(
        ["invalid", "invalid measurements"],
        REQUEST_COLUMN.INVALID_MEASUREMENTS.value,
    ),
    **dict.fromkeys(["meta", "metadata"], REQUEST_COLUMN.METADATA.value),
}

RESULT_HIDABLE_FIELDS: dict[str, str] = {
    **dict.fromkeys(["request id", "id"], RESULT_COLUMN.REQUEST_ID.value),
    **dict.fromkeys(["operation id", "operation"], RESULT_COLUMN.OPERATION_ID.value),
    **dict.fromkeys(["space id", "space"], RESULT_COLUMN.SPACE_ID.value),
    **dict.fromkeys(["request index"], RESULT_COLUMN.REQUEST_INDEX.value),
    **dict.fromkeys(["type", "request type"], RESULT_COLUMN.REQUEST_TYPE.value),
    **dict.fromkeys(["timestamp", "time"], RESULT_COLUMN.TIMESTAMP.value),
    **dict.fromkeys(["experiment id", "experiment"], RESULT_COLUMN.EXPERIMENT_ID.value),
    **dict.fromkeys(["result index"], RESULT_COLUMN.RESULT_INDEX.value),
    **dict.fromkeys(["result uid", "uid"], RESULT_COLUMN.RESULT_UID.value),
    **dict.fromkeys(["entity", "entity id"], RESULT_COLUMN.ENTITY_ID.value),
    **dict.fromkeys(["valid"], RESULT_COLUMN.VALID.value),
    **dict.fromkeys(
        ["number of properties", "properties"],
        RESULT_COLUMN.NUMBER_OF_PROPERTIES.value,
    ),
    **dict.fromkeys(
        ["invalid reason", "reason", "invalid"],
        RESULT_COLUMN.INVALID_REASON.value,
    ),
    **dict.fromkeys(
        ["request metadata", "request meta"],
        RESULT_COLUMN.REQUEST_METADATA.value,
    ),
    **dict.fromkeys(
        ["result metadata", "result meta", "meta", "metadata"],
        RESULT_COLUMN.RESULT_METADATA.value,
    ),
}

# ---------------------------------------------------------------------------
# Columns reordered to the end of the DataFrame
# ---------------------------------------------------------------------------

REQUEST_COLUMNS_MOVE_TO_END: list[str] = [REQUEST_COLUMN.METADATA.value]

RESULT_COLUMNS_MOVE_TO_END: list[str] = [
    RESULT_COLUMN.INVALID_REASON.value,
    RESULT_COLUMN.REQUEST_METADATA.value,
    RESULT_COLUMN.RESULT_METADATA.value,
]


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def build_request_level_rows(
    measurement_requests: list,
    *,
    include_operation_id: bool = False,
    operation_space_map: dict[str, str] | None = None,
) -> list[dict]:
    """Build rows for request-level view.

    Args:
        measurement_requests: List of MeasurementRequest objects to build rows from.
        include_operation_id: When True, stamp each row with the request's own
            operation ID (taken from ``request.operation_id``) under the
            ``Operation ID`` column.
        operation_space_map: When not None, a mapping from operation ID to space ID.
            Each row is stamped with the space ID for its operation under the
            ``Space ID`` column.

    Returns:
        A list of row dicts suitable for constructing a pandas DataFrame.
    """
    from ado.schema.request import ReplayedMeasurement
    from ado.schema.result import (
        InvalidMeasurementResult,
        ValidMeasurementResult,
    )

    rows = []
    for request in measurement_requests:
        row: dict = {REQUEST_COLUMN.REQUEST_ID.value: request.requestid}
        if include_operation_id:
            row[REQUEST_COLUMN.OPERATION_ID.value] = request.operation_id
        if operation_space_map is not None:
            row[REQUEST_COLUMN.SPACE_ID.value] = operation_space_map.get(
                request.operation_id
            )
        row.update(
            {
                REQUEST_COLUMN.REQUEST_INDEX.value: request.requestIndex,
                REQUEST_COLUMN.REQUEST_TYPE.value: (
                    "replayed"
                    if isinstance(request, ReplayedMeasurement)
                    else "measured"
                ),
                REQUEST_COLUMN.TIMESTAMP.value: request.timestamp,
                REQUEST_COLUMN.EXPERIMENT_ID.value: request.experimentReference,
                REQUEST_COLUMN.ENTITY_IDS.value: [
                    entity.identifier for entity in request.entities
                ],
                REQUEST_COLUMN.STATUS.value: request.status.value,
                REQUEST_COLUMN.MEASUREMENTS.value: len(request.measurements or []),
                REQUEST_COLUMN.VALID_MEASUREMENTS.value: len(
                    [
                        r
                        for r in (request.measurements or [])
                        if isinstance(r, ValidMeasurementResult)
                    ]
                ),
                REQUEST_COLUMN.INVALID_MEASUREMENTS.value: len(
                    [
                        r
                        for r in (request.measurements or [])
                        if isinstance(r, InvalidMeasurementResult)
                    ]
                ),
                REQUEST_COLUMN.METADATA.value: request.metadata,
            }
        )
        rows.append(row)
    return rows


def build_result_level_rows(
    measurement_requests: list,
    *,
    include_operation_id: bool = False,
    operation_space_map: dict[str, str] | None = None,
) -> list[dict]:
    """Build rows for result-level view with unrolled entities.

    Args:
        measurement_requests: List of MeasurementRequest objects to build rows from.
        include_operation_id: When True, stamp each row with the request's own
            operation ID (taken from ``request.operation_id``) under the
            ``Operation ID`` column.
        operation_space_map: When not None, a mapping from operation ID to space ID.
            Each row is stamped with the space ID for its operation under the
            ``Space ID`` column.

    Returns:
        A list of row dicts suitable for constructing a pandas DataFrame.
    """
    from ado.schema.request import ReplayedMeasurement
    from ado.schema.result import (
        InvalidMeasurementResult,
        ValidMeasurementResult,
    )

    rows = []
    for request in measurement_requests:
        for result_idx, result in enumerate(request.measurements or []):
            row: dict = {RESULT_COLUMN.REQUEST_ID.value: request.requestid}
            if include_operation_id:
                row[RESULT_COLUMN.OPERATION_ID.value] = request.operation_id
            if operation_space_map is not None:
                row[RESULT_COLUMN.SPACE_ID.value] = operation_space_map.get(
                    request.operation_id
                )
            row.update(
                {
                    RESULT_COLUMN.REQUEST_INDEX.value: request.requestIndex,
                    RESULT_COLUMN.REQUEST_TYPE.value: (
                        "replayed"
                        if isinstance(request, ReplayedMeasurement)
                        else "measured"
                    ),
                    RESULT_COLUMN.TIMESTAMP.value: request.timestamp,
                    RESULT_COLUMN.EXPERIMENT_ID.value: request.experimentReference,
                    RESULT_COLUMN.RESULT_INDEX.value: result_idx,
                    RESULT_COLUMN.RESULT_UID.value: result.uid,
                    RESULT_COLUMN.ENTITY_ID.value: result.entityIdentifier,
                    RESULT_COLUMN.VALID.value: isinstance(
                        result, ValidMeasurementResult
                    ),
                    RESULT_COLUMN.REQUEST_METADATA.value: request.metadata,
                    RESULT_COLUMN.RESULT_METADATA.value: result.metadata,
                }
            )

            if isinstance(result, ValidMeasurementResult):
                row[RESULT_COLUMN.NUMBER_OF_PROPERTIES.value] = len(
                    {m.property.identifier for m in result.measurements}
                )
            elif isinstance(result, InvalidMeasurementResult):
                row[RESULT_COLUMN.INVALID_REASON.value] = result.reason

            rows.append(row)
    return rows


# Made with Bob
