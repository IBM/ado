# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from typing import Annotated

import pydantic


class OperationMeasurementStatistics(pydantic.BaseModel):
    """Aggregated measurement statistics for a single operation.

    Attributes:
        operation_id: The operation identifier these statistics belong to.
        total_requests: Total number of MeasurementRequests for the operation.
        failed_requests: Number of requests whose status is FAILED.
        successful_requests: Number of requests whose status is SUCCESS.
        total_results: Total number of measurement results across all requests.
        successful_results: Number of valid results (those carrying measurements).
        failed_results: Number of invalid results (those carrying a failure reason).
        measured_entities: Count of distinct entities that have at least one result.
    """

    operation_id: Annotated[
        str,
        pydantic.Field(
            description="The operation identifier these statistics belong to."
        ),
    ]
    total_requests: Annotated[
        int,
        pydantic.Field(
            description="Total number of MeasurementRequests for the operation."
        ),
    ]
    failed_requests: Annotated[
        int,
        pydantic.Field(description="Number of requests whose status is FAILED."),
    ]
    successful_requests: Annotated[
        int,
        pydantic.Field(description="Number of requests whose status is SUCCESS."),
    ]
    total_results: Annotated[
        int,
        pydantic.Field(
            description="Total number of measurement results across all requests."
        ),
    ]
    successful_results: Annotated[
        int,
        pydantic.Field(
            description="Number of valid results (those carrying measurements)."
        ),
    ]
    failed_results: Annotated[
        int,
        pydantic.Field(
            description="Number of invalid results (those carrying a failure reason)."
        ),
    ]
    measured_entities: Annotated[
        int,
        pydantic.Field(
            description="Count of distinct entities that have at least one result."
        ),
    ]
