# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from fastapi import HTTPException, status

from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.request import MeasurementRequest

requests_memory_storage: dict[ExperimentReference, dict[str, MeasurementRequest]] = {}


def set_request_in_memory_storage(measurement_request: MeasurementRequest):
    if measurement_request.experimentReference not in requests_memory_storage:
        requests_memory_storage[measurement_request.experimentReference] = {}

    requests_memory_storage[measurement_request.experimentReference][
        measurement_request.requestid
    ] = measurement_request


def get_all_requests_in_memory_storage(
    experiment_reference: ExperimentReference,
) -> list[MeasurementRequest]:
    if experiment_reference not in requests_memory_storage:
        return []

    return list(requests_memory_storage[experiment_reference].values())


def get_request_in_memory_storage(
    experiment_reference: ExperimentReference, request_id: str
) -> MeasurementRequest:
    if (
        experiment_reference not in requests_memory_storage
        or request_id not in requests_memory_storage[experiment_reference]
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Request {request_id} not found for {experiment_reference}",
        )

    return requests_memory_storage[experiment_reference][request_id]
