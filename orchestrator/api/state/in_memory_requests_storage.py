# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from fastapi import HTTPException, status

from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.request import MeasurementRequest

requests_memory_storage: dict[ExperimentReference, dict[str, MeasurementRequest]] = {}
response_memory_storage: dict[ExperimentReference, dict[str, MeasurementRequest]] = {}


def set_request_in_memory_storage(
    experiment_reference: ExperimentReference,
    request_id: str,
    measurement_request: MeasurementRequest,
):
    if experiment_reference not in requests_memory_storage:
        requests_memory_storage[experiment_reference] = {}

    requests_memory_storage[experiment_reference][request_id] = measurement_request


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
        or request_id not in requests_memory_storage
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Request {request_id} was not found",
        )

    return requests_memory_storage[experiment_reference][request_id]
