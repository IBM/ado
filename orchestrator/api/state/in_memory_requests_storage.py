"""In-memory storage helpers for `MeasurementRequest` objects.

The storage is keyed first by the experiment reference that the
request has been created from and then by the request's unique ID.

- ``set_request_in_memory_storage``: Persist a request.
- ``get_all_requests_in_memory_storage``: Retrieve all requests for a
  given experiment.
- ``get_request_in_memory_storage``: Retrieve a single request by ID.
"""

# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from fastapi import HTTPException, status

from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.request import MeasurementRequest

# The in-memory store: experiment reference → request ID → request instance.
requests_memory_storage: dict[ExperimentReference, dict[str, MeasurementRequest]] = {}


def set_request_in_memory_storage(measurement_request: MeasurementRequest) -> None:
    """Persist a ``MeasurementRequest`` in the in-memory store.

    Args:
      measurement_request (MeasurementRequest): The request object to persist.
        Its ``experimentReference`` field is used as the first-level key and
        ``requestid`` as the second.
    """
    if measurement_request.experimentReference not in requests_memory_storage:
        requests_memory_storage[measurement_request.experimentReference] = {}

    requests_memory_storage[measurement_request.experimentReference][
        measurement_request.requestid
    ] = measurement_request


def get_all_requests_in_memory_storage(
    experiment_reference: ExperimentReference,
) -> list[MeasurementRequest]:
    """Return all requests belonging to *experiment_reference*.

    If the referenced experiment is not present, an empty list is returned.

    Args:
      experiment_reference (ExperimentReference): The experiment whose
        requests should be listed.

    Returns:
      list[MeasurementRequest]: All requests for the given experiment.
    """
    if experiment_reference not in requests_memory_storage:
        return []

    return list(requests_memory_storage[experiment_reference].values())


def get_request_in_memory_storage(
    experiment_reference: ExperimentReference, request_id: str
) -> MeasurementRequest:
    """Retrieve a specific request by *request_id* for the given experiment.

    Raises:
      fastapi.HTTPException: If the experiment does not exist or the
        ``request_id`` cannot be found.

    Args:
      experiment_reference (ExperimentReference): The experiment the
        request belongs to.
      request_id (str): The unique identifier of the desired request.

    Returns:
      MeasurementRequest: The requested measurement request.
    """
    if (
        experiment_reference not in requests_memory_storage
        or request_id not in requests_memory_storage[experiment_reference]
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Request {request_id} not found for {experiment_reference}",
        )

    return requests_memory_storage[experiment_reference][request_id]
