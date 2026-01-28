# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import enum
import typing
import uuid
from typing import Annotated

import pydantic

from orchestrator.core.operation.config import (
    DiscoveryOperationEnum,
    DiscoveryOperationResourceConfiguration,
)
from orchestrator.core.resources import (
    ADOResource,
    ADOResourceEventEnum,
    ADOResourceStatus,
    CoreResourceKinds,
)


class OperationResourceEventEnum(enum.Enum):
    """Additional events in OperationResource lifecycle"""

    STARTED = "started"
    FINISHED = "finished"


class OperationExitStateEnum(enum.Enum):
    """Enumerates the possible exit-states of an operation when it finishes"""

    SUCCESS = "success"  # The operation returned with success
    FAIL = "fail"  # The operation returned with failure
    ERROR = "error"  # Some exception was raised during operation


class OperationResourceStatus(ADOResourceStatus):
    """Records information on the status of an operation resource - a life-cycle event that occurred or an exit status"""

    event: Annotated[
        ADOResourceEventEnum | OperationResourceEventEnum,
        pydantic.Field(
            description="An event that happened to an operation resource: created, added, started, finished, updated"
        ),
    ] = None
    exit_state: Annotated[
        OperationExitStateEnum | None,
        pydantic.Field(
            description="The exit state of the operation: success, failed, error. Only can be set if on a FINISHED event"
        ),
    ] = None

    @pydantic.model_validator(mode="after")
    def check_status(self) -> "OperationResourceStatus":

        if self.exit_state and self.event != OperationResourceEventEnum.FINISHED:
            raise ValueError(
                f"Recording an exit state (here {self.exit_state}) for an operation resource status, "
                f"requires recording a corresponding FINISHED event ({self.event} given)"
            )

        return self


class OperationResource(ADOResource):

    version: Annotated[str, pydantic.Field()] = "v1"
    kind: Annotated[CoreResourceKinds, pydantic.Field()] = CoreResourceKinds.OPERATION
    operationType: Annotated[
        DiscoveryOperationEnum, pydantic.Field(description="The type of this operation")
    ]
    operatorIdentifier: Annotated[
        str,
        pydantic.Field(
            description="The id of the operator resource that executed this operation"
        ),
    ]
    config: DiscoveryOperationResourceConfiguration
    status: Annotated[
        list[OperationResourceStatus],
        pydantic.Field(
            default_factory=lambda: [
                OperationResourceStatus(event=ADOResourceEventEnum.CREATED)
            ],
            description="A list of status objects",
        ),
    ]

    @pydantic.model_validator(mode="before")
    @classmethod
    def generate_identifier_if_not_provided(
        cls, data: typing.Any  # noqa: ANN401
    ) -> "OperationResource":

        if isinstance(data, dict):

            # Do not do anything if the identifier is already present
            if data.get("identifier", None) is not None:
                return data

            # Do not attempt to generate anything if operatorIdentifier
            # (a required field) has not been provided
            if "operatorIdentifier" not in data:
                return data

            kind = CoreResourceKinds.OPERATION.value
            operator_identifier = data["operatorIdentifier"]
            data["identifier"] = f"{kind}-{operator_identifier}-{str(uuid.uuid4())[:8]}"

        return data
