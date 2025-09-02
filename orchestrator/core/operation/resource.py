# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import enum
import uuid

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

    event: ADOResourceEventEnum | OperationResourceEventEnum = pydantic.Field(
        default=None,
        description="An event that happened to an operation resource: created, added, started, finished, updated",
    )
    exit_state: OperationExitStateEnum | None = pydantic.Field(
        default=None,
        description="The exit state of the operation: success, failed, error. Only can be set if on a FINISHED event",
    )

    @pydantic.model_validator(mode="after")
    def check_status(self):

        if self.exit_state:
            assert self.event == OperationResourceEventEnum.FINISHED, (
                f"Recording an exit state (here {self.exit_state}) for an operation resource status, "
                f"requires recording a corresponding FINISHED event ({self.event} given)"
            )

        return self


class OperationResource(ADOResource):

    version: str = "v1"
    kind: CoreResourceKinds = CoreResourceKinds.OPERATION
    operationType: DiscoveryOperationEnum = pydantic.Field(
        description="The type of this operation"
    )
    operatorIdentifier: str = pydantic.Field(
        description="The id of the operator resource that executed this operation"
    )
    config: DiscoveryOperationResourceConfiguration
    status: list[OperationResourceStatus] = pydantic.Field(
        default=[OperationResourceStatus(event=ADOResourceEventEnum.CREATED)],
        description="A list of status objects",
    )

    @pydantic.model_validator(mode="after")
    def generate_identifier_if_not_provided(self):

        if self.identifier is None:
            self.identifier = (
                f"{self.kind.value}-{self.operatorIdentifier}-{str(uuid.uuid4())[:8]}"
            )

        return self
