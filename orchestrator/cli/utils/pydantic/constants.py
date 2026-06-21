# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT


from orchestrator.core.operation.resource import OperationResourceEventEnum
from orchestrator.core.resources import ADOResourceEventEnum

minimize_output_context = {"minimize_output": True}
event_importance_order = [
    OperationResourceEventEnum.FINISHED,
    OperationResourceEventEnum.STARTED,
    ADOResourceEventEnum.UPDATED,
    ADOResourceEventEnum.ADDED,
    ADOResourceEventEnum.CREATED,
]
