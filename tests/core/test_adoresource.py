# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from orchestrator.core import DiscoverySpaceResource, SampleStoreResource
from orchestrator.core.operation.resource import (
    OperationExitStateEnum,
    OperationResource,
    OperationResourceEventEnum,
    OperationResourceStatus,
)
from orchestrator.core.resources import ADOResourceEventEnum, ADOResourceStatus


def test_ado_resource_status() -> None:

    # Create an event status and check expected fields are set
    status = ADOResourceStatus(event=ADOResourceEventEnum.UPDATED)
    assert status.event == ADOResourceEventEnum.UPDATED
    assert status.recorded_at

    dump = status.model_dump()
    assert dump.get("event")

    # Add message and check it is dumped
    status = ADOResourceStatus(event=ADOResourceEventEnum.UPDATED, message="Test")
    assert status.message
    dump = status.model_dump()
    assert dump.get("event")

    # Check deser
    deser = ADOResourceStatus.model_validate(dump)
    assert deser.event == status.event
    assert deser.recorded_at == status.recorded_at
    assert deser.message == status.message


def test_resource_has_default_status(
    sample_store_resource: SampleStoreResource,
    discovery_space_resource: DiscoverySpaceResource,
    operation_resource: OperationResource,
) -> None:

    assert len(sample_store_resource.status) == 1
    assert sample_store_resource.status[0].event == ADOResourceEventEnum.CREATED
    assert len(discovery_space_resource.status) == 1
    assert discovery_space_resource.status[0].event == ADOResourceEventEnum.CREATED
    assert len(discovery_space_resource.status) == 1
    assert operation_resource.status[0].event == ADOResourceEventEnum.CREATED


def test_resource_has_status_ser_deser(
    sample_store_resource: SampleStoreResource,
    discovery_space_resource: DiscoverySpaceResource,
    operation_resource: OperationResource,
) -> None:
    """Check that status types are correct serialized and deserialized with a resource"""

    dump = sample_store_resource.model_dump()
    deser = SampleStoreResource.model_validate(dump)
    assert len(deser.status) == 1
    assert deser.status[0].event == ADOResourceEventEnum.CREATED

    dump = discovery_space_resource.model_dump()
    deser = DiscoverySpaceResource.model_validate(dump)
    assert len(deser.status) == 1
    assert deser.status[0].event == ADOResourceEventEnum.CREATED

    # Add an exit-status to operation resource
    status = OperationResourceStatus(
        event=OperationResourceEventEnum.FINISHED,
        exit_state=OperationExitStateEnum.FAIL,
    )
    operation_resource.status.append(status)
    dump = operation_resource.model_dump()
    deser = OperationResource.model_validate(dump)
    assert len(deser.status) == 2
    assert deser.status[0].event == ADOResourceEventEnum.CREATED
    assert deser.status[1].event == OperationResourceEventEnum.FINISHED
    assert deser.status[1].exit_state == OperationExitStateEnum.FAIL
