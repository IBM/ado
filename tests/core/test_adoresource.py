# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pydantic
import pytest

from ado.core import DiscoverySpaceResource, SampleStoreResource
from ado.core.operation.resource import (
    OperationExitStateEnum,
    OperationResource,
    OperationResourceEventEnum,
    OperationResourceStatus,
)
from ado.core.resources import (
    ADOResourceEventEnum,
    ADOResourcePropertyDescriptor,
    ADOResourceReference,
    ADOResourceStatus,
    CoreResourceKinds,
)


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


def test_resource_provenance_ser_deser(
    sample_store_resource: SampleStoreResource,
) -> None:
    """Check that provenance is serialized and deserialized with a resource."""
    from ado.core.metadata import PackageProvenance, ProvenanceInfo

    ado = PackageProvenance(
        distributionName="ado-core",
        distributionVersion="1.2.3",
    )
    resource = sample_store_resource.model_copy(
        update={"provenance": ProvenanceInfo(ado=ado)}
    )
    dump = resource.model_dump()
    deser = SampleStoreResource.model_validate(dump)
    assert deser.provenance.ado == ado


def test_ado_resource_reference_round_trip() -> None:
    """ADOResourceReference survives model_dump → model_validate."""
    original = ADOResourceReference(
        identifier="space-abc123",
        kind=CoreResourceKinds.DISCOVERYSPACE,
    )
    restored = ADOResourceReference.model_validate(original.model_dump())
    assert restored == original

    without_kind = ADOResourceReference(identifier="space-abc123")
    restored_without_kind = ADOResourceReference.model_validate(
        without_kind.model_dump()
    )
    assert restored_without_kind.kind is None


def test_ado_resource_reference_rejects_unknown_fields() -> None:
    """ADOResourceReference rejects fields other than identifier and kind."""
    with pytest.raises(pydantic.ValidationError):
        ADOResourceReference.model_validate(
            {"identifier": "space-abc123", "unknown": "value"}
        )


def test_ado_resource_reference(
    sample_store_resource: SampleStoreResource,
    discovery_space_resource: DiscoverySpaceResource,
    operation_resource: OperationResource,
) -> None:
    """ADOResource.reference returns an ADOResourceReference for the resource."""
    for resource, kind in (
        (sample_store_resource, CoreResourceKinds.SAMPLESTORE),
        (discovery_space_resource, CoreResourceKinds.DISCOVERYSPACE),
        (operation_resource, CoreResourceKinds.OPERATION),
    ):
        ref = resource.reference
        assert isinstance(ref, ADOResourceReference)
        assert ref.identifier == resource.identifier
        assert ref.kind == kind


def test_ado_resource_property_descriptor_basics() -> None:
    """ADOResourcePropertyDescriptor carries identifier and kind."""
    descriptor = ADOResourcePropertyDescriptor(
        identifier="discoverySpace", kind=CoreResourceKinds.DISCOVERYSPACE
    )
    assert descriptor.identifier == "discoverySpace"
    assert descriptor.kind == CoreResourceKinds.DISCOVERYSPACE


def test_ado_resource_property_descriptor_round_trip() -> None:
    """ADOResourcePropertyDescriptor survives model_dump → model_validate."""
    original = ADOResourcePropertyDescriptor(
        identifier="candidate", kind=CoreResourceKinds.OPERATION
    )
    restored = ADOResourcePropertyDescriptor.model_validate(original.model_dump())
    assert restored.identifier == original.identifier
    assert restored.kind == original.kind


def test_ado_resource_property_descriptor_extra_fields_forbidden() -> None:
    """ADOResourcePropertyDescriptor rejects unknown fields."""
    with pytest.raises(pydantic.ValidationError):
        ADOResourcePropertyDescriptor.model_validate(
            {"identifier": "x", "kind": "discoveryspace", "unknown": 1}
        )


def test_ado_resource_property_descriptor_is_a_property_descriptor() -> None:
    """ADOResourcePropertyDescriptor is a PropertyDescriptor.

    Equality with a PropertyDescriptor uses the identifier. Equality with
    another ADOResourcePropertyDescriptor also requires the same kind.
    """
    from ado.schema.property import PropertyDescriptor

    descriptor = ADOResourcePropertyDescriptor(
        identifier="baseline", kind=CoreResourceKinds.DATACONTAINER
    )
    same = ADOResourcePropertyDescriptor(
        identifier="baseline", kind=CoreResourceKinds.DATACONTAINER
    )
    different_kind = ADOResourcePropertyDescriptor(
        identifier="baseline", kind=CoreResourceKinds.OPERATION
    )
    different_identifier = ADOResourcePropertyDescriptor(
        identifier="candidate", kind=CoreResourceKinds.DATACONTAINER
    )

    assert isinstance(descriptor, PropertyDescriptor)
    assert descriptor == same
    assert hash(descriptor) == hash(same)
    assert descriptor != different_kind
    assert descriptor != different_identifier
    assert {descriptor, same, different_kind} == {descriptor, different_kind}
    assert descriptor == PropertyDescriptor(identifier="baseline")
    assert descriptor != PropertyDescriptor(identifier="candidate")
