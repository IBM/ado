# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from .actuatorconfiguration.resource import ActuatorConfigurationResource
from .datacontainer.resource import DataContainerResource
from .discoveryspace.resource import DiscoverySpaceResource
from .operation.resource import OperationResource
from .resources import ADOResource, CoreResourceKinds
from .samplestore.resource import SampleStoreResource

kindmap = {
    CoreResourceKinds.RESOURCE.value: ADOResource,
    CoreResourceKinds.OPERATION.value: OperationResource,
    CoreResourceKinds.DISCOVERYSPACE.value: DiscoverySpaceResource,
    CoreResourceKinds.SAMPLESTORE.value: SampleStoreResource,
    CoreResourceKinds.ACTUATORCONFIGURATION.value: ActuatorConfigurationResource,
    CoreResourceKinds.DATACONTAINER.value: DataContainerResource,
}
