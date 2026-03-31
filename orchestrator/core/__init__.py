# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from .actuatorconfiguration.resource import ActuatorConfigurationResource
from .datacontainer.resource import DataContainerResource
from .discoveryspace.resource import DiscoverySpaceResource
from .operation.resource import OperationResource
from .resources import ADOResource as ADOResource
from .resources import CoreResourceKinds
from .samplestore.resource import SampleStoreResource

kindmap = {
    CoreResourceKinds.OPERATION.value: OperationResource,
    CoreResourceKinds.DISCOVERYSPACE.value: DiscoverySpaceResource,
    CoreResourceKinds.SAMPLESTORE.value: SampleStoreResource,
    CoreResourceKinds.ACTUATORCONFIGURATION.value: ActuatorConfigurationResource,
    CoreResourceKinds.DATACONTAINER.value: DataContainerResource,
}
