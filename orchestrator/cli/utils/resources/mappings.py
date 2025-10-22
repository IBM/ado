# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from orchestrator.core import CoreResourceKinds

cli_names_to_resource_kinds: dict[str, CoreResourceKinds] = {
    r.value: r for r in CoreResourceKinds
}
cli_names_to_resource_kinds.update({"space": CoreResourceKinds.DISCOVERYSPACE})

resource_kinds_to_cli: dict[CoreResourceKinds, str] = {
    r: r.value for r in CoreResourceKinds
}
resource_kinds_to_cli.update({CoreResourceKinds.DISCOVERYSPACE: "space"})

resource_kinds_to_human: dict[CoreResourceKinds, str] = {
    CoreResourceKinds.ACTUATORCONFIGURATION: "actuator configuration",
    CoreResourceKinds.DATACONTAINER: "data container",
    CoreResourceKinds.DISCOVERYSPACE: "space",
    CoreResourceKinds.OPERATION: "operation",
    CoreResourceKinds.OPERATOR: "operator",
    CoreResourceKinds.SAMPLESTORE: "sample store",
}
