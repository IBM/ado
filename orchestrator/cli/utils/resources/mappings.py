# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from orchestrator.core import CoreResourceKinds

# Shorthands for CLI names
cli_shorthands_to_cli_names: dict[str, str] = {
    "ac": CoreResourceKinds.ACTUATORCONFIGURATION.value,
    "ctx": "context",
    "dcr": CoreResourceKinds.DATACONTAINER.value,
    "op": CoreResourceKinds.OPERATION.value,
    "space": CoreResourceKinds.DISCOVERYSPACE.value,
    "store": CoreResourceKinds.SAMPLESTORE.value,
}

resource_kinds_to_human: dict[CoreResourceKinds, str] = {
    CoreResourceKinds.ACTUATORCONFIGURATION: "actuator configuration",
    CoreResourceKinds.DATACONTAINER: "data container",
    CoreResourceKinds.DISCOVERYSPACE: "space",
    CoreResourceKinds.OPERATION: "operation",
    CoreResourceKinds.OPERATOR: "operator",
    CoreResourceKinds.SAMPLESTORE: "sample store",
}
