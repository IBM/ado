# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import uuid
from typing import Annotated, Any

import pydantic

from ado.core.actuatorconfiguration.config import ActuatorConfiguration
from ado.core.metadata import PackageProvenance, ProvenanceInfo
from ado.core.resources import ADOResource, CoreResourceKinds
from ado.utilities.pydantic import Defaultable


class ActuatorConfigurationProvenanceInfo(ProvenanceInfo):
    """Plugin provenance for an actuator configuration resource."""

    actuators: Annotated[
        dict[str, PackageProvenance],
        pydantic.Field(
            default_factory=dict,
            description=(
                "Mapping of actuator identifier to the Python distribution that "
                "provided it at the time this configuration was created."
            ),
        ),
    ]


class ActuatorConfigurationResource(ADOResource):
    @staticmethod
    def _identifier_from_data(data: dict[str, Any]) -> str:
        return f"{data['kind'].value}-{data['config'].actuatorIdentifier}-{str(uuid.uuid4())[:8]}"

    version: str = "v1"
    kind: CoreResourceKinds = CoreResourceKinds.ACTUATORCONFIGURATION
    config: ActuatorConfiguration
    identifier: Annotated[
        Defaultable[str],
        pydantic.Field(
            default_factory=_identifier_from_data,
        ),
    ]
    provenance: Annotated[
        ActuatorConfigurationProvenanceInfo,
        pydantic.Field(
            default_factory=ActuatorConfigurationProvenanceInfo,
            description=(
                "ado-core and plugin package provenance frozen at resource creation time."
            ),
        ),
    ]
