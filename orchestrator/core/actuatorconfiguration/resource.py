# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import uuid
from typing import Annotated, Any

import pydantic

from orchestrator.core.actuatorconfiguration.config import ActuatorConfiguration
from orchestrator.core.metadata import PackageProvenance
from orchestrator.core.resources import ADOResource, CoreResourceKinds
from orchestrator.utilities.pydantic import Defaultable


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
    actuatorProvenance: Annotated[
        PackageProvenance | None,
        pydantic.Field(
            default=None,
            description=(
                "Python distribution that provided the actuator at the time this "
                "configuration was created. None for resources created before "
                "provenance tracking was introduced, or when the distribution could "
                "not be resolved."
            ),
        ),
    ]
