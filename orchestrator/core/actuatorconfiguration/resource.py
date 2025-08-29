# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import uuid

import pydantic

from orchestrator.core.actuatorconfiguration.config import ActuatorConfiguration
from orchestrator.core.resources import ADOResource, CoreResourceKinds


class ActuatorConfigurationResource(ADOResource):

    version: str = "v1"
    kind: CoreResourceKinds = CoreResourceKinds.ACTUATORCONFIGURATION
    config: ActuatorConfiguration

    @pydantic.model_validator(mode="after")
    def generate_identifier_if_not_provided(self):

        if self.identifier is None:
            self.identifier = f"{self.kind.value}-{self.config.actuatorIdentifier}-{str(uuid.uuid4())[:8]}"

        return self
