# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import pydantic

from orchestrator.core.resources import ADOResource, CoreResourceKinds
from orchestrator.core.samplestore.config import SampleStoreConfiguration


class SampleStoreResource(ADOResource):

    version: str = "v2"
    kind: CoreResourceKinds = CoreResourceKinds.SAMPLESTORE
    config: SampleStoreConfiguration

    @pydantic.field_validator("kind", mode="before")
    @classmethod
    def convert_entity_source_kind_to_sample_store(cls, value: str):
        from orchestrator.core.resources import (
            CoreResourceKinds,
            warn_deprecated_resource_model_in_use,
        )

        old_value = "entitysource"
        new_value = "samplestore"
        if value == old_value:
            warn_deprecated_resource_model_in_use(
                affected_resource=CoreResourceKinds.SAMPLESTORE,
                deprecated_from_ado_version="v0.9.6",
                removed_from_ado_version="v1.0.0",
            )
            value = new_value

        return value
