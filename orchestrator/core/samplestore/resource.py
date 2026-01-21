# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
from typing import Annotated

import pydantic

from orchestrator.core.resources import ADOResource, CoreResourceKinds
from orchestrator.core.samplestore.config import SampleStoreConfiguration
from orchestrator.utilities.pydantic import Defaultable


class SampleStoreResource(ADOResource):

    @staticmethod
    def _generate_sample_store_identifier() -> str:
        import uuid

        # AP 26/09/2025:
        # This identifier could be a string that gets
        # parsed by --set as an int/float.
        # Examples are:
        # - 344846 -> interpreted as the number
        # - 5013e3 -> interpreted as 5013000.0
        # We check if this would happen and re-generate
        # the identifier if that's the case
        while True:
            identifier = str(uuid.uuid4())[:6]
            try:
                float(identifier)
            except ValueError:
                break

        return identifier

    version: str = "v2"
    kind: CoreResourceKinds = CoreResourceKinds.SAMPLESTORE
    config: SampleStoreConfiguration
    identifier: Annotated[
        Defaultable[str],
        pydantic.Field(
            default_factory=_generate_sample_store_identifier,
        ),
    ]
