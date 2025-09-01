# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from orchestrator.core.resources import ADOResource, CoreResourceKinds
from orchestrator.core.samplestore.config import SampleStoreConfiguration


class SampleStoreResource(ADOResource):

    version: str = "v2"
    kind: CoreResourceKinds = CoreResourceKinds.SAMPLESTORE
    config: SampleStoreConfiguration
