# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import uuid

import pydantic

from orchestrator.core.discoveryspace.config import DiscoverySpaceConfiguration
from orchestrator.core.resources import ADOResource, CoreResourceKinds
from orchestrator.schema.measurementspace import MeasurementSpaceConfiguration


class DiscoverySpaceResource(ADOResource):

    version: str = "v2"
    kind: CoreResourceKinds = CoreResourceKinds.DISCOVERYSPACE
    config: DiscoverySpaceConfiguration

    @pydantic.model_validator(mode="after")
    def generate_identifier_if_not_provided(self):

        # We can't reliably get the sample store identifier from a SampleStoreConfiguration
        # This is because (A) it may represent an uncreated SampleStore and (B) if created the location
        # of the identifier in the parameters is unknown
        # We would need the SampleStore object or SampleStoreResource
        if self.identifier is None:
            self.identifier = f"space-{str(uuid.uuid4())[:8]}"

        return self

    def _repr_pretty_(self, p, cycle=False):

        if cycle:  # pragma: nocover
            p.text("Cycle detected")
        else:
            from orchestrator.schema.entityspace import EntitySpaceRepresentation
            from orchestrator.schema.measurementspace import (
                MeasurementSpace,
            )

            p.text(f"Identifier: {self.identifier}")
            p.breakable()

            entity_space = EntitySpaceRepresentation.representationFromConfiguration(
                conf=self.config.entitySpace
            )
            if entity_space is not None:
                p.breakable()
                with p.group(2, "Entity Space:"):
                    p.breakable()
                    p.break_()
                    p.pretty(entity_space)
                    p.breakable()

            p.breakable()
            with p.group(2, "Measurement Space:"):
                if isinstance(
                    self.config.experiments,
                    MeasurementSpaceConfiguration,
                ):
                    measurement_space = MeasurementSpace(
                        configuration=self.config.experiments
                    )
                else:
                    measurement_space = MeasurementSpace.measurementSpaceFromSelection(
                        selectedExperiments=self.config.experiments
                    )
                p.breakable()
                p.pretty(measurement_space)
                p.breakable()

            p.breakable()
            with p.group(2, "Sample Store identifier:"):
                p.breakable()
                p.pretty(self.config.sampleStoreIdentifier)
                p.breakable()
