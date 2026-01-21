# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import typing
import uuid

import pydantic

from orchestrator.core.discoveryspace.config import DiscoverySpaceConfiguration
from orchestrator.core.resources import ADOResource, CoreResourceKinds
from orchestrator.schema.measurementspace import MeasurementSpaceConfiguration
from orchestrator.utilities.pydantic import Defaultable

if typing.TYPE_CHECKING:
    from IPython.lib.pretty import PrettyPrinter


class DiscoverySpaceResource(ADOResource):

    version: str = "v2"
    kind: CoreResourceKinds = CoreResourceKinds.DISCOVERYSPACE
    config: DiscoverySpaceConfiguration

    identifier: typing.Annotated[
        Defaultable[str],
        pydantic.Field(
            default_factory=lambda: f"space-{str(uuid.uuid4())[:8]}",
        ),
    ]

    def _repr_pretty_(self, p: "PrettyPrinter", cycle: bool = False) -> None:

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
