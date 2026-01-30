# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import typing
import uuid

import pydantic
import rich.box

from orchestrator.core.discoveryspace.config import DiscoverySpaceConfiguration
from orchestrator.core.resources import ADOResource, CoreResourceKinds
from orchestrator.schema.measurementspace import MeasurementSpaceConfiguration
from orchestrator.utilities.pydantic import Defaultable

if typing.TYPE_CHECKING:
    from rich.console import RenderableType


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

    def __rich__(self) -> "RenderableType":
        """Render this discovery space resource using rich."""
        from rich.console import Group
        from rich.panel import Panel
        from rich.text import Text

        from orchestrator.schema.entityspace import EntitySpaceRepresentation
        from orchestrator.schema.measurementspace import MeasurementSpace
        from orchestrator.utilities.rich import get_rich_repr

        content = [
            Text("Identifier:", style="bold", end=" "),
            get_rich_repr(self.identifier),
            Text(),
        ]

        # Entity Space section
        entity_space = EntitySpaceRepresentation.representationFromConfiguration(
            conf=self.config.entitySpace
        )
        if entity_space is not None:
            content.extend(
                [
                    Text("Entity Space:", style="bold"),
                    Panel(
                        entity_space,
                        box=rich.box.SIMPLE_HEAD,
                        padding=(0, 2),
                    ),  # Uses entity_space.__rich__()
                ]
            )

        # Measurement Space section
        if isinstance(
            self.config.experiments,
            MeasurementSpaceConfiguration,
        ):
            measurement_space = MeasurementSpace(configuration=self.config.experiments)
        else:
            measurement_space = MeasurementSpace.measurementSpaceFromSelection(
                selectedExperiments=self.config.experiments
            )

        content.extend(
            [
                Text("Measurement Space:", style="bold"),
                Panel(
                    measurement_space,
                    box=rich.box.SIMPLE_HEAD,
                    padding=(0, 2),
                ),  # Uses measurement_space.__rich__()
            ]
        )

        content.append(
            Text.assemble(
                ("Sample Store identifier: ", "bold"),
                (self.config.sampleStoreIdentifier, "cyan"),
            )
        )

        return Group(*content)
