# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
import typing
import uuid
from typing import Annotated

import pydantic
import rich.box

from orchestrator.core.discoveryspace.config import DiscoverySpaceConfiguration
from orchestrator.core.metadata import PackageProvenance, ProvenanceInfo
from orchestrator.core.resources import ADOResource, CoreResourceKinds
from orchestrator.schema.measurementspace import MeasurementSpaceConfiguration
from orchestrator.utilities.pydantic import Defaultable

if typing.TYPE_CHECKING:
    from rich.console import RenderableType


class DiscoverySpaceProvenanceInfo(ProvenanceInfo):
    """Plugin provenance for a discovery space resource."""

    actuators: Annotated[
        dict[str, PackageProvenance],
        pydantic.Field(
            default_factory=dict,
            description=(
                "Mapping of actuator identifier to the Python distribution that "
                "provided it at the time this space was created."
            ),
        ),
    ]
    customExperiments: Annotated[
        dict[str, PackageProvenance],
        pydantic.Field(
            default_factory=dict,
            description=(
                "Mapping of custom experiment identifier to the Python distribution "
                "that provided it at the time this space was created."
            ),
        ),
    ]


class DiscoverySpaceResource(ADOResource):

    version: str = "v2"
    kind: CoreResourceKinds = CoreResourceKinds.DISCOVERYSPACE
    config: DiscoverySpaceConfiguration

    identifier: Annotated[
        Defaultable[str],
        pydantic.Field(
            default_factory=lambda: f"space-{str(uuid.uuid4())[:8]}",
        ),
    ]
    provenance: Annotated[
        DiscoverySpaceProvenanceInfo,
        pydantic.Field(
            default_factory=DiscoverySpaceProvenanceInfo,
            description="Plugin package provenance frozen at resource creation time.",
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
