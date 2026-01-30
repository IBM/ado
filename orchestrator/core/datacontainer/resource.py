# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typing
import uuid
from typing import Annotated

import pydantic

import orchestrator.utilities.location
from orchestrator.core.metadata import ConfigurationMetadata
from orchestrator.core.resources import ADOResource, CoreResourceKinds
from orchestrator.utilities.pydantic import Defaultable

if typing.TYPE_CHECKING:  # pragma: nocover
    import pandas as pd
    from rich.console import RenderableType


class TabularData(pydantic.BaseModel):

    data: Annotated[
        dict, pydantic.Field(description="A dictionary representation of tabular data")
    ]

    @classmethod
    def from_dataframe(cls, dataframe: "pd.DataFrame") -> "TabularData":

        return cls(data=dataframe.to_dict(orient="list"))

    @pydantic.field_validator("data")
    def validate_data(cls, data: dict) -> dict:

        import pandas as pd

        # Ensure data is a valid DataFrame
        pd.DataFrame(data)
        return data

    def dataframe(self) -> "pd.DataFrame":

        import pandas as pd

        return pd.DataFrame(self.data)

    def __rich__(self) -> "RenderableType":
        """Render this tabular data using rich."""
        from orchestrator.utilities.rich import dataframe_to_rich_table

        return dataframe_to_rich_table(self.dataframe())


class DataContainer(pydantic.BaseModel):

    tabularData: Annotated[
        dict[str, TabularData] | None,
        pydantic.Field(
            description="Contains a dictionary whose values are TabularData objects representing dataframes"
        ),
    ] = None
    locationData: Annotated[
        dict[
            str,
            orchestrator.utilities.location.SQLStoreConfiguration
            | orchestrator.utilities.location.StorageDatabaseConfiguration
            | orchestrator.utilities.location.FilePathLocation
            | orchestrator.utilities.location.ResourceLocation,
        ]
        | None,
        pydantic.Field(
            description="A dictionary whose values are references to data i.e. data locations"
        ),
    ] = None
    data: Annotated[
        dict[str, dict | list | typing.AnyStr] | None,
        pydantic.Field(
            description="A dictionary of other pydantic objects e.g. lists, dicts, strings,"
        ),
    ] = None
    metadata: Annotated[
        ConfigurationMetadata,
        pydantic.Field(
            description="Metadata about the configuration including optional name, description, "
            "labels for filtering, and any additional custom fields"
        ),
    ] = ConfigurationMetadata()

    @pydantic.model_validator(mode="after")
    def test_data_present(self) -> "DataContainer":

        if not (self.tabularData or self.locationData or self.data):
            raise ValueError(
                "All data fields of the DataContainer (tabularData, locationData, data) were empty."
            )

        return self

    def __rich__(self) -> "RenderableType":
        """Render this data container using rich."""
        import rich.box
        from rich.console import Group
        from rich.panel import Panel
        from rich.text import Text

        from orchestrator.utilities.rich import get_rich_repr

        content = []

        if self.data:
            basic_data_items = [
                Panel(
                    Group(
                        *[
                            Text("Label:", style="bold", end=" "),
                            get_rich_repr(k),
                            get_rich_repr(v),
                        ]
                    ),
                    box=rich.box.SIMPLE,
                )
                for k, v in self.data.items()
            ]

            content.append(
                Panel(
                    Group(*basic_data_items),
                    title="Basic Data",
                    box=rich.box.HORIZONTALS,
                )
            )

        if self.tabularData:

            tabular_items = [
                Panel(
                    Group(
                        *[
                            Text("Label:", style="bold", end=" "),
                            get_rich_repr(k),
                            Text(),
                            get_rich_repr(v),
                        ]
                    ),
                    box=rich.box.SIMPLE,
                )
                for k, v in self.tabularData.items()
            ]

            content.append(
                Panel(
                    Group(*tabular_items),
                    title="Tabular Data",
                    box=rich.box.HORIZONTALS,
                )
            )

        if self.locationData:
            location_items = [
                Panel(
                    Group(
                        *[
                            Text("Label:", style="bold", end=" "),
                            get_rich_repr(k),
                            get_rich_repr(v),
                        ]
                    ),
                    box=rich.box.SIMPLE,
                )
                for k, v in self.locationData.items()
            ]

            content.append(
                Panel(
                    Group(*location_items),
                    title="Location Data",
                    box=rich.box.HORIZONTALS,
                )
            )

        return Group(*content)


class DataContainerResource(ADOResource):
    """A resource which contains non-entity data or references to it

    Note: Contained data must be a supported pydantic type.
    This model does not allow storage of arbitrary types"""

    @staticmethod
    def _identifier_from_data(data: dict[str, typing.Any]) -> str:
        return f"{data['kind'].value}-{str(uuid.uuid4())[:8]}"

    version: Annotated[str, pydantic.Field()] = "v1"
    kind: Annotated[CoreResourceKinds, pydantic.Field()] = (
        CoreResourceKinds.DATACONTAINER
    )
    config: Annotated[DataContainer, pydantic.Field(description="A collection of data")]
    identifier: Annotated[
        Defaultable[str],
        pydantic.Field(
            default_factory=_identifier_from_data,
        ),
    ]

    def __rich__(self) -> "RenderableType":
        """Render this data container resource using rich."""
        from rich.console import Group
        from rich.padding import Padding
        from rich.text import Text

        return Group(
            Text.assemble(("Identifier: ", "bold"), (self.identifier, "bold green")),
            Padding(self.config, (1, 0, 0, 0)),
        )
