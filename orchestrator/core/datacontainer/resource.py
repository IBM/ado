# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typing
import uuid
from typing import Annotated

import pydantic

import orchestrator.utilities.location
from orchestrator.core.metadata import ConfigurationMetadata
from orchestrator.core.resources import ADOResource, CoreResourceKinds

if typing.TYPE_CHECKING:  # pragma: nocover
    import pandas as pd
    from IPython.lib.pretty import PrettyPrinter


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

    def _repr_pretty_(self, p: "PrettyPrinter", cycle: bool = False) -> None:

        if cycle:  # pragma: nocover
            p.text("Cycle detected")
        else:
            p.breakable()
            p.pretty(self.dataframe())
            p.breakable()


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

    def _repr_pretty_(self, p: "PrettyPrinter", cycle: bool = False) -> None:

        if cycle:  # pragma: nocover
            p.text("Cycle detected")
        else:
            if self.data:
                with p.group(2, "Basic Data:"):
                    for k in self.data:
                        p.breakable()
                        p.breakable()
                        p.text(f"Label: {k}")
                        p.breakable()
                        p.breakable()
                        p.pretty(self.data[k])
                        p.break_()

            if self.tabularData:
                p.breakable()
                with p.group(2, "Tabular Data:"):
                    for k in self.tabularData:
                        p.breakable()
                        p.breakable()
                        p.text(f"Label: {k}")
                        p.breakable()
                        p.pretty(self.tabularData[k])
                        p.break_()

            if self.locationData:
                p.breakable()
                with p.group(2, "Location Data:"):
                    for k in self.locationData:
                        p.breakable()
                        p.breakable()
                        p.text(f"Label: {k}")
                        p.breakable()
                        p.breakable()
                        p.pretty(self.locationData[k])
                        p.break_()


class DataContainerResource(ADOResource):
    """A resource which contains non-entity data or references to it

    Note: Contained data must be a supported pydantic type.
    This model does not allow storage of arbitrary types"""

    version: Annotated[str, pydantic.Field()] = "v1"
    kind: Annotated[CoreResourceKinds, pydantic.Field()] = (
        CoreResourceKinds.DATACONTAINER
    )
    config: Annotated[DataContainer, pydantic.Field(description="A collection of data")]

    @pydantic.model_validator(mode="after")
    def generate_identifier_if_not_provided(self) -> "DataContainerResource":

        if self.identifier is None:
            self.identifier = f"{self.kind.value}-{str(uuid.uuid4())[:8]}"

        return self

    def _repr_pretty_(self, p: "PrettyPrinter", cycle: bool = False) -> None:

        if cycle:  # pragma: nocover
            p.text("Cycle detected")
        else:

            p.text(f"Identifier: {self.identifier}")
            p.breakable()
            p.pretty(self.config)
