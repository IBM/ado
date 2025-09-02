# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typing
import uuid

import pydantic

import orchestrator.utilities.location
from orchestrator.core.metadata import ConfigurationMetadata
from orchestrator.core.resources import ADOResource, CoreResourceKinds

if typing.TYPE_CHECKING:  # pragma: nocover
    import pandas as pd


class TabularData(pydantic.BaseModel):

    data: dict = pydantic.Field(description="A valid string description of a table")

    @classmethod
    def from_dataframe(cls, dataframe: "pd.DataFrame"):

        return cls(data=dataframe.to_dict(orient="list"))

    @pydantic.field_validator("data")
    def validate_data(cls, data):

        import pandas as pd

        # Ensure data is a valid DataFrame
        pd.DataFrame(data)
        return data

    def dataframe(self) -> "pd.DataFrame":

        import pandas as pd

        return pd.DataFrame(self.data)

    def _repr_pretty_(self, p, cycle=False):

        if cycle:  # pragma: nocover
            p.text("Cycle detected")
        else:
            p.breakable()
            p.pretty(self.dataframe())
            p.breakable()


class DataContainer(pydantic.BaseModel):

    tabularData: dict[str, TabularData] | None = pydantic.Field(
        default=None,
        description="Contains a dictionary whose values are string representations of dataframes",
    )
    locationData: (
        dict[
            str,
            orchestrator.utilities.location.SQLStoreConfiguration
            | orchestrator.utilities.location.StorageDatabaseConfiguration
            | orchestrator.utilities.location.FilePathLocation
            | orchestrator.utilities.location.ResourceLocation,
        ]
        | None
    ) = pydantic.Field(
        default=None,
        description="A dictionary whose values are references to data i.e. data locations",
    )
    data: dict[str, dict | list | typing.AnyStr] | None = pydantic.Field(
        default=None,
        description="A dictionary of other pydantic objects e.g. lists, dicts, strings,",
    )
    metadata: ConfigurationMetadata = pydantic.Field(
        default=ConfigurationMetadata(),
        description="User defined metadata about the configuration. A set of keys and values. "
        "Two optional keys that are used by convention are name and description",
    )

    @pydantic.model_validator(mode="after")
    def test_data_present(self):

        assert (
            self.tabularData or self.locationData or self.data
        ), "All data fields empty in DataContainer"

        return self

    def _repr_pretty_(self, p, cycle=False):

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

    version: str = "v1"
    kind: CoreResourceKinds = CoreResourceKinds.DATACONTAINER
    config: DataContainer = pydantic.Field(description="A collection of data")

    @pydantic.model_validator(mode="after")
    def generate_identifier_if_not_provided(self):

        if self.identifier is None:
            self.identifier = f"{self.kind.value}-{str(uuid.uuid4())[:8]}"

        return self

    def _repr_pretty_(self, p, cycle=False):

        if cycle:  # pragma: nocover
            p.text("Cycle detected")
        else:

            p.text(f"Identifier: {self.identifier}")
            p.breakable()
            p.pretty(self.config)
