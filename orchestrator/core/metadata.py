# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


import pydantic
from pydantic import ConfigDict


class ConfigurationMetadata(pydantic.BaseModel):

    model_config = ConfigDict(extra="allow")

    name: str | None = pydantic.Field(
        default=None,
        description="A descriptive name for this configuration. Does not have to be unique",
    )
    description: str | None = pydantic.Field(
        default=None,
        description="One or more sentences describing this configuration. ",
    )
    labels: dict[str, str] | None = pydantic.Field(
        default=None,
        description="Optional labels to allow for quick filtering of this resource",
    )
