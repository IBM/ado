# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


from typing import Annotated

import pydantic
from pydantic import ConfigDict


class ConfigurationMetadata(pydantic.BaseModel):

    model_config = ConfigDict(extra="allow")

    name: Annotated[
        str | None,
        pydantic.Field(
            description="A descriptive name for this configuration. Does not have to be unique"
        ),
    ] = None
    description: Annotated[
        str | None,
        pydantic.Field(
            description="One or more sentences describing this configuration. "
        ),
    ] = None
    labels: Annotated[
        dict[str, str] | None,
        pydantic.Field(
            description="Optional labels to allow for quick filtering of this resource"
        ),
    ] = None
