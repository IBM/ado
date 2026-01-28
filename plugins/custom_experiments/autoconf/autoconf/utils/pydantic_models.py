# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from typing import Annotated

import pydantic
from pydantic import BeforeValidator

from autoconf.utils.config_mapper import map_valid_model_name


class JobConfig(pydantic.BaseModel):
    model_name: Annotated[str, BeforeValidator(map_valid_model_name)]
    method: Annotated[str, pydantic.Field()]
    gpu_model: Annotated[str, pydantic.Field()]
    tokens_per_sample: Annotated[
        int, pydantic.Field(ge=1, description="Max sequence length")
    ]
    batch_size: Annotated[int, pydantic.Field(ge=1)]
    is_valid: Annotated[
        int | None,
        pydantic.Field(
            description="Ground truth. 1 if job was successful. It is not used for prediction purposes",
        ),
    ] = None
    number_gpus: Annotated[
        int | None, pydantic.Field(ge=1, description="Number of GPUs used")
    ] = None
