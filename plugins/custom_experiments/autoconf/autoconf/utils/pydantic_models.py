# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field

from autoconf.utils.config_mapper import map_valid_model_name


class JobConfig(BaseModel):
    model_name: Annotated[str, BeforeValidator(map_valid_model_name)]
    method: str
    gpu_model: str
    tokens_per_sample: int = Field(..., ge=1, description="Max sequence length")
    batch_size: int = Field(..., ge=1)
    is_valid: int | None = Field(
        default=None,
        description="Ground truth. 1 if job was successful. It is not used for prediction purposes",
    )
    number_gpus: int | None = Field(
        default=None, ge=1, description="Number of GPUs used"
    )
