# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pydantic
from pydantic import BaseModel


class NoPriorsParameters(BaseModel):

    # TODO: get_source_target df requires this atm but you can just use the source df getter and target df getter independently
    targetOutput: str = pydantic.Field(
        default="",
        description="The measured property you will treat as a target variable",
    )

    samples: int = pydantic.Field(default=18, description="Points to sample")

    batchSize: int = pydantic.Field(
        default=5,
        description="Batch size parameter of randomWalk, default is setting this equal to iterationSize",
    )


if __name__ == "__main__":
    params = NoPriorsParameters.model_validate(NoPriorsParameters())
    print(
        f"type of model_validate output on no-priors-characterization default is {type(params)}, printing the full object gives {params}"
    )
