# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field


class NoPriorsParameters(BaseModel):
    """
    Parameters for sampling high-dimensional spaces without prior model structure.

    The `sampling_strategy` must be one of the Literals supported.
    Source of truth for supported strategies is the comment block right here:

        strategy (str): sampling subroutine:
        - 'random': selects random points from the beginning
        - 'one_shift': refer to one_shift_then_random_points_high_dimensional_sampling
        - 'recursive_aggregation': refer to recursive_aggregation_high_dimensional_sampling
        - 'clhs': refer to concatenated_latin_hypercube_sampling
        - 'sobol': sobol sampling
    """

    targetOutput: Annotated[
        str,
        Field(
            description="The measured property you will treat as a target variable.",
        ),
    ]

    samples: Annotated[
        int,
        Field(
            ge=1,
            description="Number of unique points to sample (must be >= 1).",
        ),
    ] = 20

    batchSize: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Batch size parameter used by certain samplers (e.g., randomWalk) via continuous batching; "
                "by default set equal to iterationSize in those contexts. Must be >= 1."
            ),
        ),
    ] = 1

    sampling_strategy: Annotated[
        Literal["random", "one_shift", "recursive_aggregation", "clhs", "sobol"],
        BeforeValidator(lambda s: s.lower()),
        Field(
            description=(
                "Sampling subroutine. Supported values:\n"
                " - 'random': selects random points from the beginning\n"
                " - 'one_shift': see one_shift_then_random_points_high_dimensional_sampling\n"
                " - 'recursive_aggregation': see recursive_aggregation_high_dimensional_sampling\n"
                " - 'clhs': dimension-wise random without replacement until each dim cycles\n"
                " - 'sobol': sobol sampling via scipy\n"
                "Aliases: 'random_shifts' → 'recursive_aggregation'.\n"
                "Validation is case-insensitive; value is normalized to lowercase."
            ),
        ),
    ] = "clhs"


if __name__ == "__main__":
    params = NoPriorsParameters.model_validate(NoPriorsParameters(targetOutput="test"))
    print(
        f"type of model_validate output on no-priors-characterization default is {type(params)}, printing the full object gives {params}"
    )
