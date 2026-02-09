# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from typing import ClassVar

import pydantic
from pydantic import BaseModel, Field, field_validator


class NoPriorsParameters(BaseModel):
    """
    Parameters for sampling high-dimensional spaces without prior model structure.

    The `sampling_strategy` is validated against the supported set documented below.
    Source of truth for supported strategies is the comment block right here:

        strategy (str): sampling subroutine:
        - 'random': selects random points from the beginning
        - 'one_shift': refer to one_shift_then_random_points_high_dimensional_sampling
        - 'recursive_aggregation': refer to recursive_aggregation_high_dimensional_sampling
        - 'clhs': refer to concatenated_latin_hypercube_sampling
        - 'sobol': sobol sampling
    """

    # --- Supported strategies (centralized) ---
    SUPPORTED_STRATEGIES: ClassVar[set[str]] = {
        "random",
        "one_shift",
        "recursive_aggregation",
        "clhs",
        "sobol",
    }

    targetOutput: str = Field(
        default="",
        description="The measured property you will treat as a target variable.",
    )

    samples: int = Field(
        default=20,
        ge=1,
        description="Number of unique points to sample (must be >= 1).",
    )

    batchSize: int = Field(
        default=1,
        ge=1,
        description=(
            "Batch size parameter used by certain samplers (e.g., randomWalk) via continuous batching; "
            "by default set equal to iterationSize in those contexts. Must be >= 1."
        ),
    )

    sampling_strategy: str = Field(
        default="clhs",
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
    )

    @field_validator("sampling_strategy")
    @classmethod
    def _validate_strategy(cls, v: str) -> str:
        """
        Validate and normalize the sampling strategy:
        - strip/normalize to lowercase
        - ensure result is in SUPPORTED_STRATEGIES
        """
        if not isinstance(v, str):
            raise TypeError("sampling_strategy must be a string.")

        normalized = v.strip().lower()

        if normalized not in cls.SUPPORTED_STRATEGIES:
            raise pydantic.ValidationError(
                [
                    pydantic.errors.PydanticCustomError(
                        "unsupported_strategy",
                        (f"Unsupported sampling_strategy '{v}'. "),
                    )
                ],
                cls,
            )
        return normalized


if __name__ == "__main__":
    params = NoPriorsParameters.model_validate(NoPriorsParameters())
    print(
        f"type of model_validate output on no-priors-characterization default is {type(params)}, printing the full object gives {params}"
    )
