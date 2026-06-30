# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import enum
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, field_validator, model_validator


class MissingTargetMode(str, enum.Enum):
    """Policy for entities that do not produce a target variable measurement.

    Attributes:
        RaiseError: Raise ``InsufficientDataError`` immediately (default).
        InjectDefaultValue: Inject ``MissingTargetMeasurements.defaultValue``
            as a synthetic row so the entity still counts towards the sampling
            quota.
        Skip: Permanently drop the entity; it does **not** count towards the
            sampling quota but does count against ``budget``.
    """

    RaiseError = "RaiseError"
    InjectDefaultValue = "InjectDefaultValue"
    Skip = "Skip"


class MissingTargetMeasurements(BaseModel):
    """Configuration for how TRIM handles entities that do not produce a target measurement.

    Three modes are supported (see :class:`MissingTargetMode`):

    - ``RaiseError`` (default): raise ``InsufficientDataError`` immediately.
    - ``InjectDefaultValue``: inject ``defaultValue`` as a synthetic row and count
      the entity towards the sampling quota / budget.
    - ``Skip``: permanently drop the entity; it does **not** count towards the
      sampling quota but does count against ``budget``.

    ``budget`` (default ``None``) caps the total number of missing-target entities
    that are tolerated across the whole sampler run.  When exceeded, an
    ``InsufficientDataError`` is raised regardless of mode.
    """

    mode: Annotated[
        MissingTargetMode,
        Field(
            description=(
                "How to handle entities that do not produce a measurement for the "
                "target variable. 'RaiseError' raises immediately (default). "
                "'InjectDefaultValue' injects ``defaultValue`` as a synthetic row. "
                "'Skip' permanently drops the entity from consideration."
            ),
            default=MissingTargetMode.RaiseError,
        ),
    ] = MissingTargetMode.RaiseError

    budget: Annotated[
        int | None,
        Field(
            description=(
                "Maximum number of entities that may fail to measure the target "
                "variable before an InsufficientDataError is raised. "
                "Must be > 0 when set. Use None for no limit."
            ),
            default=None,
        ),
    ] = None

    defaultValue: Annotated[
        float | None,
        Field(
            description=(
                "The value injected for the target variable when mode is "
                "'InjectDefaultValue' and an entity does not produce a measurement. "
                "Required when mode is 'InjectDefaultValue'; ignored otherwise."
            ),
            default=None,
        ),
    ] = None

    @field_validator("budget", mode="after")
    @classmethod
    def budget_must_be_positive(cls, v: int | None) -> int | None:
        """Validate that budget, when set, is strictly positive.

        Args:
            v: The budget value to validate.

        Returns:
            The validated budget value.

        Raises:
            ValueError: When ``v`` is not None and not greater than 0.
        """
        if v is not None and v <= 0:
            raise ValueError(f"budget must be > 0, got {v}")
        return v

    @model_validator(mode="after")
    def default_value_required_for_inject_mode(self) -> "MissingTargetMeasurements":
        """Validate that defaultValue is provided when mode is InjectDefaultValue.

        Returns:
            The validated model instance.

        Raises:
            ValueError: When mode is ``InjectDefaultValue`` and defaultValue is None.
        """
        if (
            self.mode == MissingTargetMode.InjectDefaultValue
            and self.defaultValue is None
        ):
            raise ValueError(
                "defaultValue must be set when mode is 'InjectDefaultValue'"
            )
        return self


class BaseTrimSamplerParameters(BaseModel):
    """Base parameter class shared by all TRIM sampler parameter models."""

    missing_target_variables: Annotated[
        MissingTargetMeasurements,
        Field(
            description=(
                "Configures how both the no-priors sampler and the TRIM iterative "
                "sampler handle entities that do not produce a measurement for the "
                "target variable."
            ),
            default_factory=MissingTargetMeasurements,
        ),
    ]


class NoPriorsParameters(BaseTrimSamplerParameters):
    """
    Parameters for sampling high-dimensional spaces without prior model structure.

    The `sampling_strategy` must be one of the Literals supported.
    Source of truth for supported strategies is the comment block right here:

        strategy (str): sampling subroutine:
        - 'random': selects random points from the beginning
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
        Literal["random", "clhs", "sobol"],
        BeforeValidator(lambda s: s.lower()),
        Field(
            description=(
                "Sampling subroutine. Supported values:\n"
                " - 'random': selects random points from the beginning\n"
                " - 'clhs': dimension-wise random without replacement until each dim cycles\n"
                " - 'sobol': sobol sampling via scipy\n"
                "Validation is case-insensitive; value is normalized to lowercase."
            ),
        ),
    ] = "clhs"
