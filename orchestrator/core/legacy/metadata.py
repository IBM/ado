# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Metadata models for legacy validators"""

from collections.abc import Callable
from typing import Annotated

import pydantic

from orchestrator.core.resources import CoreResourceKinds


class LegacyValidatorMetadata(pydantic.BaseModel):
    """Metadata for a legacy validator function"""

    identifier: Annotated[
        str,
        pydantic.Field(
            description="Unique identifier for this validator (e.g., 'csv_constitutive_columns_migration')"
        ),
    ]

    resource_type: Annotated[
        CoreResourceKinds,
        pydantic.Field(description="Resource type this validator applies to"),
    ]

    deprecated_fields: Annotated[
        list[str],
        pydantic.Field(description="Fields that this validator handles"),
    ]

    deprecated_from_version: Annotated[
        str,
        pydantic.Field(description="ADO version when these fields were deprecated"),
    ]

    removed_from_version: Annotated[
        str,
        pydantic.Field(description="ADO version when automatic upgrade was removed"),
    ]

    description: Annotated[
        str,
        pydantic.Field(
            description="Human-readable description of what this validator does"
        ),
    ]

    validator_function: Annotated[
        Callable[[dict], dict],
        pydantic.Field(
            description="The actual migration function",
            exclude=True,  # Don't serialize the function
        ),
    ]

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)


# Made with Bob
