# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT


from typing import Annotated, Self

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


class PackageProvenance(pydantic.BaseModel):
    """Records the Python distribution package that provided a plugin at resource creation time.

    Captures the PyPI distribution name and installed version so that the exact
    package used when a resource was created can be identified later for
    replication or debugging.

    Attributes:
        distributionName: The PyPI distribution name (e.g. ``"ado-ray-tune"``).
        distributionVersion: The installed version of the distribution (e.g. ``"1.7.1"``).
    """

    model_config = ConfigDict(frozen=True)

    distributionName: Annotated[
        str,
        pydantic.Field(
            description="PyPI distribution name (e.g. 'ado-ray-tune', 'ado-core')."
        ),
    ]
    distributionVersion: Annotated[
        str,
        pydantic.Field(
            description="Installed version of the distribution (e.g. '1.7.1')."
        ),
    ]


class ProvenanceInfo(pydantic.BaseModel):
    """Base model for plugin provenance stored on ADO resources.

    Subclasses declare named maps of plugin identifiers to the distribution
    that provided each plugin at resource creation time. All declared fields
    must be ``dict[str, PackageProvenance]``.
    """

    model_config = ConfigDict(extra="forbid")

    @pydantic.model_validator(mode="after")
    def validate_provenance_field_values(self) -> Self:
        """Ensure every declared field is a dict of PackageProvenance instances."""
        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{field_name} must be a dict[str, PackageProvenance], "
                    f"got {type(value)}"
                )
            for key, item in value.items():
                if not isinstance(item, PackageProvenance):
                    raise ValueError(
                        f"{field_name}[{key!r}] must be PackageProvenance, "
                        f"got {type(item)}"
                    )
        return self
