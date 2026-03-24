# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Registry for legacy validators that have been removed from active code"""

from collections.abc import Callable
from functools import wraps
from typing import ClassVar

from orchestrator.core.legacy.metadata import LegacyValidatorMetadata
from orchestrator.core.resources import CoreResourceKinds


class LegacyValidatorRegistry:
    """Registry for legacy validators that have been removed from active code"""

    _validators: ClassVar[dict[str, LegacyValidatorMetadata]] = {}

    @classmethod
    def register(cls, metadata: LegacyValidatorMetadata) -> None:
        """Register a legacy validator

        Args:
            metadata: The validator metadata to register
        """
        cls._validators[metadata.identifier] = metadata

    @classmethod
    def get_validator(cls, identifier: str) -> LegacyValidatorMetadata | None:
        """Get a specific validator by identifier

        Args:
            identifier: The unique identifier of the validator

        Returns:
            The validator metadata if found, None otherwise
        """
        return cls._validators.get(identifier)

    @classmethod
    def get_validators_for_resource(
        cls, resource_type: CoreResourceKinds
    ) -> list[LegacyValidatorMetadata]:
        """Get all validators for a specific resource type

        Args:
            resource_type: The resource type to filter by

        Returns:
            List of validator metadata for the specified resource type
        """
        return [v for v in cls._validators.values() if v.resource_type == resource_type]

    @classmethod
    def find_validators_for_fields(
        cls, resource_type: CoreResourceKinds, field_names: set[str]
    ) -> list[LegacyValidatorMetadata]:
        """Find validators that handle specific deprecated fields

        Args:
            resource_type: The resource type to filter by
            field_names: Set of field names to search for

        Returns:
            List of validator metadata that handle any of the specified fields
        """
        return [
            v
            for v in cls.get_validators_for_resource(resource_type)
            if any(field in v.deprecated_fields for field in field_names)
        ]

    @classmethod
    def list_all(cls) -> list[LegacyValidatorMetadata]:
        """List all registered validators

        Returns:
            List of all registered validator metadata
        """
        return list(cls._validators.values())


def legacy_validator(
    identifier: str,
    resource_type: CoreResourceKinds,
    deprecated_fields: list[str],
    deprecated_from_version: str,
    removed_from_version: str,
    description: str,
    field_paths: list[str] | None = None,
) -> Callable[[Callable[[dict], dict]], Callable[[dict], dict]]:
    """Decorator to register a legacy validator function

    Args:
        identifier: Unique identifier for this validator
        resource_type: Resource type this validator applies to
        deprecated_fields: Fields that this validator handles
        deprecated_from_version: ADO version when these fields were deprecated
        removed_from_version: ADO version when automatic upgrade was removed
        description: Human-readable description of what this validator does
        field_paths: Optional explicit paths to fields (e.g., 'config.properties')

    Returns:
        Decorator function that registers the validator

    Example:
        @legacy_validator(
            identifier="csv_constitutive_columns_migration",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_fields=["constitutivePropertyColumns", "propertyMap"],
            field_paths=["config.specification.constitutivePropertyColumns"],
            deprecated_from_version="1.3.5",
            removed_from_version="1.6.0",
            description="Migrates CSV sample stores from v1 to v2 format"
        )
        def migrate_csv_v1_to_v2(data: dict) -> dict:
            # Migration logic here
            return data
    """

    def decorator(func: Callable[[dict], dict]) -> Callable[[dict], dict]:
        metadata = LegacyValidatorMetadata(
            identifier=identifier,
            resource_type=resource_type,
            deprecated_fields=deprecated_fields,
            deprecated_from_version=deprecated_from_version,
            removed_from_version=removed_from_version,
            description=description,
            validator_function=func,
            field_paths=field_paths or [],
        )
        LegacyValidatorRegistry.register(metadata)

        @wraps(func)
        def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Made with Bob
