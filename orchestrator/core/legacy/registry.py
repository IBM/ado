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
    def find_validators_for_field_paths(
        cls, resource_type: CoreResourceKinds, field_paths: set[str]
    ) -> list[LegacyValidatorMetadata]:
        """Find validators that handle specific field paths

        Matches validators based on their declared field_paths, providing
        more precise matching than deprecated_fields (leaf names).

        Args:
            resource_type: The resource type to filter by
            field_paths: Set of full dotted paths (e.g., 'config.properties')

        Returns:
            List of validator metadata that handle any of the specified paths
        """
        return [
            v
            for v in cls.get_validators_for_resource(resource_type)
            if any(path in v.field_paths for path in field_paths)
        ]

    @classmethod
    def list_all(cls) -> list[LegacyValidatorMetadata]:
        """List all registered validators

        Returns:
            List of all registered validator metadata
        """
        return list(cls._validators.values())

    @classmethod
    def resolve_dependencies(
        cls, validator_ids: list[str]
    ) -> tuple[list[str], list[str]]:
        """Resolve validator dependencies and return ordered list

        Uses topological sort to order validators based on their dependencies.
        Detects circular dependencies. Automatically includes all transitive
        dependencies.

        Args:
            validator_ids: List of validator identifiers to order

        Returns:
            Tuple of (ordered_validator_ids, missing_dependencies)
            - ordered_validator_ids: Validators in dependency order (includes all dependencies)
            - missing_dependencies: List of dependency IDs that don't exist

        Raises:
            ValueError: If circular dependencies are detected
        """
        # Build dependency graph - recursively add all dependencies
        graph: dict[str, list[str]] = {}
        in_degree: dict[str, int] = {}
        missing_deps: set[str] = set()
        to_process = list(validator_ids)
        processed = set()

        while to_process:
            vid = to_process.pop(0)
            if vid in processed:
                continue
            processed.add(vid)

            validator = cls.get_validator(vid)
            if validator is None:
                continue

            # Initialize this validator in the graph
            if vid not in graph:
                graph[vid] = []
                in_degree[vid] = 0

            # Process dependencies
            for dep_id in validator.dependencies:
                if dep_id not in cls._validators:
                    missing_deps.add(dep_id)
                    continue

                # Add dependency to graph if not already there
                if dep_id not in graph:
                    graph[dep_id] = []
                    in_degree[dep_id] = 0
                    # Add to processing queue to handle transitive dependencies
                    to_process.append(dep_id)

                # Add edge from dependency to dependent
                if vid not in graph[dep_id]:
                    graph[dep_id].append(vid)

        # Calculate in-degrees
        for vid in graph:
            validator = cls.get_validator(vid)
            if validator:
                for dep_id in validator.dependencies:
                    if dep_id in graph:
                        in_degree[vid] += 1

        # Topological sort using Kahn's algorithm
        queue = [vid for vid in graph if in_degree[vid] == 0]
        ordered = []

        while queue:
            # Sort queue for deterministic ordering
            queue.sort()
            current = queue.pop(0)
            ordered.append(current)

            # Reduce in-degree for dependents
            for dependent in graph[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # Check for circular dependencies
        if len(ordered) != len(graph):
            remaining = [vid for vid in graph if vid not in ordered]
            raise ValueError(
                f"Circular dependency detected among validators: {', '.join(remaining)}"
            )

        return ordered, list(missing_deps)


def legacy_validator(
    identifier: str,
    resource_type: CoreResourceKinds,
    deprecated_fields: list[str],
    deprecated_from_version: str,
    removed_from_version: str,
    description: str,
    field_paths: list[str] | None = None,
    dependencies: list[str] | None = None,
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
        dependencies: Optional list of validator identifiers that must run before this one

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
            description="Migrates CSV sample stores from v1 to v2 format",
            dependencies=["samplestore_kind_entitysource_to_samplestore"]
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
            dependencies=dependencies or [],
        )
        LegacyValidatorRegistry.register(metadata)

        @wraps(func)
        def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Made with Bob
