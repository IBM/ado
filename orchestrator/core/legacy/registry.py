# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Registry for legacy migrators that have been removed from active code"""

from collections.abc import Callable
from functools import wraps
from typing import ClassVar

from orchestrator.core.legacy.metadata import LegacyMigratorMetadata
from orchestrator.core.resources import CoreResourceKinds


class LegacyMigratorRegistry:
    """Registry for legacy migrators that have been removed from active code"""

    _migrators: ClassVar[dict[str, LegacyMigratorMetadata]] = {}

    @classmethod
    def register(cls, metadata: LegacyMigratorMetadata) -> None:
        """Register a legacy migrator

        Args:
            metadata: The migrator metadata to register
        """
        cls._migrators[metadata.identifier] = metadata

    @classmethod
    def get_migrator(cls, identifier: str) -> LegacyMigratorMetadata | None:
        """Get a specific migrator by identifier

        Args:
            identifier: The unique identifier of the migrator

        Returns:
            The migrator metadata if found, None otherwise
        """
        return cls._migrators.get(identifier)

    @classmethod
    def get_migrators_for_resource(
        cls, resource_type: CoreResourceKinds
    ) -> list[LegacyMigratorMetadata]:
        """Get all migrators for a specific resource type

        Args:
            resource_type: The resource type to filter by

        Returns:
            List of migrator metadata for the specified resource type
        """
        return [v for v in cls._migrators.values() if v.resource_type == resource_type]

    @classmethod
    def find_migrators_for_deprecated_field_paths(
        cls,
        resource_type: CoreResourceKinds,
        deprecated_field_paths: set[str],
    ) -> list[LegacyMigratorMetadata]:
        """Find migrators that handle specific field paths

        Matches migrators based on their declared field_paths, providing
        more precise matching than deprecated_fields (leaf names).

        Args:
            resource_type: The resource type to filter by
            deprecated_field_paths: Set of full dotted paths (e.g., 'config.properties')

        Returns:
            List of migrator metadata that handle any of the specified paths
        """
        return [
            v
            for v in cls.get_migrators_for_resource(resource_type)
            if any(path in v.deprecated_field_paths for path in deprecated_field_paths)
        ]

    @classmethod
    def list_all(cls) -> list[LegacyMigratorMetadata]:
        """List all registered migrators

        Returns:
            List of all registered migrator metadata
        """
        return list(cls._migrators.values())

    @classmethod
    def resolve_dependencies(
        cls, migrator_ids: list[str]
    ) -> tuple[list[str], list[str]]:
        """Resolve migrator dependencies and return ordered list

        Uses topological sort to order migrators based on their dependencies.
        Detects circular dependencies. Automatically includes all transitive
        dependencies.

        Args:
            migrator_ids: List of migrator identifiers to order

        Returns:
            Tuple of (ordered_migrator_ids, missing_dependencies)
            - ordered_migrator_ids: Migrators in dependency order (includes all dependencies)
            - missing_dependencies: List of dependency IDs that don't exist

        Raises:
            ValueError: If circular dependencies are detected
        """
        # Build dependency graph - recursively add all dependencies
        graph: dict[str, list[str]] = {}
        in_degree: dict[str, int] = {}
        missing_deps: set[str] = set()
        to_process = list(migrator_ids)
        processed = set()

        while to_process:
            vid = to_process.pop(0)
            if vid in processed:
                continue
            processed.add(vid)

            migrator = cls.get_migrator(vid)
            if migrator is None:
                continue

            # Initialize this migrator in the graph
            if vid not in graph:
                graph[vid] = []
                in_degree[vid] = 0

            # Process dependencies
            for dep_id in migrator.dependencies:
                if dep_id not in cls._migrators:
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
            migrator = cls.get_migrator(vid)
            if migrator:
                for dep_id in migrator.dependencies:
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
                f"Circular dependency detected among migrators: {', '.join(remaining)}"
            )

        return ordered, list(missing_deps)


def legacy_migrator(
    identifier: str,
    resource_type: CoreResourceKinds,
    deprecated_field_paths: list[str],
    deprecated_from_version: str,
    removed_from_version: str,
    description: str,
    dependencies: list[str] | None = None,
) -> Callable[[Callable[[dict], dict]], Callable[[dict], dict]]:
    """Decorator to register a legacy migrator function

    Args:
        identifier: Unique identifier for this migrator
        resource_type: Resource type this migrator applies to
        deprecated_field_paths: Explicit paths to fields (e.g., 'config.properties', 'config.specification.moduleType')
        deprecated_from_version: ADO version when these fields were deprecated
        removed_from_version: ADO version when automatic upgrade was removed
        description: Human-readable description of what this migrator does
        dependencies: Optional list of migrator identifiers that must run before this one

    Returns:
        Decorator function that registers the migrator

    Example:
        @legacy_migrator(
            identifier="csv_constitutive_columns_migration",
            resource_type=CoreResourceKinds.SAMPLESTORE,
            deprecated_field_paths=["config.constitutivePropertyColumns", "config.experiments"],
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
        metadata = LegacyMigratorMetadata(
            identifier=identifier,
            resource_type=resource_type,
            deprecated_from_version=deprecated_from_version,
            removed_from_version=removed_from_version,
            description=description,
            migrator_function=func,
            deprecated_field_paths=deprecated_field_paths,
            dependencies=dependencies or [],
        )
        LegacyMigratorRegistry.register(metadata)

        @wraps(func)
        def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Made with Bob
