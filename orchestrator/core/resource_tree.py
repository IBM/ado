# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Build resource relationship trees from metastore edges."""

from __future__ import annotations

import dataclasses
import datetime
from typing import TYPE_CHECKING, Annotated

import pydantic

from orchestrator.core.metadata import ConfigurationMetadata
from orchestrator.core.resources import ADOResource, CoreResourceKinds

if TYPE_CHECKING:
    import pandas as pd

    from orchestrator.metastore.base import ResourceStore

INPUT_REFERENCE_SUBJECT_KINDS: frozenset[str] = frozenset(
    {CoreResourceKinds.ACTUATORCONFIGURATION.value}
)


@dataclasses.dataclass(frozen=True)
class ResourceRelationship:
    """A directed edge in the resource relationship graph."""

    subject: str
    object: str
    subject_kind: str
    object_kind: str


class ResourceTreeEdgePolicy:
    """Classifies relationship edges for default vs full-DAG tree modes."""

    def __init__(
        self,
        input_reference_subject_kinds: frozenset[str] | None = None,
    ) -> None:
        self._input_reference_subject_kinds = (
            input_reference_subject_kinds or INPUT_REFERENCE_SUBJECT_KINDS
        )

    def is_input_reference_edge(self, edge: ResourceRelationship) -> bool:
        """Return True when the edge is an input-reference (excluded from default tree)."""
        return edge.subject_kind in self._input_reference_subject_kinds

    def include_edge(
        self, edge: ResourceRelationship, *, all_relationships: bool
    ) -> bool:
        """Return True when the edge should be traversed for the selected mode."""
        if all_relationships:
            return True
        return not self.is_input_reference_edge(edge)


class ResourceTreeNode(pydantic.BaseModel):
    """A node in a rendered resource tree."""

    identifier: str
    kind: str
    name: Annotated[str | None, pydantic.Field(default=None)] = None
    description: Annotated[str | None, pydantic.Field(default=None)] = None
    age: Annotated[str | None, pydantic.Field(default=None)] = None
    labels: Annotated[dict[str, str] | None, pydantic.Field(default=None)] = None
    children: Annotated[list[ResourceTreeNode], pydantic.Field(default_factory=list)]


@dataclasses.dataclass
class ResourceTreeOptions:
    """Options controlling resource tree construction."""

    all_relationships: bool = False
    invert: bool = False
    depth: int | None = None
    dedupe: bool = False
    include_orphans: bool = False
    kind_filter: frozenset[str] | None = None
    matching_identifiers: frozenset[str] | None = None
    scoped_root_identifier: str | None = None
    sort: bool = False


def relationships_from_dataframe(
    edges_df: pd.DataFrame,
) -> list[ResourceRelationship]:
    """Convert a relationships DataFrame to ResourceRelationship instances."""
    if edges_df.empty:
        return []
    return [
        ResourceRelationship(
            subject=row["SUBJECT"],
            object=row["OBJECT"],
            subject_kind=row["SUBJECT_KIND"],
            object_kind=row["OBJECT_KIND"],
        )
        for _, row in edges_df.iterrows()
    ]


def _sort_key(identifier: str, created_at: datetime.datetime | None) -> tuple:
    """Sort siblings by created timestamp then identifier."""
    if created_at is None:
        return (datetime.datetime.max.replace(tzinfo=datetime.timezone.utc), identifier)
    return (created_at, identifier)


def _metadata_from_resource(resource: ADOResource) -> ConfigurationMetadata:
    config = resource.config
    metadata = getattr(config, "metadata", None)
    if metadata is None:
        return ConfigurationMetadata()
    if isinstance(metadata, ConfigurationMetadata):
        return metadata
    return ConfigurationMetadata.model_validate(metadata)


def _timedelta_to_string(seconds: float) -> str:
    from orchestrator.utilities.time import timedelta_to_string

    return timedelta_to_string(seconds)


def _time_since_timestamp(
    timestamp: datetime.datetime,
) -> datetime.timedelta:
    from orchestrator.utilities.time import time_since_timestamp

    return time_since_timestamp(timestamp)


def enrich_tree_nodes(
    roots: list[ResourceTreeNode],
    resources: dict[str, ADOResource],
    *,
    show_names: bool = False,
    show_age: bool = False,
    show_metadata: bool = False,
) -> list[ResourceTreeNode]:
    """Attach metadata from loaded resources to tree nodes."""

    def _enrich(node: ResourceTreeNode) -> ResourceTreeNode:
        resource = resources.get(node.identifier)
        name: str | None = None
        description: str | None = None
        age: str | None = None
        labels: dict[str, str] | None = None
        if resource is not None:
            resource_metadata = _metadata_from_resource(resource)
            if show_names:
                name = resource_metadata.name
            if show_metadata:
                description = resource_metadata.description
                labels = resource_metadata.labels
            if show_age:
                age = _timedelta_to_string(
                    _time_since_timestamp(resource.created).total_seconds()
                )
        return ResourceTreeNode(
            identifier=node.identifier,
            kind=node.kind,
            name=name,
            description=description,
            age=age,
            labels=labels,
            children=[_enrich(child) for child in node.children],
        )

    return [_enrich(root) for root in roots]


def collect_tree_identifiers(roots: list[ResourceTreeNode]) -> set[str]:
    """Collect all identifiers present in a tree forest."""
    identifiers: set[str] = set()

    def _walk(node: ResourceTreeNode) -> None:
        identifiers.add(node.identifier)
        for child in node.children:
            _walk(child)

    for root in roots:
        _walk(root)
    return identifiers


class ResourceTreeBuilder:
    """Builds resource trees from relationship edges and store metadata."""

    def __init__(
        self,
        store: ResourceStore,
        edge_policy: ResourceTreeEdgePolicy | None = None,
    ) -> None:
        self._store = store
        self._edge_policy = edge_policy or ResourceTreeEdgePolicy()

    def build(
        self,
        options: ResourceTreeOptions,
    ) -> list[ResourceTreeNode]:
        """Build a resource tree forest according to the supplied options."""
        edges_df = self._store.get_all_resource_relationships()
        edges = relationships_from_dataframe(edges_df)
        included_edges = [
            edge
            for edge in edges
            if self._edge_policy.include_edge(
                edge, all_relationships=options.all_relationships
            )
        ]

        node_kinds = self._node_kinds_from_edges(included_edges)
        downstream, upstream = self._adjacency_lists(included_edges)

        if options.scoped_root_identifier is not None:
            return self._build_scoped_tree(
                options=options,
                root_identifier=options.scoped_root_identifier,
                node_kinds=node_kinds,
                downstream=downstream,
                upstream=upstream,
            )

        roots = self._default_roots(
            options=options,
            edges=edges,
            included_edges=included_edges,
            node_kinds=node_kinds,
        )
        created_times = self._created_times_for_options(
            {
                node_id
                for root_id in roots
                for node_id in self._reachable(root_id, downstream)
            }
            | set(roots),
            sort=options.sort,
        )
        forest = [
            self._build_subtree(
                node_id=root_id,
                node_kinds=node_kinds,
                adjacency=downstream if not options.invert else upstream,
                options=options,
                created_times=created_times,
                visited_global=set() if not options.dedupe else None,
            )
            for root_id in sorted(
                roots,
                key=lambda identifier: _sort_key(
                    identifier, created_times.get(identifier)
                ),
            )
        ]

        if options.matching_identifiers is not None:
            forest = self._prune_to_matching(
                forest,
                matching_identifiers=options.matching_identifiers,
                invert=options.invert,
            )

        if options.kind_filter is not None:
            forest = self._prune_to_kind_filter(
                forest,
                kind_filter=options.kind_filter,
                invert=options.invert,
            )

        if options.include_orphans:
            forest.extend(
                self._orphan_nodes(
                    edges=edges,
                    existing_roots={root.identifier for root in forest},
                    node_kinds=node_kinds,
                    created_times=created_times,
                )
            )

        return [node for node in forest if node.identifier or node.children]

    def matching_identifiers_for_selectors(
        self,
        field_selectors: list[dict[str, str]] | None,
    ) -> frozenset[str] | None:
        """Return resource identifiers matching all field selectors, or None if unset."""
        if not field_selectors:
            return None
        matching: set[str] = set()
        for kind in CoreResourceKinds:
            identifiers = self._store.getResourceIdentifiersOfKind(
                kind=kind.value,
                field_selectors=field_selectors,
            )
            matching.update(identifiers["IDENTIFIER"].tolist())
        return frozenset(matching)

    def _default_roots(
        self,
        *,
        options: ResourceTreeOptions,
        edges: list[ResourceRelationship],
        included_edges: list[ResourceRelationship],
        node_kinds: dict[str, str],
    ) -> list[str]:
        sample_stores = self._store.getResourceIdentifiersOfKind(
            kind=CoreResourceKinds.SAMPLESTORE.value
        )
        roots = sample_stores["IDENTIFIER"].tolist()

        if options.all_relationships:
            objects = {edge.object for edge in edges}
            for edge in included_edges:
                if (
                    self._edge_policy.is_input_reference_edge(edge)
                    and edge.subject not in objects
                ):
                    roots.append(edge.subject)

        return list(dict.fromkeys(roots))

    def _build_scoped_tree(
        self,
        *,
        options: ResourceTreeOptions,
        root_identifier: str,
        node_kinds: dict[str, str],
        downstream: dict[str, list[str]],
        upstream: dict[str, list[str]],
    ) -> list[ResourceTreeNode]:
        adjacency = upstream if options.invert else downstream
        reachable = self._reachable(root_identifier, adjacency)
        created_times = self._created_times_for_options(
            reachable | {root_identifier},
            sort=options.sort,
        )
        tree = self._build_subtree(
            node_id=root_identifier,
            node_kinds=node_kinds,
            adjacency=adjacency,
            options=options,
            created_times=created_times,
            visited_global=set() if not options.dedupe else None,
        )
        forest = [tree]
        if options.matching_identifiers is not None:
            forest = self._prune_to_matching(
                forest,
                matching_identifiers=options.matching_identifiers,
                invert=options.invert,
            )
        if options.kind_filter is not None:
            forest = self._prune_to_kind_filter(
                forest,
                kind_filter=options.kind_filter,
                invert=options.invert,
            )
        return forest

    def _node_kinds_from_edges(
        self, edges: list[ResourceRelationship]
    ) -> dict[str, str]:
        kinds: dict[str, str] = {}
        for edge in edges:
            kinds[edge.subject] = edge.subject_kind
            kinds[edge.object] = edge.object_kind
        return kinds

    def _adjacency_lists(
        self, edges: list[ResourceRelationship]
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        downstream: dict[str, list[str]] = {}
        upstream: dict[str, list[str]] = {}
        for edge in edges:
            downstream.setdefault(edge.subject, []).append(edge.object)
            upstream.setdefault(edge.object, []).append(edge.subject)
        return downstream, upstream

    def _build_subtree(
        self,
        *,
        node_id: str,
        node_kinds: dict[str, str],
        adjacency: dict[str, list[str]],
        options: ResourceTreeOptions,
        created_times: dict[str, datetime.datetime | None],
        visited_global: set[str] | None,
        current_depth: int = 0,
    ) -> ResourceTreeNode:
        kind = node_kinds.get(node_id, "")
        if visited_global is not None and node_id in visited_global:
            return ResourceTreeNode(identifier=node_id, kind=kind, children=[])
        if visited_global is not None:
            visited_global.add(node_id)

        if options.depth is not None and current_depth >= options.depth:
            return ResourceTreeNode(identifier=node_id, kind=kind, children=[])

        child_ids = adjacency.get(node_id, [])
        child_ids = sorted(
            child_ids,
            key=lambda identifier: _sort_key(identifier, created_times.get(identifier)),
        )
        children = [
            self._build_subtree(
                node_id=child_id,
                node_kinds=node_kinds,
                adjacency=adjacency,
                options=options,
                created_times=created_times,
                visited_global=visited_global,
                current_depth=current_depth + 1,
            )
            for child_id in child_ids
        ]
        return ResourceTreeNode(identifier=node_id, kind=kind, children=children)

    def _created_times_for_options(
        self, identifiers: set[str], *, sort: bool
    ) -> dict[str, datetime.datetime | None]:
        """Return created timestamps when sorting is enabled; otherwise empty."""
        if not sort or not identifiers:
            return {}
        return self._created_times_for_identifiers(identifiers)

    def _created_times_for_identifiers(
        self, identifiers: set[str]
    ) -> dict[str, datetime.datetime | None]:
        if not identifiers:
            return {}
        resources = self._store.getResources(list(identifiers))
        return {
            identifier: (
                resources[identifier].created if identifier in resources else None
            )
            for identifier in identifiers
        }

    def _reachable(self, start: str, adjacency: dict[str, list[str]]) -> set[str]:
        visited: set[str] = set()
        stack = [start]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(adjacency.get(current, []))
        visited.discard(start)
        return visited

    def _prune_to_matching(
        self,
        forest: list[ResourceTreeNode],
        *,
        matching_identifiers: frozenset[str],
        invert: bool,
    ) -> list[ResourceTreeNode]:
        visible: set[str] = set()

        def _mark_ancestors(node: ResourceTreeNode, path: list[str]) -> None:
            if node.identifier in matching_identifiers:
                visible.update(path)
                visible.add(node.identifier)
            for child in node.children:
                _mark_ancestors(child, [*path, node.identifier])

        for root in forest:
            _mark_ancestors(root, [])

        return self._prune_forest(forest, visible)

    def _prune_to_kind_filter(
        self,
        forest: list[ResourceTreeNode],
        *,
        kind_filter: frozenset[str],
        invert: bool,
    ) -> list[ResourceTreeNode]:
        visible: set[str] = set()

        def _mark(node: ResourceTreeNode, path: list[str]) -> None:
            if node.kind in kind_filter:
                visible.update(path)
                visible.add(node.identifier)
            for child in node.children:
                _mark(child, [*path, node.identifier])

        for root in forest:
            _mark(root, [])

        return self._prune_forest(forest, visible)

    def _prune_forest(
        self, forest: list[ResourceTreeNode], visible: set[str]
    ) -> list[ResourceTreeNode]:
        def _prune(node: ResourceTreeNode) -> ResourceTreeNode | None:
            if node.identifier not in visible:
                return None
            pruned_children = [
                pruned
                for child in node.children
                if (pruned := _prune(child)) is not None
            ]
            return ResourceTreeNode(
                identifier=node.identifier,
                kind=node.kind,
                name=node.name,
                description=node.description,
                age=node.age,
                labels=node.labels,
                children=pruned_children,
            )

        pruned_forest: list[ResourceTreeNode] = []
        for root in forest:
            pruned = _prune(root)
            if pruned is not None:
                pruned_forest.append(pruned)
        return pruned_forest

    def _orphan_nodes(
        self,
        *,
        edges: list[ResourceRelationship],
        existing_roots: set[str],
        node_kinds: dict[str, str],
        created_times: dict[str, datetime.datetime | None],
    ) -> list[ResourceTreeNode]:
        related = set()
        for edge in edges:
            related.add(edge.subject)
            related.add(edge.object)
        all_identifiers = set(node_kinds.keys())
        for kind in CoreResourceKinds:
            identifiers = self._store.getResourceIdentifiersOfKind(kind=kind.value)
            all_identifiers.update(identifiers["IDENTIFIER"].tolist())
        orphans = sorted(
            all_identifiers - related - existing_roots,
            key=lambda identifier: _sort_key(identifier, created_times.get(identifier)),
        )
        return [
            ResourceTreeNode(
                identifier=orphan_id,
                kind=node_kinds.get(orphan_id, ""),
                children=[],
            )
            for orphan_id in orphans
        ]
