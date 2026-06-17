# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Render resource tree forests for CLI output."""

from __future__ import annotations

import json
import typing

if typing.TYPE_CHECKING:
    from rich.tree import Tree

    from orchestrator.core.resource_tree import ResourceTreeNode


def format_tree_node_label(
    node: ResourceTreeNode,
    *,
    show_names: bool,
    show_age: bool,
    show_metadata: bool,
) -> str:
    """Format a single tree node label for text output."""
    label = node.identifier
    if show_names and node.name:
        label = f"{node.identifier} ({node.name})"
    detail_parts: list[str] = []
    if show_age and node.age:
        detail_parts.append(f"age={node.age}")
    if show_metadata:
        if node.description:
            detail_parts.append(node.description)
        if node.labels:
            detail_parts.append(f"labels={json.dumps(node.labels)}")
    if detail_parts:
        label = f"{label} [{' | '.join(detail_parts)}]"
    return label


def render_tree_forest_to_rich(
    roots: list[ResourceTreeNode],
    *,
    show_names: bool,
    show_age: bool,
    show_metadata: bool,
) -> Tree:
    """Build a Rich tree containing one branch per root node."""
    from rich.tree import Tree

    forest = Tree("resources")
    for root in roots:
        _add_rich_subtree(
            forest,
            root,
            show_names=show_names,
            show_age=show_age,
            show_metadata=show_metadata,
        )
    return forest


def _add_rich_subtree(
    parent: Tree,
    node: ResourceTreeNode,
    *,
    show_names: bool,
    show_age: bool,
    show_metadata: bool,
) -> None:
    branch = parent.add(
        format_tree_node_label(
            node,
            show_names=show_names,
            show_age=show_age,
            show_metadata=show_metadata,
        )
    )
    for child in node.children:
        _add_rich_subtree(
            branch,
            child,
            show_names=show_names,
            show_age=show_age,
            show_metadata=show_metadata,
        )


def render_tree_forest_to_text(
    roots: list[ResourceTreeNode],
    *,
    show_names: bool,
    show_age: bool,
    show_metadata: bool,
) -> str:
    """Render a resource tree forest as plain text using Rich."""
    from orchestrator.utilities.rich import render_to_string

    if not roots:
        return ""
    return render_to_string(
        render_tree_forest_to_rich(
            roots,
            show_names=show_names,
            show_age=show_age,
            show_metadata=show_metadata,
        )
    )


def render_tree_forest_to_json(roots: list[ResourceTreeNode]) -> str:
    """Render a resource tree forest as JSON."""

    def _node_to_dict(node: ResourceTreeNode) -> dict[str, typing.Any]:
        return {
            "identifier": node.identifier,
            "kind": node.kind,
            "name": node.name,
            "description": node.description,
            "age": node.age,
            "labels": node.labels,
            "children": [_node_to_dict(child) for child in node.children],
        }

    return json.dumps([_node_to_dict(root) for root in roots], indent=2)


def render_tree_forest_to_flat(roots: list[ResourceTreeNode]) -> str:
    """Render a resource tree forest as flat depth-parent-id-kind rows."""

    rows: list[str] = ["depth\tparent\tidentifier\tkind"]

    def _walk(
        node: ResourceTreeNode,
        *,
        depth: int,
        parent_identifier: str,
    ) -> None:
        rows.append(
            f"{depth}\t{parent_identifier}\t{node.identifier}\t{node.kind}",
        )
        for child in node.children:
            _walk(child, depth=depth + 1, parent_identifier=node.identifier)

    for root in roots:
        _walk(root, depth=0, parent_identifier="")

    return "\n".join(rows) + "\n"
