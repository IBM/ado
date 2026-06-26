#!/usr/bin/env python3
# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Report identifiers of stored resources that fail validation.

Usage:
    uv run scripts/validate_resources.py <resource-type>
    uv run scripts/validate_resources.py <resource-type> --context <context-name>

Arguments:
    resource-type   One of: operation, discoveryspace, samplestore,
                    actuatorconfiguration, datacontainer

Options:
    --context TEXT  ado context name to use (default: active context)

Examples:
    uv run scripts/validate_resources.py operation
    uv run scripts/validate_resources.py discoveryspace --context my-project
"""

import argparse
import sys

from rich.status import Status

from orchestrator.cli.core.config import AdoConfiguration
from orchestrator.cli.utils.output.prints import ADO_SPINNER_QUERYING_DB
from orchestrator.core import kindmap
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.metastore.sqlstore import SQLStore
from orchestrator.utilities.pydantic import ignore_plugin_validation_context


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Report resource identifiers that fail validation.",
    )
    parser.add_argument(
        "resource_type",
        choices=[k.value for k in CoreResourceKinds],
        metavar="resource-type",
        help=(
            "Type of resource to validate. "
            f"One of: {', '.join(k.value for k in CoreResourceKinds)}"
        ),
    )
    parser.add_argument(
        "--context",
        default=None,
        metavar="CONTEXT",
        help="ado context name to use (default: active context).",
    )
    return parser.parse_args()


def main() -> None:
    """Load all resources of the given type and report those that fail validation."""
    args = parse_args()

    # Load ado configuration and resolve the project context
    ado_config = AdoConfiguration.load()

    if args.context is not None:
        project_context = ado_config.project_context_model_for_context(args.context)
    else:
        project_context = ado_config.project_context

    if project_context is None:
        print("ERROR: no active ado context found. Run 'ado context' to set one.")
        sys.exit(1)

    sql_store = SQLStore(project_context=project_context)

    kind = CoreResourceKinds(args.resource_type)
    resource_class = kindmap[kind.value]

    with Status(ADO_SPINNER_QUERYING_DB) as status:
        identifiers_df = sql_store.getResourceIdentifiersOfKind(kind=kind.value)
        identifiers: list[str] = identifiers_df["IDENTIFIER"].tolist()

        total = len(identifiers)
        status.update(
            f"Validating {total} {kind.value} resource(s) "
            f"in context '{project_context.project}' …"
        )

        failed: list[tuple[str, str]] = []

        for idx, identifier in enumerate(identifiers, start=1):
            status.update(f"Validating {kind.value} resources ({idx}/{total}) …")
            raw = sql_store.getResourceRaw(identifier)
            if raw is None:
                continue
            try:
                resource_class.model_validate(
                    raw, context=ignore_plugin_validation_context
                )
            except Exception as exc:
                failed.append((identifier, str(exc)))

    if not failed:
        print("All resources passed validation.")
        return

    print(f"\n{len(failed)} resource(s) failed validation:\n")
    for identifier, error in failed:
        print(f"  {identifier}")
        print(f"    {error}\n")

    sys.exit(1)


if __name__ == "__main__":
    main()
