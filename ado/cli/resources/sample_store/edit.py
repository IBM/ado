# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from ado.cli.models.parameters import AdoEditCommandParameters
from ado.cli.utils.resources.handlers import (
    apply_patch_to_resources,
    interactively_edit_resource_metadata,
)
from ado.core.resources import CoreResourceKinds


def edit_sample_store(parameters: AdoEditCommandParameters) -> None:
    """Edit metadata on a sample store resource."""
    if parameters.metadata_patch is not None or parameters.metadata_path is not None:
        apply_patch_to_resources(
            resource_ids=parameters.resource_ids,
            resource_type=CoreResourceKinds.SAMPLESTORE,
            project_context=parameters.ado_configuration.project_context,
            metadata_patch=parameters.metadata_patch,
            metadata_path=parameters.metadata_path,
        )
    else:
        interactively_edit_resource_metadata(
            resource_id=parameters.resource_ids[0],
            resource_type=CoreResourceKinds.SAMPLESTORE,
            project_context=parameters.ado_configuration.project_context,
            editor=parameters.editor,
        )
