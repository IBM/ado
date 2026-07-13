# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from ado.cli.models.parameters import AdoEditCommandParameters
from ado.cli.utils.resources.handlers import handle_edit_resource_metadata
from ado.core.resources import CoreResourceKinds


def edit_operation(parameters: AdoEditCommandParameters) -> None:
    handle_edit_resource_metadata(
        resource_id=parameters.resource_id,
        resource_type=CoreResourceKinds.OPERATION,
        project_context=parameters.ado_configuration.project_context,
        editor=parameters.editor,
        metadata_path=parameters.metadata_path,
        metadata_patch=parameters.metadata_patch,
    )
