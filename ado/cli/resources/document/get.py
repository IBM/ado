# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from ado.cli.models.parameters import AdoGetCommandParameters
from ado.cli.utils.resources.handlers import handle_ado_get
from ado.core.resources import CoreResourceKinds


def get_document(parameters: AdoGetCommandParameters) -> None:
    """Get one or more document resources."""
    handle_ado_get(parameters=parameters, resource_type=CoreResourceKinds.DOCUMENT)
