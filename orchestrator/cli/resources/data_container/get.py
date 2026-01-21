# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from orchestrator.cli.models.parameters import AdoGetCommandParameters
from orchestrator.cli.models.types import (
    AdoGetSupportedOutputFormats,
)
from orchestrator.cli.utils.resources.handlers import (
    handle_ado_get_special_formats,
)
from orchestrator.core.resources import CoreResourceKinds


def get_data_container(parameters: AdoGetCommandParameters) -> None:
    from orchestrator.cli.utils.resources.handlers import (
        handle_ado_get_default_format,
    )

    if parameters.output_format == AdoGetSupportedOutputFormats.DEFAULT:
        handle_ado_get_default_format(
            parameters=parameters,
            resource_type=CoreResourceKinds.DATACONTAINER,
        )
    else:
        handle_ado_get_special_formats(
            parameters=parameters,
            resource_type=CoreResourceKinds.DATACONTAINER,
        )
