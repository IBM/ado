# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from orchestrator.cli.models.parameters import AdoUpgradeCommandParameters
from orchestrator.core import CoreResourceKinds


def upgrade_data_container(parameters: AdoUpgradeCommandParameters) -> None:
    from orchestrator.cli.utils.resources.handlers import (
        handle_ado_upgrade,
    )

    handle_ado_upgrade(
        parameters=parameters, resource_type=CoreResourceKinds.DATACONTAINER
    )
