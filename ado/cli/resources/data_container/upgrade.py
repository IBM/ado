# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from ado.cli.models.parameters import AdoUpgradeCommandParameters
from ado.core import CoreResourceKinds


def upgrade_data_container(parameters: AdoUpgradeCommandParameters) -> None:
    from ado.cli.utils.resources.handlers import (
        handle_ado_upgrade,
    )

    handle_ado_upgrade(
        parameters=parameters, resource_type=CoreResourceKinds.DATACONTAINER
    )
