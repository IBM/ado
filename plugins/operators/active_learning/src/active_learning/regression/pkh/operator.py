# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""ADO operator registration for PKH."""

from __future__ import annotations

from active_learning.regression._shared import _run_predictive_operator
from active_learning.regression.pkh.parameters import PKHOperatorParameters
from active_learning.regression.pkh.sampler import PKHSampleSelector
from ado.core.discoveryspace.space import DiscoverySpace  # noqa: TC001
from ado.core.operation.config import FunctionOperationInfo  # noqa: TC001
from ado.core.operation.operation import OperationOutput  # noqa: TC001
from ado.modules.operators.collections import characterize_operation


@characterize_operation(
    name="pkh",
    configuration_model=PKHOperatorParameters,
    example_configuration=PKHOperatorParameters.example_configuration(),
    description=(
        "Selects entities from a finite Discovery Space using predictive "
        "kernel herding."
    ),
    version="0.1.0",
)
def pkh(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
    **kwargs: object,
) -> OperationOutput:
    """Characterize a finite discovery space with PKH acquisition."""

    parameters = PKHOperatorParameters.model_validate(kwargs)
    return _run_predictive_operator(
        discovery_space=discoverySpace,
        operation_info=operationInfo,
        parameters=parameters,
        number_entities=parameters.numberEntities,
        sampler_class=PKHSampleSelector,
    )
