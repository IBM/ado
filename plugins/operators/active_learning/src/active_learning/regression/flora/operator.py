# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""ADO operator registration for FLORA."""

from __future__ import annotations

from active_learning.regression._shared import _run_predictive_operator
from active_learning.regression.flora.parameters import FLORAOperatorParameters
from active_learning.regression.flora.sampler import FLORASampleSelector
from ado.core.discoveryspace.space import DiscoverySpace  # noqa: TC001
from ado.core.operation.config import FunctionOperationInfo  # noqa: TC001
from ado.core.operation.operation import OperationOutput  # noqa: TC001
from ado.modules.operators.collections import characterize_operation


@characterize_operation(
    name="flora",
    configuration_model=FLORAOperatorParameters,
    example_configuration=FLORAOperatorParameters.example_configuration(),
    description=(
        "Selects entities from a finite Discovery Space using pointwise "
        "random-forest disagreement."
    ),
    version="0.1.0",
)
def flora(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
    **kwargs: object,
) -> OperationOutput:
    """Characterize a finite discovery space with FLORA acquisition."""

    parameters = FLORAOperatorParameters.model_validate(kwargs)
    return _run_predictive_operator(
        discovery_space=discoverySpace,
        operation_info=operationInfo,
        parameters=parameters,
        number_entities=parameters.numberEntities,
        sampler_class=FLORASampleSelector,
    )
