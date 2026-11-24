# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""FLORA characterize operator for ADO."""

from active_learning.data import validate_target_output
from active_learning.flora.parameters import FLORAParameters
from active_learning.flora.sampler import FLORASampleSelector
from active_learning.operator import PredictorExport, collect_measurements
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.config import FunctionOperationInfo
from ado.core.operation.operation import OperationOutput
from ado.modules.operators.collections import characterize_operation

FLORA_VERSION = "0.2.0"


def _finalize_model(export: PredictorExport, parameters: FLORAParameters) -> None:
    """Train the chosen final predictor on all available FLORA labels and save it."""
    if parameters.finalPredictor == "autogluon":
        model = export.fit_autogluon(
            numeric_as_float=True, random_state=parameters.seed
        )
    else:
        model = export.save_forest()
    export.save_predictions(model)


@characterize_operation(
    name="flora",
    configuration_model=FLORAParameters,
    example_configuration=FLORAParameters.example_configuration(),
    description="Selects entities using disagreement between Random Forest predictions.",
    version=FLORA_VERSION,
)
def flora(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
    *,
    parameters: FLORAParameters,
) -> OperationOutput:
    """Collect measurements with FLORA and save a final predictor of the target property."""
    target = validate_target_output(parameters.targetOutput, discoverySpace)
    with collect_measurements(
        discoverySpace,
        operationInfo,
        parameters,
        target,
        FLORASampleSelector,
    ) as (output, export):
        if export is not None:
            _finalize_model(export, parameters)
        return output
