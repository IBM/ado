# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""FLORA characterize operator for ADO."""

from active_learning.data import measured_labels, validate_target_output
from active_learning.flora.parameters import FLORAParameters
from active_learning.flora.sampler import FLORASampleSelector
from active_learning.operator import PredictorExport, collect_measurements
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.config import FunctionOperationInfo
from ado.core.operation.operation import OperationOutput
from ado.modules.operators.collections import characterize_operation

FLORA_VERSION = "0.2.0"


def _require_autogluon_labels(count: int) -> None:
    """Check that AutoGluon has enough labels for its training and validation sets."""
    if count < 4:
        raise ValueError(
            f"FLORA requires at least four target measurements for AutoGluon; found {count}."
        )


def _finalize_model(export: PredictorExport, parameters: FLORAParameters) -> None:
    """Train the chosen final predictor on all available FLORA labels and save it."""
    if parameters.finalPredictor == "autogluon":
        _require_autogluon_labels(int(export.measured.sum()))
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
    if parameters.finalPredictor == "autogluon":
        prior_labels = measured_labels(
            discoverySpace, target.identifier, FLORASampleSelector.target_label
        )
        _require_autogluon_labels(
            min(
                discoverySpace.entitySpace.size,
                len(prior_labels) + parameters.numberEntities,
            )
        )
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
