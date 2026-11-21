# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Predictive Kernel Herding (PKH) characterize operator for ADO."""

import numpy as np

from active_learning.data import validate_target_output
from active_learning.operator import PredictorExport, collect_measurements
from active_learning.pkh.parameters import FinalPredictorMetadata, PKHParameters
from active_learning.pkh.sampler import PKHSampleSelector, PKHState
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.config import FunctionOperationInfo
from ado.core.operation.operation import OperationOutput
from ado.modules.operators.collections import characterize_operation

PKH_VERSION = "0.2.0"


def _finalize_model(export: PredictorExport, parameters: PKHParameters) -> float | None:
    """Fit the final predictor and report the measured sample's MMD."""
    state = PKHState(
        export.features,
        initial_measured_indices=np.flatnonzero(export.measured).tolist(),
        initial_labels=export.labels[export.measured].tolist(),
        params=parameters,
    )
    state.fit_forest_and_histograms()
    mmd_squared = state.mmd_squared()
    if parameters.finalPredictor == "autogluon":
        model = export.fit_autogluon()
    else:
        model = export.save_forest(state.forest)
    export.save_predictions(
        model, FinalPredictorMetadata, pkhVersion=PKH_VERSION, mmdSquared=mmd_squared
    )
    return mmd_squared


@characterize_operation(
    name="pkh",
    configuration_model=PKHParameters,
    example_configuration=PKHParameters.example_configuration(),
    description="Predictive Kernel Herding (PKH) active learning characterization operator.",
    version=PKH_VERSION,
)
def pkh(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
    *,
    parameters: PKHParameters,
) -> OperationOutput:
    """Collect measurements with PKH and save a final predictor of the target property."""
    target = validate_target_output(parameters.targetOutput, discoverySpace)
    with collect_measurements(
        discoverySpace, operationInfo, parameters, target, PKHSampleSelector
    ) as (output, export):
        if export is not None:
            mmd_squared = _finalize_model(export, parameters)
            if mmd_squared is not None:
                output.metadata["pkhMmdSquared"] = mmd_squared
        return output
