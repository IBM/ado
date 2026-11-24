# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Predictive Kernel Herding (PKH) characterize operator for ADO."""

import contextlib
import datetime as dt
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from active_learning.pkh.data import (
    candidate_entities,
    encode_feature_matrix,
    measured_labels,
    validate_target_output,
)
from active_learning.pkh.parameters import FinalPredictorMetadata, PKHParameters
from active_learning.pkh.sampler import PKHState, build_forest
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.config import FunctionOperationInfo
from ado.core.operation.operation import OperationOutput
from ado.core.operation.resource import OperationExitStateEnum
from ado.modules.operators.collections import characterize_operation, explore

if TYPE_CHECKING:
    from autogluon.tabular import TabularPredictor

    from ado.modules.operators.randomwalk import RandomWalkParameters

logger = logging.getLogger(__name__)

PKH_VERSION = "0.1.0"
_MIN_AUTOGLUON_LABELS = 4


def _fit_predictor(
    state: PKHState,
    frame: pd.DataFrame,
    target_name: str,
    encoder: ColumnTransformer,
    final_directory: Path,
) -> "Pipeline | TabularPredictor":
    """Fit and persist the final predictor on the measured entities."""
    parameters = state.params
    if parameters.finalPredictor == "autogluon":
        from autogluon.tabular import TabularPredictor

        model = TabularPredictor(
            label=target_name,
            problem_type="regression",
            path=str(final_directory / "autogluon"),
            verbosity=1,
        )
        fit_kwargs = (
            parameters.autoGluon.model_dump(by_alias=True, exclude_none=True)
            if parameters.autoGluon
            else {}
        )
        train_data = frame.iloc[state.measured_indices].copy()
        train_data[target_name] = state.measured_labels
        categorical_columns = train_data.select_dtypes(include="category").columns
        if not categorical_columns.empty:
            from active_learning.pkh.autogluon import PKHCategoricalFeatureGenerator

            train_data = train_data.astype(dict.fromkeys(categorical_columns, "string"))
            fit_kwargs["feature_generator"] = PKHCategoricalFeatureGenerator()
        model.fit(train_data=train_data, **fit_kwargs)
        return model
    forest = state.forest
    if forest is None:
        forest = build_forest(parameters, parameters.seed).fit(
            state.features[state.measured_indices], state.measured_labels
        )
    model = Pipeline([("features", encoder), ("forest", forest)])
    with (final_directory / "model.pkl").open("wb") as model_file:
        pickle.dump(model, model_file, protocol=5)
    return model


def _finalize_model(
    discovery_space: DiscoverySpace,
    parameters: PKHParameters,
    final_directory: Path,
    sampled_identifiers: set[str],
) -> float | None:
    """Save a predictor, full-space predictions, and MMD in a reserved directory."""
    target = validate_target_output(parameters.targetOutput, discovery_space)
    target_name = target.targetProperty.identifier
    # Refresh labels before enumerating the pool to retain stored entity identifiers.
    labels = measured_labels(discovery_space, target.identifier)
    if missing := sampled_identifiers.difference(labels):
        raise ValueError(
            f"Missing finite target measurements for '{target.identifier}': "
            f"{', '.join(sorted(missing))}"
        )
    entities = candidate_entities(discovery_space)
    frame, features, encoder = encode_feature_matrix(discovery_space, entities)
    frame.index = [entity.identifier for entity in entities]
    target_values = pd.Series(labels, dtype=float).reindex(frame.index)
    measured = target_values.notna().to_numpy()
    measured_count = int(measured.sum())
    if (
        parameters.finalPredictor == "autogluon"
        and measured_count < _MIN_AUTOGLUON_LABELS
    ):
        raise ValueError(
            f"PKH requires at least {_MIN_AUTOGLUON_LABELS} target labels for AutoGluon "
            f"and collected {measured_count}."
        )
    if measured_count == 0:
        raise ValueError(f"No completed measurements for '{target.identifier}'.")
    if len(sampled_identifiers) < parameters.numberEntities and not measured.all():
        raise ValueError(
            "PKH measurement budget is incomplete: collected "
            f"{len(sampled_identifiers)} of {parameters.numberEntities} new entities "
            "while unmeasured candidates remain."
        )

    state = PKHState(
        features,
        initial_measured_indices=np.flatnonzero(measured).tolist(),
        initial_labels=target_values[measured].tolist(),
        params=parameters,
    )
    state.fit_forest_and_histograms()
    mmd_squared = state.mmd_squared()

    model = _fit_predictor(state, frame, target_name, encoder, final_directory)

    table = frame.assign(entityIdentifier=frame.index)
    table[target_name] = target_values
    table["is_predicted"] = ~measured
    if (~measured).any():
        table.loc[~measured, target_name] = np.asarray(
            model.predict(frame.loc[~measured])
        )
    table.to_csv(final_directory / "combined_predictions.csv", index=False)
    metadata = FinalPredictorMetadata(
        backend=parameters.finalPredictor,
        targetOutput=target_name,
        featureNames=frame.columns.tolist(),
        categoricalColumns=frame.select_dtypes(include="category").columns.tolist(),
        measuredCount=measured_count,
        createdTimestamp=dt.datetime.now(dt.UTC).isoformat(),
        pkhVersion=PKH_VERSION,
        mmdSquared=mmd_squared,
    )
    (final_directory / "metadata.json").write_text(metadata.model_dump_json(indent=2))
    return mmd_squared


def _sampling_parameters(parameters: PKHParameters) -> "RandomWalkParameters":
    """Configure a sequential measurement run using PKH's feedback-driven sampler."""
    # RandomWalk loads operator plugins, so import its configuration after registration.
    from ado.modules.operators.randomwalk import (
        CustomSamplerConfiguration,
        RandomWalkParameters,
        SamplerModuleConf,
    )

    return RandomWalkParameters(
        batchSize=1,
        numberEntities=parameters.numberEntities,
        samplerConfig=CustomSamplerConfiguration(
            module=SamplerModuleConf(
                moduleName="active_learning.pkh.sampler",
                moduleClass="PKHSampleSelector",
            ),
            parameters=parameters,
        ),
    )


def _output_after_sampling(
    walk: OperationOutput,
    discovery_space: DiscoverySpace,
    parameters: PKHParameters,
    final_directory: Path,
) -> OperationOutput:
    """Preserve the measurement run's status and publish a model on success."""
    result = OperationOutput(resources=[walk.operation], exitStatus=walk.exitStatus)
    if walk.exitStatus and walk.exitStatus.exit_state == OperationExitStateEnum.SUCCESS:
        mmd_squared = _finalize_model(
            discovery_space,
            parameters,
            final_directory,
            discovery_space.sample_store.entity_identifiers_in_operations(
                walk.operation.identifier
            ),
        )
        logger.info("PKH model and predictions finalized at: %s", final_directory)
        if mmd_squared is not None:
            result.metadata["pkhMmdSquared"] = mmd_squared
    return result


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
    if parameters.finalPredictor == "autogluon":
        label_count = parameters.numberEntities
        if label_count < _MIN_AUTOGLUON_LABELS:
            label_count += len(measured_labels(discoverySpace, target.identifier))
        max_labels = min(
            label_count, getattr(discoverySpace.entitySpace, "size", label_count)
        )
        if max_labels < _MIN_AUTOGLUON_LABELS:
            raise ValueError(
                f"PKH requires at least {_MIN_AUTOGLUON_LABELS} target labels for AutoGluon "
                f"and can reach at most {max_labels} with this budget and space."
            )
    sampling_parameters = _sampling_parameters(parameters)
    random_walk = explore.operators["random_walk"].function
    if random_walk is None:
        raise RuntimeError("The random_walk operator is not available.")
    base_directory = Path(parameters.outputDirectory).expanduser().resolve()
    final_directory = base_directory.with_name(f"{base_directory.name}_finalized")
    final_directory.mkdir(parents=True, exist_ok=False)
    try:
        walk = random_walk(
            discoverySpace, operationInfo, parameters=sampling_parameters
        )
        return _output_after_sampling(walk, discoverySpace, parameters, final_directory)
    finally:
        with contextlib.suppress(OSError):
            final_directory.rmdir()
