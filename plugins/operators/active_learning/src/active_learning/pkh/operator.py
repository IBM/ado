# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Predictive Kernel Herding (PKH) characterize operator for ADO."""

import numpy as np
import pandas as pd

from active_learning.pkh.data import (
    candidate_entities,
    encode_feature_matrix,
    feature_frame,
    measured_labels,
    validate_candidate_space,
    validate_target_output,
)
from active_learning.pkh.parameters import PKHParameters
from active_learning.pkh.sampler import PKHState
from ado.core.datacontainer.resource import (
    DataContainer,
    DataContainerResource,
    TabularData,
)
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.metadata import ConfigurationMetadata
from ado.core.operation.config import FunctionOperationInfo
from ado.core.operation.operation import OperationOutput
from ado.core.operation.resource import OperationExitStateEnum
from ado.modules.operators.collections import characterize_operation, explore
from ado.schema.observed_property import ObservedProperty

PKH_VERSION = "0.1.0"
_MIN_AUTOGLUON_LABELS = 4


def _validate_prediction(
    parameters: PKHParameters, label_count: int, *, before_sampling: bool = False
) -> None:
    """Check the available labels and libraries before fitting the predictor."""
    if label_count < _MIN_AUTOGLUON_LABELS:
        available = "can reach at most" if before_sampling else "collected"
        raise ValueError(
            f"PKH requires at least {_MIN_AUTOGLUON_LABELS} target labels for AutoGluon "
            f"and {available} {label_count}."
        )
    parameters.validate_autogluon_installation()


def _complete_dataset(
    discovery_space: DiscoverySpace,
    parameters: PKHParameters,
    target: ObservedProperty,
    sampled_identifiers: set[str],
) -> tuple[DataContainerResource, float | None]:
    """Combine finite observations with predictions for the remaining candidates."""
    # Refresh labels before enumerating the pool to retain stored entity identifiers.
    labels = measured_labels(discovery_space, target.identifier)
    if missing := sampled_identifiers.difference(labels):
        raise ValueError(
            f"Missing finite target measurements for '{target.identifier}': "
            f"{', '.join(sorted(missing))}"
        )
    entities = candidate_entities(discovery_space)
    frame = feature_frame(discovery_space, entities)
    frame.index = [entity.identifier for entity in entities]
    target_values = pd.Series(labels, dtype=float).reindex(frame.index)
    measured = target_values.notna().to_numpy()
    measured_count = int(measured.sum())
    if measured_count == 0:
        raise ValueError(f"No completed measurements for '{target.identifier}'.")

    target_name = target.targetProperty.identifier
    table = frame.assign(entityIdentifier=frame.index)
    table[target_name] = target_values
    table["is_predicted"] = ~measured
    metadata = ConfigurationMetadata(name="PKH predictions")
    # The full pool has zero discrepancy with itself for any fitted forest.
    mmd_squared = 0.0 if measured_count >= 2 else None
    if not measured.all():
        if len(sampled_identifiers) < parameters.numberEntities:
            raise ValueError(
                "PKH measurement budget is incomplete: collected "
                f"{len(sampled_identifiers)} of {parameters.numberEntities} new entities "
                "while unmeasured candidates remain."
            )
        _validate_prediction(parameters, measured_count)
        from active_learning.pkh.autogluon import predict_unmeasured

        state = PKHState(
            encode_feature_matrix(frame),
            initial_measured_indices=np.flatnonzero(measured).tolist(),
            initial_labels=target_values[measured].tolist(),
            params=parameters,
        )
        state.fit_forest_and_histograms()
        mmd_squared = state.mmd_squared()
        predictions, model_name = predict_unmeasured(
            frame, target_values, target_name, parameters
        )
        if not np.isfinite(predictions).all():
            raise ValueError(f"Non-finite predictions for '{target.identifier}'.")
        table.loc[~measured, target_name] = predictions
        metadata.labels = {"predictor": model_name}
    dataset = DataContainerResource(
        config=DataContainer(
            tabularData={"predictions": TabularData.from_dataframe(table)},
            metadata=metadata,
        )
    )
    return dataset, mmd_squared


def _output_after_sampling(
    walk: OperationOutput,
    discovery_space: DiscoverySpace,
    parameters: PKHParameters,
    target: ObservedProperty,
) -> OperationOutput:
    """Return the measurement operation and publish its completed dataset on success."""
    result = OperationOutput(resources=[walk.operation], exitStatus=walk.exitStatus)
    if walk.exitStatus and walk.exitStatus.exit_state == OperationExitStateEnum.SUCCESS:
        dataset, mmd_squared = _complete_dataset(
            discovery_space,
            parameters,
            target,
            discovery_space.sample_store.entity_identifiers_in_operations(
                walk.operation.identifier
            ),
        )
        result.resources.append(dataset)
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
    """Measure entities with PKH and return the completed table for the finite space."""
    # RandomWalk loads operator plugins during registration.
    from ado.modules.operators.randomwalk import (
        CustomSamplerConfiguration,
        RandomWalkParameters,
        SamplerModuleConf,
    )

    target = validate_target_output(parameters.targetOutput, discoverySpace)
    pool_size = validate_candidate_space(discoverySpace.entitySpace)
    if parameters.numberEntities < pool_size:
        prior = measured_labels(discoverySpace, target.identifier)
        attainable = len(prior) + parameters.numberEntities
        if attainable < pool_size:
            _validate_prediction(parameters, attainable, before_sampling=True)
    random_walk = explore.operators["random_walk"].function
    if random_walk is None:
        raise RuntimeError("The random_walk operator is not available.")
    sampling = RandomWalkParameters(
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
    walk = random_walk(discoverySpace, operationInfo, parameters=sampling)
    return _output_after_sampling(walk, discoverySpace, parameters, target)
