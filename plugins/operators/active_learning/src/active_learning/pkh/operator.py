# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Predictive Kernel Herding (PKH) characterize operator for ADO."""

import logging

import numpy as np
import pandas as pd

from active_learning.pkh.data import (
    candidate_coordinates,
    candidate_entities,
    encode_feature_matrix,
    feature_frame,
    measured_labels,
    validate_candidate_space,
    validate_target_output,
    walk_measurement_labels,
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
from ado.core.operation.operation import OperationException, OperationOutput
from ado.core.operation.resource import (
    OperationExitStateEnum,
    OperationResourceEventEnum,
    OperationResourceStatus,
)
from ado.modules.operators.collections import characterize_operation, explore
from ado.schema.entityspace import EntitySpaceRepresentation
from ado.schema.observed_property import ObservedProperty

logger = logging.getLogger(__name__)

PKH_VERSION = "0.1.0"
# AutoGluon's default holdout leaves only one training row with three labels.
_MIN_AUTOGLUON_LABELS = 4


def _validate_prediction(
    parameters: PKHParameters,
    label_count: int,
    entity_space: EntitySpaceRepresentation,
    *,
    before_sampling: bool = False,
) -> None:
    """Check the available labels and libraries before fitting the predictor."""
    if label_count < _MIN_AUTOGLUON_LABELS:
        available = "can reach at most" if before_sampling else "collected"
        raise ValueError(
            f"PKH requires at least {_MIN_AUTOGLUON_LABELS} target labels "
            f"for AutoGluon and {available} {label_count}."
        )
    parameters.validate_autogluon_installation()
    from active_learning.pkh.autogluon import validate_knn_prediction

    validate_knn_prediction(parameters, label_count, entity_space)


def _complete_dataset(
    discovery_space: DiscoverySpace,
    parameters: PKHParameters,
    target: ObservedProperty,
    sampled_labels: dict[str, int | float | None],
) -> tuple[DataContainerResource, float | None]:
    """Combine finite observations with predictions for the remaining candidates."""
    entities = candidate_entities(discovery_space)
    labels = measured_labels(discovery_space, target, entities)
    if missing := [
        identifier
        for identifier, label in sampled_labels.items()
        if label is None or identifier not in labels
    ]:
        raise ValueError(
            f"Missing finite target measurements for '{target.identifier}': "
            f"{', '.join(sorted(missing))}"
        )
    if not labels:
        raise ValueError(f"No completed measurements for '{target.identifier}'.")
    labels.update(
        (identifier, label)
        for identifier, label in sampled_labels.items()
        if label is not None
    )

    identifiers = [entity.identifier for entity in entities]
    measured = np.array([identifier in labels for identifier in identifiers])
    target_name = target.targetProperty.identifier
    coordinates = candidate_coordinates(entities)
    metadata = ConfigurationMetadata(name="PKH predictions")
    if len(labels) == len(entities):
        # The full pool has zero discrepancy with itself for any fitted forest.
        mmd_squared = 0.0 if len(labels) >= 2 else None
    else:
        if len(sampled_labels) < parameters.numberEntities:
            raise ValueError(
                "PKH measurement budget is incomplete: collected "
                f"{len(sampled_labels)} of {parameters.numberEntities} "
                "new entities while unmeasured candidates remain."
            )
        _validate_prediction(parameters, len(labels), discovery_space.entitySpace)
        from active_learning.pkh.autogluon import predict_unmeasured

        frame = feature_frame(discovery_space, coordinates)
        frame.index = identifiers
        target_values = pd.Series(labels, dtype=float).reindex(identifiers)
        state = PKHState(
            encode_feature_matrix(frame),
            initial_measured_indices=np.flatnonzero(measured).tolist(),
            initial_labels=target_values[measured].tolist(),
            params=parameters,
        )
        state.fit_forest_and_histograms()
        mmd_squared = state.mmd_squared()
        predictions, predictor_metadata = predict_unmeasured(
            frame, target_values, target_name, parameters
        )
        if not np.isfinite(predictions).all():
            raise ValueError(f"Non-finite predictions for '{target.identifier}'.")
        labels.update(zip(frame.index[~measured], predictions.tolist(), strict=True))
        metadata.labels = predictor_metadata
    columns = {
        **coordinates,
        "entityIdentifier": identifiers,
        target_name: [labels[identifier] for identifier in identifiers],
        "is_predicted": (~measured).tolist(),
    }
    dataset = DataContainerResource(
        config=DataContainer(
            tabularData={"predictions": TabularData(data=columns)},
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
    """Return the measurement operation and publish its completed dataset on success.

    If publication fails, keep the measurement operation in the output, so that
    ADO still records it as related to this operation, and report the failure.
    """
    result = OperationOutput(resources=[walk.operation], exitStatus=walk.exitStatus)
    if walk.exitStatus.exit_state != OperationExitStateEnum.SUCCESS:
        return result

    try:
        sampled_labels = walk_measurement_labels(
            discovery_space, target, walk.operation.identifier
        )
        if len(sampled_labels) < walk.operation.metadata.get("entities_submitted", 0):
            raise ValueError("The child walk is missing measurement results.")
        dataset, mmd_squared = _complete_dataset(
            discovery_space, parameters, target, sampled_labels
        )
    except Exception as error:
        logger.exception("PKH could not publish the completed dataset")
        result.exitStatus = OperationResourceStatus(
            event=OperationResourceEventEnum.FINISHED,
            exit_state=OperationExitStateEnum.ERROR,
            message=(
                "PKH measured the entities but could not publish the "
                f"completed dataset: {error}"
            ),
        )
        return result
    result.resources.append(dataset)
    if mmd_squared is not None:
        result.metadata["pkhMmdSquared"] = mmd_squared
    return result


@characterize_operation(
    name="pkh",
    configuration_model=PKHParameters,
    example_configuration=PKHParameters.example_configuration(),
    description=(
        "Predictive Kernel Herding (PKH) active learning characterization operator."
    ),
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
        prior = measured_labels(discoverySpace, target)
        attainable = len(prior) + parameters.numberEntities
        if attainable < pool_size:
            _validate_prediction(
                parameters, attainable, discoverySpace.entitySpace, before_sampling=True
            )
    random_walk = explore.operators["random_walk"].function
    if random_walk is None:
        raise RuntimeError("The random_walk operator is not available.")
    sampling = RandomWalkParameters(
        batchSize=1,
        singleMeasurement=False,
        numberEntities=parameters.numberEntities,
        samplerConfig=CustomSamplerConfiguration(
            module=SamplerModuleConf(
                moduleName="active_learning.pkh.sampler",
                moduleClass="PKHSampleSelector",
            ),
            parameters=parameters,
        ),
    )
    try:
        walk = random_walk(discoverySpace, operationInfo, parameters=sampling)
    except OperationException as error:
        return OperationOutput(
            resources=[error.operation],
            exitStatus=next(
                status
                for status in reversed(error.operation.status)
                if status.event == OperationResourceEventEnum.FINISHED
            ),
        )
    return _output_after_sampling(walk, discoverySpace, parameters, target)
