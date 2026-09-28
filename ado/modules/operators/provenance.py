# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Builds the package provenance recorded on operation resources."""

from ado.core.metadata import PackageProvenance
from ado.core.operation.config import (
    DiscoveryOperationEnum,
    OperatorModuleConf,
    OperatorReference,
)
from ado.core.operation.resource import OperationProvenanceInfo
from ado.modules.actuators.errors import (
    DeprecatedExperimentError,
    MissingActuatorConfigurationForCatalogError,
    UnexpectedCatalogRetrievalError,
    UnknownActuatorError,
    UnknownExperimentError,
)
from ado.modules.actuators.registry import ActuatorRegistry
from ado.modules.operators.collections import operationCollectionMap
from ado.schema.measurementspace import MeasurementSpace


def provenance_for_operator(
    name: str, op_type: DiscoveryOperationEnum
) -> PackageProvenance | None:
    """Return the package provenance for a registered operator.

    Looks up the operator in the collection for ``op_type`` and returns the
    :class:`~ado.core.metadata.PackageProvenance` recorded on its
    registry metadata at registration time.

    Args:
        name: Canonical operator name.
        op_type: The discovery operation type the operator belongs to.

    Returns:
        A :class:`~ado.core.metadata.PackageProvenance` instance,
        or ``None`` if provenance is unavailable.
    """
    collection = operationCollectionMap.get(op_type)
    if collection is None:
        return None
    metadata = collection.operators.get(name)
    if metadata is None:
        return None
    return metadata.provenance


def operation_provenance(
    operator_module: OperatorModuleConf | OperatorReference,
) -> OperationProvenanceInfo:
    """Return the provenance for a general operation

    Args:
        operator_module: The operator the operation runs. Provenance is only
            available for a registered :class:`OperatorReference`.

    Returns:
        An :class:`~ado.core.operation.resource.OperationProvenanceInfo` whose
        ``operators`` field maps the operator identifier to its distribution,
        and which is empty if the operator provenance cannot be resolved.
    """
    operators: dict[str, PackageProvenance] = {}
    if isinstance(operator_module, OperatorReference):
        operator_provenance = provenance_for_operator(
            operator_module.operatorName, operator_module.operationType
        )
        if operator_provenance is not None:
            operators[operator_module.operatorIdentifier] = operator_provenance

    return OperationProvenanceInfo(operators=operators)


def explore_operation_provenance(
    operator_module: OperatorModuleConf | OperatorReference,
    measurement_space: MeasurementSpace,
) -> OperationProvenanceInfo:
    """Return the provenance for an explore operation.

    Explore operations measure entities, so in addition to the operator this
    records the experiments that will satisfy the measurement space and the
    packages providing the actuators that will execute them.

    Experiments that cannot be resolved from the registry are skipped rather
    than raising: an unresolvable experiment has no provenance to record, and
    the operation itself will fail later with a more specific error.

    Args:
        operator_module: The operator the operation runs.
        measurement_space: The measurement space the operation will explore.

    Returns:
        An :class:`~ado.core.operation.resource.OperationProvenanceInfo` with
        the operators, experiments, and actuators for the operation.
    """
    provenance = operation_provenance(operator_module)

    registry = ActuatorRegistry.globalRegistry()
    for space_experiment in measurement_space.experiments:
        try:
            catalog_experiment = registry.experimentForReference(
                space_experiment.reference, resolve=True
            )
        except (
            UnknownExperimentError,
            UnknownActuatorError,
            DeprecatedExperimentError,
            UnexpectedCatalogRetrievalError,
            MissingActuatorConfigurationForCatalogError,
        ):
            continue

        provenance.experiments.append(catalog_experiment.reference)
        actuator_id = catalog_experiment.actuatorIdentifier
        if actuator_id not in provenance.actuators:
            actuator_provenance = registry.provenance_for_actuator(actuator_id)
            if actuator_provenance is not None:
                provenance.actuators[actuator_id] = actuator_provenance

    return provenance
