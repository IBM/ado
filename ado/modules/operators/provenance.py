# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Builds the package provenance recorded on operation resources."""

from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.metadata import PackageProvenance
from ado.core.operation.config import (
    DiscoveryOperationEnum,
    OperatorMetadata,
    OperatorModuleConf,
    OperatorReference,
)
from ado.core.operation.resource import OperationProvenanceInfo
from ado.modules.operators.collections import operationCollectionMap
from ado.schema.reference import ExperimentReference


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
    """Return the provenance for a general operation.

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


def experiment_and_actuator_provenance_from_spaces(
    spaces: list[DiscoverySpace],
) -> tuple[list[ExperimentReference], dict[str, PackageProvenance]]:
    """Resolve catalog experiments and actuator packages for discovery-space inputs.

    Experiments that cannot be resolved against the actuator catalog are skipped.
    Experiment references are deduplicated by equality. Actuators are
    deduplicated by identifier.

    Args:
        spaces: Discovery spaces whose measurement spaces should be recorded.

    Returns:
        Resolved experiment references and actuator package provenance. Both
        collections are empty when *spaces* is empty.
    """
    from ado.modules.actuators.errors import (
        DeprecatedExperimentError,
        MissingActuatorConfigurationForCatalogError,
        UnexpectedCatalogRetrievalError,
        UnknownActuatorError,
        UnknownExperimentError,
    )
    from ado.modules.actuators.registry import ActuatorRegistry

    experiments: list[ExperimentReference] = []
    actuators: dict[str, PackageProvenance] = {}
    registry = ActuatorRegistry.globalRegistry()
    for space in spaces:
        for space_experiment in space.measurementSpace.experiments:
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

            reference = catalog_experiment.reference
            if reference not in experiments:
                experiments.append(reference)

            actuator_id = catalog_experiment.actuatorIdentifier
            if actuator_id not in actuators:
                actuator_provenance = registry.provenance_for_actuator(actuator_id)
                if actuator_provenance is not None:
                    actuators[actuator_id] = actuator_provenance

    return experiments, actuators


def explore_operation_provenance(
    operator_metadata: OperatorMetadata,
    spaces: list[DiscoverySpace],
) -> OperationProvenanceInfo:
    """Build provenance for an explore operation.

    Records the operator package plus the experiments and actuators that
    satisfy the discovery-space measurement spaces.

    Args:
        operator_metadata: Registered metadata for the explore operator.
        spaces: Discovery spaces whose measurement spaces should be recorded.

    Returns:
        Provenance with operator, experiment, and actuator entries. Operator
        provenance is omitted when the operator has none. Experiment and
        actuator collections are empty when *spaces* is empty.
    """
    provenance = operation_provenance(operator_metadata.reference)
    experiments, actuators = experiment_and_actuator_provenance_from_spaces(spaces)
    provenance.experiments.extend(experiments)
    provenance.actuators.update(actuators)
    return provenance
