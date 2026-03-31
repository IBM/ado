# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import logging

from no_priors_characterization.no_priors_pydantic import NoPriorsParameters
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import FunctionOperationInfo
from orchestrator.core.operation.operation import OperationOutput
from orchestrator.modules.operators.collections import characterize_operation

logger = logging.getLogger(__name__)


@characterize_operation(
    name="no_priors_characterization",
    configuration_model=NoPriorsParameters,
    configuration_model_default=NoPriorsParameters(targetOutput="default_target"),
    description="""
                No-priors characterization samples a discovery space using high-dimensional
                sampling strategies (random, CLHS, Sobol, etc.) without relying on prior
                model knowledge or feature importance. This operator is useful for initial
                exploration of discovery spaces when no training data exists yet.
                """,
)
def no_priors_characterization(
    discoverySpace: DiscoverySpace = None,  # type: ignore[name-defined]
    operationInfo: FunctionOperationInfo | None = None,
    **kwargs: object,
) -> OperationOutput:
    """
    Execute no-priors characterization on a discovery space.

    Samples entities using high-dimensional sampling strategies without requiring
    prior model training or feature importance information. Useful for initial
    characterization when no measured data exists.

    Args:
        discoverySpace: The discovery space to characterize
        operationInfo: Optional operation metadata
        **kwargs: Additional parameters validated against NoPriorsParameters model

    Returns:
        OperationOutput containing the operation resources and metadata
    """
    # Lazy import to avoid circular import issues during plugin loading
    from orchestrator.modules.operators.randomwalk import (
        CustomSamplerConfiguration,
        RandomWalkParameters,
        SamplerModuleConf,
        random_walk,
    )

    params = NoPriorsParameters.model_validate(kwargs)
    logger.info(
        f"No-priors characterization starts. Target variable = {params.targetOutput}"
    )
    logger.info(f"Parameters: {params}")

    # Configure the no-priors sampler
    no_priors_module = SamplerModuleConf(
        moduleClass="NoPriorsSampleSelector",
        moduleName="no_priors_characterization.no_priors_sampler",
    )

    no_priors_sampler_config = CustomSamplerConfiguration(
        module=no_priors_module, parameters=params
    )

    no_priors_random_walk_params = RandomWalkParameters(
        samplerConfig=no_priors_sampler_config,
        batchSize=params.batchSize,
        numberEntities=params.samples,
        singleMeasurement=True,
    )

    # Execute the random walk with the no-priors sampler
    from orchestrator.core.metadata import ConfigurationMetadata

    # Create metadata with custom fields for tracking no-priors parameters
    metadata = ConfigurationMetadata(
        name="No-priors characterization",
        description=f"No-priors characterization using {params.sampling_strategy} strategy with {params.samples} samples",
    )
    # Add custom fields using extra="allow" in ConfigurationMetadata
    metadata.sampling_strategy = params.sampling_strategy  # type: ignore[attr-defined]
    metadata.samples = params.samples  # type: ignore[attr-defined]

    updated_operation_info = FunctionOperationInfo(
        metadata=metadata,
        actuatorConfigurationIdentifiers=(
            operationInfo.actuatorConfigurationIdentifiers if operationInfo else []
        ),
    )

    op_output = random_walk(
        discoverySpace=discoverySpace,
        operationInfo=updated_operation_info,
        **no_priors_random_walk_params.model_dump(),
    )

    logger.info("No-priors characterization completed")

    return op_output
