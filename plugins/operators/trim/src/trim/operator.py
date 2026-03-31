# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT


import logging

from no_priors_characterization.utils import get_source_and_target

from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import FunctionOperationInfo
from orchestrator.core.operation.operation import OperationOutput
from orchestrator.modules.operators.collections import characterize_operation
from trim.trim_pydantic import (
    TrimParameters,
)  # Importing this way works when the package is installed
from trim.utils.logging_utils import (
    log_and_save_characterization,
    log_unable_to_proceed_with_iterative_modeling_and_raise_error,
)

logger_trim = logging.getLogger(__name__)


@characterize_operation(
    name="trim",
    configuration_model=TrimParameters,
    configuration_model_default=TrimParameters.defaultOperationParameters(),
    description="""
                Trim is used to characterise a Discovery space.
                In its first implementation it starts from a space,
                Retrieves all measured entities from the entity source and samples the others following a certain order.
                If the number of measured entity is too small, Trim instantiates a no-priors characterization operation.
                """,
)
def trim(
    discoverySpace: DiscoverySpace = None,  # type: ignore[name-defined]
    operationInfo: FunctionOperationInfo | None = None,
    **kwargs: object,
) -> OperationOutput:
    """
    Execute the TRIM (Transfer Refined Iterative Modeling) operation on a discovery space.

    TRIM characterizes a discovery space by first ensuring sufficient measured entities exist,
    then performing iterative modeling to sample additional entities in an informed order.
    If insufficient data exists, it runs a no-priors characterization first.

    Args:
        discoverySpace: The discovery space to characterize
        operationInfo: Optional operation metadata
        **kwargs: Additional parameters validated against TrimParameters model

    Returns:
        OperationOutput containing the operation resources and metadata
    """
    # Lazy import to avoid circular import issues during plugin loading
    from orchestrator.modules.operators.collections import characterize
    from orchestrator.modules.operators.randomwalk import (
        CustomSamplerConfiguration,
        RandomWalkParameters,
        SamplerModuleConf,
        random_walk,
    )

    params = TrimParameters.model_validate(kwargs)
    logger_trim.info(
        "Transfer Refined Iterative Modeling starts."
        f"Target variable = {params.targetOutput}"
    )
    logger_trim.info(f"Parameters are {params}")

    # Checks if the source space has been already characterized appropriately
    source_df, target_df = get_source_and_target(
        discoverySpace, params.targetOutput, log_string="First query"
    )
    initial_source_space_size = len(source_df)

    op_output_characterization_no_prior = OperationOutput.model_validate(
        {
            "metadata": {
                "skipping operation": f"Prior source space characterization: {len(source_df)} sample. Minimal sample size: {params.samplingBudget.minPoints }"
            }
        }
    )

    if logger_trim.isEnabledFor(logging.DEBUG):
        log_and_save_characterization(source_df, target_df)

    if len(source_df) < params.samplingBudget.minPoints:
        logger_trim.warning(
            f"Only {len(source_df)} points in the source space.\n"
            "Starting with no-prior characterization operation, "
            f"it will sample {params.samplingBudget.minPoints - len(source_df)} points.\n"
            f"Note: Trim sampler has been called with a minimum budget of {params.samplingBudget.minPoints} points."
        )

        # Call the no-priors-characterization operator directly
        no_priors_operator = characterize.no_priors_characterization
        op_output_characterization_no_prior = no_priors_operator(
            discoverySpace=discoverySpace,
            operationInfo=FunctionOperationInfo.model_validate(
                {
                    "metadata": {
                        "completed operation": "Characterization with no priors",
                        "summary of collected data": f"No-priors characterization will sample {params.samplingBudget.minPoints - len(source_df)} points with the required property {params.targetOutput}. Minimal sample size: {params.samplingBudget.minPoints}",
                    },
                    "actuatorConfigurationIdentifiers": (
                        operationInfo.actuatorConfigurationIdentifiers
                        if operationInfo
                        else []
                    ),
                }
            ),
            **params.noPriorParameters.model_dump(),
        )

        source_df, target_df = get_source_and_target(
            discoverySpace, params.targetOutput
        )

        if logger_trim.isEnabledFor(logging.DEBUG):
            logger_trim.debug(
                "Saving updated source space after no-priors characterization"
            )
            log_and_save_characterization(source_df, target_df)

        if len(source_df) < params.samplingBudget.minPoints:
            log_unable_to_proceed_with_iterative_modeling_and_raise_error(
                discoverySpace,
                target_output=params.targetOutput,
                additional_info=f"This was detected during the no-priors characterization phase: {params.samplingBudget.minPoints - len(source_df)} out of {params.samplingBudget.minPoints}.",
            )

    # TRIM Iterative Modeling
    trim_module = SamplerModuleConf(
        moduleClass="TrimSampleSelector",  # this is the name of our custom sampler class -> which I guess is CustomSequentialSampleSelector
        moduleName="trim.trim_sampler",  ### If CustomSequentialSampleSelector is imported as "from trim.trim_sampler import TrimSampleSelector" then this is correct
    )
    trim_sampler_config = CustomSamplerConfiguration(
        module=trim_module, parameters=params
    )
    numberEntities_iterative_modeling = (
        len(source_df) - initial_source_space_size
        if op_output_characterization_no_prior.operation
        else params.samplingBudget.maxPoints
    )
    trim_rwparams = RandomWalkParameters(
        samplerConfig=trim_sampler_config,
        batchSize=1,
        numberEntities=numberEntities_iterative_modeling,
        singleMeasurement=True,
    )

    op_output_iterative_modeling = random_walk(
        discoverySpace=discoverySpace,
        operationInfo=FunctionOperationInfo.model_validate(
            {
                "metadata": {"completed operation": "Iterative Modeling Operation"},
                "actuatorConfigurationIdentifiers": operationInfo.actuatorConfigurationIdentifiers,
            }
        ),
        **trim_rwparams.model_dump(),
    )

    logger_trim.info(
        f"op_output_iterative_modeling.operation = {op_output_iterative_modeling.operation} "
    )

    if op_output_characterization_no_prior.operation:
        return OperationOutput(
            other=[],
            resources=[
                op_output_characterization_no_prior.operation,
                op_output_iterative_modeling.operation,
            ],
            metadata={},
        )

    return OperationOutput(
        other=[], resources=[op_output_iterative_modeling.operation], metadata={}
    )
