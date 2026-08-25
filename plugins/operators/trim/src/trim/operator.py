# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT


import logging

from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.config import FunctionOperationInfo
from ado.core.operation.operation import OperationOutput
from ado.modules.operators.collections import characterize_operation
from trim.samplers.no_priors_utils import get_source_and_target
from trim.trim_pydantic import (
    TrimParameters,
    TrimSamplerParameters,
)  # Importing this way works when the package is installed
from trim.utils.logging_utils import (
    log_and_save_characterization,
    log_unable_to_proceed_with_iterative_modeling_and_raise_error,
)

logger_trim = logging.getLogger(__name__)


@characterize_operation(
    name="trim",
    configuration_model=TrimParameters,
    example_configuration=TrimParameters.example_configuration(),
    description="""
                Trim is used to characterise a Discovery space.
                In its first implementation it starts from a space,
                Retrieves all measured entities from the entity source and samples the others following a certain order.
                If the number of measured entity is too small, Trim instantiates a no-priors characterization operation.
                """,
    version="2.0.5",
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
    from ado.modules.operators.collections import explore
    from ado.modules.operators.randomwalk import (
        CustomSamplerConfiguration,
        RandomWalkParameters,
        SamplerModuleConf,
    )

    random_walk = explore.operators["random_walk"].function
    if random_walk is None:
        raise RuntimeError("The random_walk operator has no registered function")

    params = TrimParameters.model_validate(kwargs)

    if params.noPriorParameters.batchSize != 1:
        raise ValueError(
            f"TRIM requires batchSize=1 for the no-priors sampler, got {params.noPriorParameters.batchSize}"
        )

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
                "skipping operation": f"Prior source space characterization: {len(source_df)} sample. Minimal sample size: {params.samplingBudget.minPoints}"
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

        # Use random-walk with no-priors sampler instead of direct operator call
        no_priors_module = SamplerModuleConf(
            moduleClass="NoPriorsSampleSelector",
            moduleName="trim.samplers.no_priors_sampler",
        )
        no_priors_sampler_config = CustomSamplerConfiguration(
            module=no_priors_module,
            parameters=params.noPriorParameters,
        )
        no_priors_rwparams = RandomWalkParameters(
            samplerConfig=no_priors_sampler_config,
            batchSize=params.noPriorParameters.batchSize,
            numberEntities=params.samplingBudget.minPoints - len(source_df),
            singleMeasurement=True,
        )

        op_output_characterization_no_prior = random_walk(
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
            **no_priors_rwparams.model_dump(),
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

    # maxPoints is a shared new-sample budget across no-priors and iterative phases
    numberEntities_iterative_modeling = params.samplingBudget.maxPoints - (
        len(source_df) - initial_source_space_size
    )
    if numberEntities_iterative_modeling <= 0:
        logger_trim.warning(
            "No sampling budget remains for iterative modeling; skipping it."
        )
        resources = (
            [op_output_characterization_no_prior.operation]
            if op_output_characterization_no_prior.operation is not None
            else []
        )
        return OperationOutput(
            other=[],
            resources=resources,
            metadata={},
        )

    # TRIM Iterative Modeling
    trim_module = SamplerModuleConf(
        moduleClass="TrimSampleSelector",  # this is the name of our custom sampler class -> which I guess is CustomSequentialSampleSelector
        moduleName="trim.trim_sampler",  ### If CustomSequentialSampleSelector is imported as "from trim.trim_sampler import TrimSampleSelector" then this is correct
    )
    # TrimSamplerParameters extends TrimParameters with numberEntitiesIterativeModeling
    # which tells the sampler exactly how many entities RandomWalk will draw. The sampler
    # uses this to call finalize_model() after yielding the last entity, since RandomWalk
    # stops calling anext() once the budget is met and never exhausts the generator.
    # This field must NOT live on TrimParameters itself because that model is serialised
    # to YAML as the operation configuration.
    sampler_params = TrimSamplerParameters(
        **params.model_dump(),
        numberEntitiesIterativeModeling=numberEntities_iterative_modeling,
    )
    trim_sampler_config = CustomSamplerConfiguration(
        module=trim_module, parameters=sampler_params
    )
    trim_rwparams = RandomWalkParameters(
        samplerConfig=trim_sampler_config,
        batchSize=1,
        # VV: Configuring RandomWalk to request one additional Entity, this enables the
        # TrimSampler to know that it's about to run out of entities and thus it should
        # finalize the model.
        numberEntities=numberEntities_iterative_modeling + 1,
        singleMeasurement=True,
    )

    op_output_iterative_modeling = random_walk(
        discoverySpace=discoverySpace,
        operationInfo=FunctionOperationInfo.model_validate(
            {
                "metadata": {"completed operation": "Iterative Modeling Operation"},
                "actuatorConfigurationIdentifiers": (
                    operationInfo.actuatorConfigurationIdentifiers
                    if operationInfo
                    else []
                ),
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
