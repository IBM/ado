# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


import logging

from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import FunctionOperationInfo
from orchestrator.core.operation.operation import OperationOutput
from orchestrator.modules.operators.collections import characterize_operation
from orchestrator.modules.operators.randomwalk import (
    CustomSamplerConfiguration,
    RandomWalkParameters,
    SamplerModuleConf,
    random_walk,
)
from trim.trim_pydantic import (
    TrimParameters,
)  # Importing this way works when the package is installed
from trim.utils.logging_utilities import (
    log_and_save_characterization,
)
from trim.utils.space_df_connector import get_source_and_target

logger_trim = logging.getLogger(__name__)
logger_trim.setLevel(logging.DEBUG)


@characterize_operation(
    name="trim",
    configuration_model=TrimParameters,  # pydantic model
    configuration_model_default=TrimParameters.defaultOperation(),  # instance of pydantic model NOTE: this does not have the discovery space
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
    **kwargs,
) -> OperationOutput:

    params = TrimParameters.model_validate(kwargs)
    logger_trim.info(
        "Transfer Refined Iterative Modeling starts."
        f"Target variable = {params.targetOutput}"
    )

    logger_trim.info(f"Parameters are {kwargs}")

    # raise ValueError

    nopriors_module = SamplerModuleConf(
        moduleClass="NoPriorsSampleSelector", moduleName="trim.no_priors_sampler"
    )

    # Checks if the source space has been already characterized appropriately
    source_df, target_df = get_source_and_target(discoverySpace, params.targetOutput)
    op_output_characterization_no_prior = OperationOutput.model_validate(
        {
            "metadata": {
                "skipping operation": f"Prior source space characterization: {len(source_df)} sample. Minimal sample size: {params.samplingBudget.minPoints }"
            }
        }
    )

    if logger_trim.isEnabledFor(logging.DEBUG):
        log_and_save_characterization(source_df, target_df, logger_trim)

    # TODO: think about a better solution for the fact that the target output may
    # not be acquired for every entity of your space given your experiment.
    max_iter = 3
    current_iter = 0
    while (
        len(source_df) < params.samplingBudget.minPoints
    ) and current_iter < max_iter:
        logger_trim.warning(
            f"""Only {len(source_df)} points in the source space.
                                Starting with no-prior characterization for {params.samplingBudget.minPoints - len(source_df)}
                                Note: Trim sampler has been called with a minimum budget of {params.samplingBudget.minPoints} points.
            """
        )

        no_priors_params = TrimParameters().noPriorParameters
        if current_iter != 0:
            # Computing number of missing samples
            missing_points = params.samplingBudget.minPoints - len(source_df)
            no_priors_params.samples = no_priors_params.samples + missing_points
            logger_trim.warning(
                f"After {current_iter} prior characterizations, we still need {missing_points} additional samples. "
                f"This is likely because the target output '{params.targetOutput}' may not be acquired for every entity "
                f"in your space given your experiment. "
                f"Another no-priors characterization will be run to sample {missing_points} additional points. "
                f"Note that the maximum number of no-priors characterization operations is {max_iter}. "
                f"No-priors characterization operation will be instantiated for n_samples = {no_priors_params.samples}."
            )

        no_priors_sampler_config = CustomSamplerConfiguration(
            module=nopriors_module, parameters=no_priors_params
        )

        no_priors_rwparams = RandomWalkParameters(
            samplerConfig=no_priors_sampler_config,
            # here you set up the rw params
            batchSize=no_priors_params.batchSize,
            numberEntities=no_priors_params.samples,
            singleMeasurement=True,
        )

        op_output_characterization_no_prior = random_walk(
            discoverySpace=discoverySpace,
            operationInfo=FunctionOperationInfo.model_validate(
                {"metadata": {"completed operation": "Characterization with no priors"}}
            ),
            **no_priors_rwparams.model_dump(),
        )

        current_iter += 1
        source_df, target_df = get_source_and_target(
            discoverySpace, params.targetOutput
        )
        op_output_characterization_no_prior = OperationOutput.model_validate(
            {
                "metadata": {
                    "skipping operation": f"Prior source space characterization: {len(source_df)} sample. Minimal sample size: {params.samplingBudget.minPoints }"
                }
            }
        )
        logger_trim.info(
            f"op_output_characterization_no_prior.operation = {op_output_characterization_no_prior.operation} "
        )
        if logger_trim.isEnabledFor(logging.DEBUG):
            logger_trim.debug(
                "Saving updated source space after no-priors characterization"
            )
            log_and_save_characterization(source_df, target_df, logger_trim)

    # Continuing with TRIM Iterative Modeling
    trim_module = SamplerModuleConf(
        moduleClass="TrimSampleSelector",  # this is the name of our custom sampler class -> which I guess is CustomSequentialSampleSelector
        moduleName="trim.trim_sampler",  ### If CustomSequentialSampleSelector is imported as "from trim.trim_sampler import TrimSampleSelector" then this is correct
    )
    trim_sampler_config = CustomSamplerConfiguration(
        module=trim_module, parameters=params
    )
    trim_rwparams = RandomWalkParameters(
        samplerConfig=trim_sampler_config,
        # here you set up the rw params
        batchSize=(
            params.batchSize
            if isinstance(params.batchSize, int)
            else params.iterationSize
        ),
        numberEntities=params.samplingBudget.maxPoints,
        singleMeasurement=True,
    )

    op_output_iterative_modeling = random_walk(
        discoverySpace=discoverySpace,
        operationInfo=FunctionOperationInfo.model_validate(
            {"metadata": {"completed operation": "Iterative Modeling Operation"}}
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
        )

    return OperationOutput(other=[], resources=[op_output_iterative_modeling.operation])
