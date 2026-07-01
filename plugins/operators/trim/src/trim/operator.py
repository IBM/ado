# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT


import logging

from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import FunctionOperationInfo
from orchestrator.core.operation.operation import OperationOutput
from orchestrator.modules.operators.collections import characterize_operation
from trim.samplers.no_priors_utils import get_source_and_target
from trim.trim_pydantic import (
    TrimParameters,
)  # Importing this way works when the package is installed
from trim.utils.logging_utils import (
    log_and_save_characterization,
)

logger_trim = logging.getLogger(__name__)


def _resolve_target_output(
    params: "TrimParameters",
    discoverySpace: DiscoverySpace,
) -> "TrimParameters":
    """Resolve and validate ``params.targetOutput`` against the measurement space.

    ``targetOutput`` must ultimately be the *observed property identifier* —
    the fully-qualified ``"{experimentId}-{targetPropertyId}"`` string — because
    ``get_source_and_target`` uses ``property_type="observed"`` which keys
    columns by that identifier.

    As a convenience, if the user supplied the bare *target property identifier*
    (e.g. ``"pressure"`` instead of ``"calculate_pressure_ideal_gas-pressure"``)
    **and** exactly one experiment in the space produces that target, this
    function silently rewrites both ``params.targetOutput`` and
    ``params.noPriorParameters.targetOutput`` to the full form.

    Args:
        params: Validated ``TrimParameters`` as parsed from kwargs.
        discoverySpace: The discovery space being characterised.

    Returns:
        ``params`` with ``targetOutput`` (and the nested copy in
        ``noPriorParameters``) set to the fully-qualified observed property
        identifier.

    Raises:
        ValueError: When ``targetOutput`` is a bare target property identifier
            that matches zero or more than one observed property in the space.
    """
    observed_properties = discoverySpace.measurementSpace.observedProperties

    # Already a fully-qualified observed property identifier — validate it exists.
    if params.targetOutput in {op.identifier for op in observed_properties}:
        return params

    # Try to resolve as a bare target property identifier.
    matches = [
        op
        for op in observed_properties
        if op.targetProperty.identifier == params.targetOutput
    ]

    if len(matches) == 1:
        resolved = matches[0].identifier
        logger_trim.info(
            f"targetOutput '{params.targetOutput}' resolved to observed property "
            f"identifier '{resolved}'."
        )
        params.targetOutput = resolved
        params.noPriorParameters.targetOutput = resolved
        return params

    if len(matches) == 0:
        valid = sorted({op.identifier for op in observed_properties})
        raise ValueError(
            f"targetOutput '{params.targetOutput}' does not match any observed "
            f"property in the measurement space. "
            f"Valid observed property identifiers are: {valid}"
        )

    # len(matches) > 1: ambiguous — multiple experiments produce the same target.
    candidates = sorted(op.identifier for op in matches)
    raise ValueError(
        f"targetOutput '{params.targetOutput}' is ambiguous: multiple experiments "
        f"in the measurement space produce this target property. "
        f"Specify the fully-qualified observed property identifier instead. "
        f"Candidates: {candidates}"
    )


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
    version="2.0.0",
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
    from orchestrator.modules.operators.collections import explore
    from orchestrator.modules.operators.randomwalk import (
        CustomSamplerConfiguration,
        RandomWalkParameters,
        SamplerModuleConf,
    )

    random_walk = explore.operators["random_walk"].function

    params = TrimParameters.model_validate(kwargs)

    if params.no_priors_operation_id is not None:
        raise ValueError(
            "The 'no_priors_operation_id' field of TrimParameters is set "
            "automatically by the TRIM operator and must not be configured by the user."
        )

    params = _resolve_target_output(params, discoverySpace)

    logger_trim.info(
        "Transfer Refined Iterative Modeling starts."
        f"Target variable = {params.targetOutput}"
    )
    logger_trim.info(f"Parameters are {params}")

    # Checks if the source space has been already characterized appropriately
    source_df, target_df = get_source_and_target(
        discoverySpace, params.targetOutput, log_string="First query"
    )

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
        # Pass the full unsampled pool to random_walk so it never cuts the
        # iterator short.  The NoPriorsSampleSelector's own quota_count guard
        # stops iterating once enough hits are collected regardless of mode.
        no_priors_rwparams = RandomWalkParameters(
            samplerConfig=no_priors_sampler_config,
            batchSize=1,
            numberEntities="all",
            singleMeasurement=True,
        )

        op_output_characterization_no_prior: OperationOutput = random_walk(
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

        params.no_priors_operation_id = (
            op_output_characterization_no_prior.operation.identifier
        )

        if logger_trim.isEnabledFor(logging.DEBUG):
            source_df, target_df = get_source_and_target(
                discoverySpace, params.targetOutput
            )
            log_and_save_characterization(source_df, target_df)

    # TRIM Iterative Modeling
    trim_module = SamplerModuleConf(
        moduleClass="TrimSampleSelector",  # this is the name of our custom sampler class -> which I guess is CustomSequentialSampleSelector
        moduleName="trim.trim_sampler",  ### If CustomSequentialSampleSelector is imported as "from trim.trim_sampler import TrimSampleSelector" then this is correct
    )
    trim_sampler_config = CustomSamplerConfiguration(
        module=trim_module, parameters=params
    )

    # Pass the full unsampled pool to random_walk so it never cuts the
    # iterator short. Similar to NoPriorsSampleSelector the TrimSampleSelector quota
    # stops iterating once enough hits are collected regardless of mode.
    trim_rwparams = RandomWalkParameters(
        samplerConfig=trim_sampler_config,
        batchSize=1,
        numberEntities="all",
        singleMeasurement=True,
    )

    op_output_iterative_modeling: OperationOutput = random_walk(
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
