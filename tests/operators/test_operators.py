# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import itertools
import re
import typing

import pydantic
import pytest
from ado_ray_tune.operator import RayTune

import orchestrator.core.operation.config
import orchestrator.core.operation.operation
import orchestrator.modules.module
import orchestrator.modules.operators.base
import orchestrator.modules.operators.collections
from orchestrator.core.discoveryspace.samplers import (
    ExplicitEntitySpaceGridSampleGenerator,
    RandomSampleSelector,
    SequentialSampleSelector,
    WalkModeEnum,
)
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.resource import (
    DiscoveryOperationResourceConfiguration,
    OperationExitStateEnum,
    OperationResourceEventEnum,
)
from orchestrator.core.resources import (
    ADOResourceEventEnum,
    CoreResourceKinds,
)
from orchestrator.modules.operators.randomwalk import (
    BaseSamplerConfiguration,
    CustomSamplerConfiguration,
    RandomWalk,
    RandomWalkParameters,
    SamplerModuleConf,
    random_walk,
)


def test_randomwalk_class_methods() -> None:

    import orchestrator.metastore.project

    assert (
        RandomWalk.operationType()
        == orchestrator.core.operation.config.DiscoveryOperationEnum.SEARCH
    )
    assert RandomWalk.operatorIdentifier().split("-")[0] == "randomwalk"


def test_raytune_class_methods() -> None:
    import orchestrator.metastore.project

    assert (
        RayTune.operationType()
        == orchestrator.core.operation.config.DiscoveryOperationEnum.SEARCH
    )
    assert RayTune.operatorIdentifier().split("-")[0] == "raytune"


def test_operator_function_conf() -> None:

    function = orchestrator.core.operation.config.OperatorFunctionConf(
        operationType=orchestrator.core.operation.config.DiscoveryOperationEnum.MODIFY,
        operatorName="rifferla",
    )

    assert function.operationFunction()
    assert (
        function.operationType
        == orchestrator.core.operation.config.DiscoveryOperationEnum.MODIFY
    )
    assert function.validateOperatorExists()
    assert function.operatorName == "rifferla"
    assert function.operatorIdentifier.split("-")[0] == "rifferla"


def test_operator_module_conf(
    operator_module_conf: orchestrator.core.operation.config.OperatorModuleConf,
) -> None:

    assert (
        operator_module_conf.operationType
        == orchestrator.core.operation.config.DiscoveryOperationEnum.SEARCH
    )
    assert (
        operator_module_conf.operatorIdentifier.split("-")[0]
        == operator_module_conf.moduleClass.lower()
    )


def test_operator_module_conf_random_walk() -> None:

    module = orchestrator.core.operation.config.OperatorModuleConf(
        moduleName="orchestrator.modules.operators.randomwalk",
        moduleClass="RandomWalk",
    )

    assert module.operatorIdentifier
    assert isinstance(module.operatorIdentifier, str)
    assert (
        module.operationType
        == orchestrator.core.operation.config.DiscoveryOperationEnum.SEARCH
    )
    assert module.operatorIdentifier.split("-")[0] == "randomwalk"


def test_characterize(expected_characterize_operators: list[str]) -> None:

    assert len(
        orchestrator.modules.operators.collections.characterize.list_operations()
    ) == len(expected_characterize_operators)

    for operation in expected_characterize_operators:
        assert (
            operation
            in orchestrator.modules.operators.collections.characterize.list_operations()
        )
        assert orchestrator.modules.operators.collections.characterize.__getattr__(
            operation
        )


def test_explore(expected_explore_operators: list[str]) -> None:

    assert len(
        orchestrator.modules.operators.collections.explore.list_operations()
    ) == len(expected_explore_operators)

    for operation in expected_explore_operators:
        assert (
            operation
            in orchestrator.modules.operators.collections.explore.list_operations()
        )
        assert orchestrator.modules.operators.collections.explore.__getattr__(operation)


def test_characterize_operator_function_configurations(
    expected_characterize_operators: list[str],
) -> None:

    for operationName in expected_characterize_operators:
        operationConf = orchestrator.core.operation.config.OperatorFunctionConf(
            operatorName=operationName,
            operationType=orchestrator.core.operation.config.DiscoveryOperationEnum.CHARACTERIZE,
        )
        assert operationConf is not None


def test_explore_operator_function_configurations(
    expected_explore_operators: list[str],
) -> None:

    for operationName in expected_explore_operators:
        operationConf = orchestrator.core.operation.config.OperatorFunctionConf(
            operatorName=operationName,
            operationType=orchestrator.core.operation.config.DiscoveryOperationEnum.SEARCH,
        )
        assert operationConf is not None


def test_operator_function_configuration_incorrect_type(
    expected_explore_operators: list[str],
) -> None:

    operation_type = (
        orchestrator.core.operation.config.DiscoveryOperationEnum.CHARACTERIZE
    )

    for operator_name in expected_explore_operators:
        operationConf = orchestrator.core.operation.config.OperatorFunctionConf(
            operatorName=operator_name,
            operationType=operation_type,
        )

        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Operator {operator_name} had no functions of type {operation_type}"
            ),
        ):
            operationConf.validateOperatorExists()


def test_operator_function_configuration_unknown_function() -> None:

    operator_name = "UnknownOperationName"
    operation_type = (
        orchestrator.core.operation.config.DiscoveryOperationEnum.CHARACTERIZE
    )

    operationConf = orchestrator.core.operation.config.OperatorFunctionConf(
        operatorName="UnknownOperationName",
        operationType=orchestrator.core.operation.config.DiscoveryOperationEnum.CHARACTERIZE,
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Operator {operator_name} had no functions of type {operation_type}"
        ),
    ):
        operationConf.validateOperatorExists()


def test_operator_function_configuration_unknown_type() -> None:

    operator_name = "raytune"
    operation_type = orchestrator.core.operation.config.DiscoveryOperationEnum.STUDY

    operationConf = orchestrator.core.operation.config.OperatorFunctionConf(
        operatorName=operator_name,
        operationType=operation_type,
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Operator {operator_name} had no functions of type {operation_type}"
        ),
    ):
        operationConf.validateOperatorExists()


def test_random_walk_operation_configuration() -> None:

    from orchestrator.modules.operators.randomwalk import (
        RandomWalk,
        RandomWalkParameters,
    )

    assert random_walk
    assert (
        orchestrator.modules.operators.collections.explore.configuration_model_for_operation(
            "random_walk"
        )
        == RandomWalkParameters
    )
    assert (
        orchestrator.modules.operators.collections.explore.default_configuration_model_for_operation(
            "random_walk"
        )
        == RandomWalk.defaultOperationParameters()
    )


def test_raytune_operation_configuration(
    raytuneConf: DiscoveryOperationResourceConfiguration,
) -> None:

    import ado_ray_tune.operator_function
    from ado_ray_tune.operator import (
        RayTune,
        RayTuneConfiguration,
    )

    assert ado_ray_tune.operator_function.ray_tune
    assert (
        orchestrator.modules.operators.collections.explore.configuration_model_for_operation(
            "ray_tune"
        )
        == RayTuneConfiguration
    )
    assert (
        orchestrator.modules.operators.collections.explore.default_configuration_model_for_operation(
            "ray_tune"
        )
        == RayTune.defaultOperationParameters()
    )


# all - returns a config that uses "all" for numberEntities
# value - returns a config with a value for numberEntities


# TODO: Add a test for all with unbounded space
# This requires creating an alternate discoverySpace for `test_random_walk_fail_invalid_config"
#


def test_random_walk_config(
    randomWalkConf: DiscoveryOperationResourceConfiguration,
) -> None:
    """Test random walk configuration model"""

    import pydantic

    assert randomWalkConf is not None
    assert RandomWalk.validateOperationParameters(
        parameters=randomWalkConf.operation.parameters
    )

    parameters_model: RandomWalkParameters = RandomWalk.validateOperationParameters(
        parameters=randomWalkConf.operation.parameters
    )

    # Test sampler
    assert isinstance(parameters_model.samplerConfig, BaseSamplerConfiguration)
    sampler = parameters_model.samplerConfig.sampler()
    assert isinstance(sampler, ExplicitEntitySpaceGridSampleGenerator)
    assert sampler.mode == WalkModeEnum.RANDOM

    # Test extra params not allowed

    parameters = randomWalkConf.operation.parameters.copy()
    parameters["foo"] = "bar"

    with pytest.raises(pydantic.ValidationError):
        RandomWalk.validateOperationParameters(parameters=parameters)

    # Test extra params not allowed

    parameters = randomWalkConf.operation.parameters.copy()
    parameters.pop("numberEntities")
    parameters["number-iterations"] = 6

    with pytest.raises(pydantic.ValidationError):
        RandomWalk.validateOperationParameters(parameters=parameters)


def test_random_walk_custom_sampler_config() -> None:

    config = CustomSamplerConfiguration(
        module=SamplerModuleConf(
            moduleClass="ExplicitEntitySpaceGridSampleGenerator",
            moduleName="orchestrator.core.discoveryspace.samplers",
        ),
        parameters=ExplicitEntitySpaceGridSampleGenerator.parameters_model()(
            mode=WalkModeEnum.RANDOM
        ),
    )

    sampler = config.sampler()
    assert isinstance(
        sampler, ExplicitEntitySpaceGridSampleGenerator
    ), "Expected the sampler to be an instance of ExplicitEntitySpaceGridSampleGenerator"
    assert (
        sampler.mode == WalkModeEnum.RANDOM
    ), "Expected the samplers mode to be RANDOM"

    dump = config.model_dump()

    # Check deserialization
    new_config = CustomSamplerConfiguration.model_validate(dump)
    sampler = new_config.sampler()
    assert isinstance(
        sampler, ExplicitEntitySpaceGridSampleGenerator
    ), "Expected the sampler to be an instance of ExplicitEntitySpaceGridSampleGenerator"
    assert (
        sampler.mode == WalkModeEnum.RANDOM
    ), "Expected the samplers mode to be RANDOM"

    # Check validation
    dump["module"]["moduleClass"] = "NonExistantClass"

    with pytest.raises(pydantic.ValidationError):
        CustomSamplerConfiguration.model_validate(dump)

    dump = config.model_dump()
    dump["parameters"]["fake_param"] = 10

    with pytest.raises(pydantic.ValidationError):
        CustomSamplerConfiguration.model_validate(dump)


@pytest.mark.parametrize(
    ("mode", "samplerType"),
    list(itertools.product(WalkModeEnum, ["generator", "selector"])),
)
def test_random_walk_base_sampler_config(
    mode: WalkModeEnum, samplerType: typing.Literal["generator", "selector"]
) -> None:
    config = BaseSamplerConfiguration(mode=mode.value, samplerType=samplerType)

    sampler = config.sampler()

    if samplerType == "generator":
        assert isinstance(
            sampler, ExplicitEntitySpaceGridSampleGenerator
        ), "Expected the sampler to be an instance of ExplicitEntitySpaceGridSampleGenerator"
        assert sampler.mode == mode
    else:
        if mode == WalkModeEnum.RANDOM:
            assert isinstance(sampler, RandomSampleSelector)
        elif mode == WalkModeEnum.SEQUENTIAL:
            assert isinstance(sampler, SequentialSampleSelector)


def test_ray_tune_config(
    raytuneConf: DiscoveryOperationResourceConfiguration,
) -> None:
    """Test running a random_walk operation via the operation functions"""

    import pydantic

    assert raytuneConf is not None
    assert RayTune.validateOperationParameters(
        parameters=raytuneConf.operation.parameters
    )

    parameters = raytuneConf.operation.parameters.copy()
    parameters["foo"] = "bar"

    with pytest.raises(pydantic.ValidationError):
        RandomWalk.validateOperationParameters(parameters=parameters)


def test_run_random_walk_operation(
    ml_multi_cloud_space: DiscoverySpace,
    randomWalkConf: DiscoveryOperationResourceConfiguration,
) -> None:
    """Test running a random_walk operation via the operation functions"""

    import orchestrator.core.resources

    discoverySpace = ml_multi_cloud_space

    assert discoverySpace is not None
    assert randomWalkConf is not None
    randomWalkConf.spaces[0] = ml_multi_cloud_space.uri
    assert RandomWalk.validateOperationParameters(
        parameters=randomWalkConf.operation.parameters
    )

    operationOutput = random_walk(discoverySpace, **randomWalkConf.operation.parameters)

    assert isinstance(
        operationOutput, orchestrator.core.operation.operation.OperationOutput
    )
    assert operationOutput.operation
    # We expect the operationOutput to have an exit status - we know random uses default so it should be SUCCESS
    assert operationOutput.exitStatus.exit_state == OperationExitStateEnum.SUCCESS
    assert operationOutput.exitStatus.event == OperationResourceEventEnum.FINISHED

    # We expect the most recent status to have been updated
    assert operationOutput.operation.status[-1].event == ADOResourceEventEnum.UPDATED
    # We expect the wrapper to have added the exitStatus to the second last status
    assert operationOutput.operation.status[-2] == operationOutput.exitStatus

    # Check the operation is in the metastore
    operation = discoverySpace.metadataStore.getResource(
        operationOutput.operation.identifier, kind=CoreResourceKinds.OPERATION
    )

    assert operation
    # Check the operation status are as expected - CREATED, ADDED, STARTED, UPDATED, FINISHED, UPDATED
    assert operation.status[0].event == ADOResourceEventEnum.CREATED
    assert operation.status[1].event == ADOResourceEventEnum.ADDED
    assert operation.status[2].event == OperationResourceEventEnum.STARTED
    assert (
        operation.status[3].event == ADOResourceEventEnum.UPDATED
    )  # This is the UPDATED event caused by storing the START even in the DB
    assert operation.status[4].event == OperationResourceEventEnum.FINISHED
    assert operation.status[4].exit_state == OperationExitStateEnum.SUCCESS
    assert operation.status[5].event == ADOResourceEventEnum.UPDATED

    # Check it is related to the space
    spaces = discoverySpace.metadataStore.getRelatedResourceIdentifiers(
        identifier=operationOutput.operation.identifier,
        kind=CoreResourceKinds.DISCOVERYSPACE.value,
    )
    assert spaces.shape[0] > 0
    assert spaces.IDENTIFIER[0] == discoverySpace.uri

    ## CHECK THE EXPECTED NUMBER OF EXPERIMENTS HAVE BEEN RUN
    assert operationOutput.operation.metadata["entities_submitted"] == 48
    assert (
        operationOutput.operation.metadata["experiments_requested"] == 74
    )  # There are multiple measuremenst for some entities


def test_random_walk_fail_invalid_config(
    ml_multi_cloud_space: DiscoverySpace,
    invalidRandomWalkConf: DiscoveryOperationResourceConfiguration,
) -> None:
    """Test running a random_walk operation via the operation functions"""

    discoverySpace = ml_multi_cloud_space

    import orchestrator.core.resources
    import orchestrator.modules.actuators
    import orchestrator.modules.operators.base

    assert discoverySpace is not None
    assert invalidRandomWalkConf is not None

    # Note: Number of entities being greater than space size (valueGreaterThanSize) raises a ValueError
    # as it is detected at RandomWalk.run() not during configuration validation (which can't check this as it has no access to the space)
    # This is captured and raise as a OperationException
    try:
        random_walk(
            discoverySpace, **invalidRandomWalkConf.operation.parameters
        )  # type: orchestrator.modules.operators.base.OperationOutput
    except orchestrator.core.operation.operation.OperationException as error:
        operation = error.operation
        assert operation
        # Check the operation status are as expected - CREATED, ADDED, STARTED, FINISHED, UPDATED
        assert operation.status[0].event == ADOResourceEventEnum.CREATED
        assert operation.status[1].event == ADOResourceEventEnum.ADDED
        assert operation.status[2].event == OperationResourceEventEnum.STARTED
        assert operation.status[3].event == ADOResourceEventEnum.UPDATED
        assert operation.status[4].event == OperationResourceEventEnum.FINISHED
        assert operation.status[4].exit_state == OperationExitStateEnum.ERROR
        assert operation.status[5].event == ADOResourceEventEnum.UPDATED

        operation = discoverySpace.metadataStore.getResource(
            operation.identifier, kind=CoreResourceKinds.OPERATION
        )
        assert operation
        # Check the operation status are as expected - CREATED, ADDED, STARTED, UPDATED, FINISHED, UPDATED
        assert operation.status[0].event == ADOResourceEventEnum.CREATED
        assert operation.status[1].event == ADOResourceEventEnum.ADDED
        assert operation.status[2].event == OperationResourceEventEnum.STARTED
        assert operation.status[3].event == ADOResourceEventEnum.UPDATED
        assert operation.status[4].event == OperationResourceEventEnum.FINISHED
        assert operation.status[4].exit_state == OperationExitStateEnum.ERROR
        assert operation.status[5].event == ADOResourceEventEnum.UPDATED
    except ValueError:
        pass
    else:
        pytest.fail("Expected exception to be raised and none was")


def test_run_ray_tune_operation(
    ml_multi_cloud_space: DiscoverySpace,
    raytuneConf: DiscoveryOperationResourceConfiguration,
) -> None:
    """Test running a ray_tune operation via the operation functions"""

    import orchestrator.core.resources

    discoverySpace = ml_multi_cloud_space

    assert discoverySpace is not None
    assert raytuneConf is not None
    assert RayTune.validateOperationParameters(
        parameters=raytuneConf.operation.parameters
    )

    import ado_ray_tune.operator_function

    operationOutput = ado_ray_tune.operator_function.ray_tune(
        discoverySpace, **raytuneConf.operation.parameters
    )

    assert isinstance(
        operationOutput, orchestrator.core.operation.operation.OperationOutput
    )
    assert operationOutput.operation

    # We expect the operationOutput to have an exit status - we know raytune uses default so it should be SUCCESS
    assert operationOutput.exitStatus.exit_state == OperationExitStateEnum.SUCCESS
    assert operationOutput.exitStatus.event == OperationResourceEventEnum.FINISHED

    # We expect the last status registered is the update status
    assert operationOutput.operation.status[-1].event == ADOResourceEventEnum.UPDATED

    # We expect the wrapper to have added the exitStatus to the operation - it will be second last as the last update
    assert operationOutput.operation.status[-2] == operationOutput.exitStatus

    # Check the operation is in the metastore
    operation = discoverySpace.metadataStore.getResource(
        operationOutput.operation.identifier, kind=CoreResourceKinds.OPERATION
    )
    assert operation
    # Check the operation status are as expected - CREATED, ADDED, STARTED, UPDATED, FINISHED, UPDATED
    assert operation.status[0].event == ADOResourceEventEnum.CREATED
    assert operation.status[1].event == ADOResourceEventEnum.ADDED
    assert operation.status[2].event == OperationResourceEventEnum.STARTED
    assert operation.status[3].event == ADOResourceEventEnum.UPDATED
    assert operation.status[4].event == OperationResourceEventEnum.FINISHED
    assert operation.status[4].exit_state == OperationExitStateEnum.SUCCESS
    assert operation.status[5].event == ADOResourceEventEnum.UPDATED

    # Check it is related to the space
    spaces = discoverySpace.metadataStore.getRelatedResourceIdentifiers(
        identifier=operationOutput.operation.identifier,
        kind=CoreResourceKinds.DISCOVERYSPACE.value,
    )
    assert spaces.shape[0] > 0
    assert spaces.IDENTIFIER[0] == discoverySpace.uri


def test_operator_default_and_validate(
    optimizer_operator: type[RandomWalk] | type[RayTune],
) -> None:

    assert optimizer_operator
    default = optimizer_operator.defaultOperationParameters()
    parameters = default.model_dump() if not isinstance(default, dict) else default

    assert optimizer_operator.validateOperationParameters(parameters=parameters)
