# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
import itertools
import logging
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
)


def test_operator_function_conf() -> None:

    function = orchestrator.core.operation.config.OperatorReference(
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
    assert function.operatorIdentifier.split("@")[0] == "rifferla"


def test_operator_module_conf(
    operator_module_conf: orchestrator.core.operation.config.OperatorModuleConf,
) -> None:

    from orchestrator.modules.module import load_module_class_or_function

    assert (
        operator_module_conf.operationType
        == orchestrator.core.operation.config.DiscoveryOperationEnum.SEARCH
    )
    cls = load_module_class_or_function(operator_module_conf)
    expected_name = cls.operator_metadata().name
    assert operator_module_conf.operatorIdentifier.startswith(f"{expected_name}@")


def test_characterize(expected_characterize_operators: list[str]) -> None:

    assert len(
        orchestrator.modules.operators.collections.characterize.list_operators()
    ) == len(expected_characterize_operators)

    for operation in expected_characterize_operators:
        assert (
            operation
            in orchestrator.modules.operators.collections.characterize.list_operators()
        )
        assert orchestrator.modules.operators.collections.characterize.__getattr__(
            operation
        )


def test_explore(expected_explore_operators: list[str]) -> None:

    assert len(
        orchestrator.modules.operators.collections.explore.list_operators()
    ) == len(expected_explore_operators)

    for operation in expected_explore_operators:
        assert (
            operation
            in orchestrator.modules.operators.collections.explore.list_operators()
        )
        assert orchestrator.modules.operators.collections.explore.__getattr__(operation)


def test_characterize_operator_function_configurations(
    expected_characterize_operators: list[str],
) -> None:

    for operationName in expected_characterize_operators:
        operationConf = orchestrator.core.operation.config.OperatorReference(
            operatorName=operationName,
            operationType=orchestrator.core.operation.config.DiscoveryOperationEnum.CHARACTERIZE,
        )
        assert operationConf is not None


def test_explore_operator_function_configurations(
    expected_explore_operators: list[str],
) -> None:

    for operationName in expected_explore_operators:
        operationConf = orchestrator.core.operation.config.OperatorReference(
            operatorName=operationName,
            operationType=orchestrator.core.operation.config.DiscoveryOperationEnum.SEARCH,
        )
        assert operationConf is not None
        assert operationConf.validateOperatorExists()
        assert operationConf.operatorName == operationName
        # operatorIdentifier must be <registeredName>@<version>, matching ado get operators
        assert operationConf.operatorIdentifier.startswith(f"{operationName}@")


def test_explore_operator_class_registration(
    expected_explore_operators: list[str],
) -> None:
    """Each explore operator must have its actor class registered in the collection."""
    for name in expected_explore_operators:
        operator = orchestrator.modules.operators.collections.explore.operators[name]
        cls = operator.cls
        assert cls is not None


def test_explore_operator_function_conf_identifier_matches_registered_name() -> None:
    """operatorIdentifier via OperatorReference must use the registered name."""
    for name in ["random_walk", "ray_tune"]:
        conf = orchestrator.core.operation.config.OperatorReference(
            operatorName=name,
            operationType=orchestrator.core.operation.config.DiscoveryOperationEnum.SEARCH,
        )
        # The identifier must start with the registered function name, not the
        # class name (e.g. "random_walk@2.0.0", not "RandomWalk@...")
        identifier = conf.operatorIdentifier
        assert identifier.startswith(
            f"{name}@"
        ), f"Expected identifier to start with '{name}@', got '{identifier}'"


def test_operator_function_configuration_incorrect_type(
    expected_explore_operators: list[str],
) -> None:

    operation_type = (
        orchestrator.core.operation.config.DiscoveryOperationEnum.CHARACTERIZE
    )

    for operator_name in expected_explore_operators:
        operationConf = orchestrator.core.operation.config.OperatorReference(
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

    operationConf = orchestrator.core.operation.config.OperatorReference(
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

    operationConf = orchestrator.core.operation.config.OperatorReference(
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

    assert (
        orchestrator.modules.operators.collections.explore.operators[
            "random_walk"
        ].function
        is not None
    )
    assert (
        orchestrator.modules.operators.collections.explore.operators[
            "random_walk"
        ].configuration_model
        == RandomWalkParameters
    )
    assert (
        orchestrator.modules.operators.collections.explore.operators[
            "random_walk"
        ].example_configuration
        == RandomWalk.operator_metadata().example_configuration
    )


def test_raytune_operation_configuration(
    raytuneConf: DiscoveryOperationResourceConfiguration,
) -> None:

    from ado_ray_tune.operator import (
        RayTune,
        RayTuneConfiguration,
    )

    assert (
        orchestrator.modules.operators.collections.explore.operators[
            "ray_tune"
        ].function
        is not None
    )
    assert (
        orchestrator.modules.operators.collections.explore.operators[
            "ray_tune"
        ].configuration_model
        == RayTuneConfiguration
    )
    assert (
        orchestrator.modules.operators.collections.explore.operators[
            "ray_tune"
        ].example_configuration
        == RayTune.operator_metadata().example_configuration
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
    assert RandomWalkParameters.model_validate(randomWalkConf.operation.parameters)

    parameters_model: RandomWalkParameters = RandomWalkParameters.model_validate(
        randomWalkConf.operation.parameters
    )

    # Test sampler
    assert isinstance(parameters_model.samplerConfig, BaseSamplerConfiguration)
    sampler = parameters_model.samplerConfig.sampler()
    assert isinstance(sampler, ExplicitEntitySpaceGridSampleGenerator)
    assert sampler.mode == WalkModeEnum.RANDOM

    # Test extra params not allowed

    parameters_dict = randomWalkConf.operation.parameters.model_dump()
    parameters_dict["foo"] = "bar"

    with pytest.raises(pydantic.ValidationError):
        RandomWalkParameters.model_validate(parameters_dict)

    # Test extra params not allowed

    parameters_dict = randomWalkConf.operation.parameters.model_dump()
    parameters_dict.pop("numberEntities")
    parameters_dict["number-iterations"] = 6

    with pytest.raises(pydantic.ValidationError):
        RandomWalkParameters.model_validate(parameters_dict)


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
    from ado_ray_tune.operator import RayTuneConfiguration

    assert raytuneConf is not None
    assert RayTuneConfiguration.model_validate(raytuneConf.operation.parameters)

    parameters_dict = raytuneConf.operation.parameters.model_dump()
    parameters_dict["foo"] = "bar"

    with pytest.raises(pydantic.ValidationError):
        RandomWalkParameters.model_validate(parameters_dict)


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
    assert RandomWalkParameters.model_validate(randomWalkConf.operation.parameters)

    random_walk_fn = orchestrator.modules.operators.collections.explore.operators[
        "random_walk"
    ].function
    assert random_walk_fn is not None

    operationOutput = random_walk_fn(
        discoverySpace, **randomWalkConf.operation.parameters.model_dump()
    )

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
    spaces = discoverySpace.metadataStore.get_resources_by_relationship(
        kind=CoreResourceKinds.OPERATION,
        identifier=operationOutput.operation.identifier,
        hierarchy_direction="up",
        max_hops=1,
        identifiers_only=True,
    ).get(CoreResourceKinds.DISCOVERYSPACE, set())
    assert len(spaces) == 1
    assert discoverySpace.uri in spaces

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
    random_walk_fn = orchestrator.modules.operators.collections.explore.operators[
        "random_walk"
    ].function
    assert random_walk_fn is not None

    try:
        random_walk_fn(
            discoverySpace, **invalidRandomWalkConf.operation.parameters.model_dump()
        )
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

    from ado_ray_tune.operator import RayTuneConfiguration

    import orchestrator.core.resources

    discoverySpace = ml_multi_cloud_space

    assert discoverySpace is not None
    assert raytuneConf is not None
    assert RayTuneConfiguration.model_validate(raytuneConf.operation.parameters)

    ray_tune_fn = orchestrator.modules.operators.collections.explore.operators[
        "ray_tune"
    ].function
    assert ray_tune_fn is not None

    operationOutput = ray_tune_fn(
        discoverySpace, **raytuneConf.operation.parameters.model_dump()
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
    spaces = discoverySpace.metadataStore.get_resources_by_relationship(
        kind=CoreResourceKinds.OPERATION,
        identifier=operationOutput.operation.identifier,
        hierarchy_direction="up",
        max_hops=1,
        identifiers_only=True,
    ).get(CoreResourceKinds.DISCOVERYSPACE, set())
    assert len(spaces) == 1
    assert discoverySpace.uri in spaces


def test_operator_default_and_validate(
    optimizer_operator: type[RandomWalk] | type[RayTune],
) -> None:

    meta = optimizer_operator.operator_metadata()
    assert meta.configuration_model is not None
    assert meta.example_configuration is not None
    parameters = meta.example_configuration.model_dump()
    assert meta.configuration_model.model_validate(parameters)


# ---------------------------------------------------------------------------
# OperatorMetadata and explore_operation class-decorator tests
# ---------------------------------------------------------------------------


def test_operator_metadata_identifier_property() -> None:
    """OperatorMetadata.operatorIdentifier returns '{name}@{version}'."""
    import pydantic

    from orchestrator.core.operation.config import (
        DiscoveryOperationEnum,
        OperatorMetadata,
    )

    class _P(pydantic.BaseModel):
        pass

    meta = OperatorMetadata(
        name="my_op",
        version="2.0.0",
        configuration_model=_P,
        example_configuration=_P(),
        type=DiscoveryOperationEnum.SEARCH,
    )
    assert meta.operatorIdentifier == "my_op@2.0.0"


def test_operator_metadata_identifier_default_version() -> None:
    """OperatorMetadata.operatorIdentifier uses '2.0.0' when version is not supplied."""
    import pydantic

    from orchestrator.core.operation.config import (
        DiscoveryOperationEnum,
        OperatorMetadata,
    )

    class _P(pydantic.BaseModel):
        pass

    meta = OperatorMetadata(
        name="op",
        configuration_model=_P,
        example_configuration=_P(),
        type=DiscoveryOperationEnum.SEARCH,
    )
    assert meta.operatorIdentifier == "op@0.1.0"


def test_operator_metadata_version_valid_semver() -> None:
    """OperatorMetadata accepts valid strict SemVer version strings."""
    import pydantic

    from orchestrator.core.operation.config import (
        DiscoveryOperationEnum,
        OperatorMetadata,
    )

    class _P(pydantic.BaseModel):
        pass

    valid_versions = [
        "0.1.0",
        "1.2.3",
        "2.0.0",
        "10.20.30",
    ]
    for ver in valid_versions:
        meta = OperatorMetadata(
            name="op",
            version=ver,
            configuration_model=_P,
            example_configuration=_P(),
            type=DiscoveryOperationEnum.SEARCH,
        )
        assert meta.version == ver


def test_operator_metadata_version_invalid_semver() -> None:
    """OperatorMetadata rejects strings that are not valid strict SemVer versions."""
    import pydantic

    from orchestrator.core.operation.config import (
        DiscoveryOperationEnum,
        OperatorMetadata,
    )

    class _P(pydantic.BaseModel):
        pass

    invalid_versions = [
        "not-a-version",
        "hello",
        "1.0.0-final",
        "1.0.2.dev17+5e50632",
        "2.0.0a1",
        "v1.0.0",
        "1.0",
    ]
    for ver in invalid_versions:
        with pytest.raises(pydantic.ValidationError, match="SemVer"):
            OperatorMetadata(
                name="op",
                version=ver,
                configuration_model=_P,
                example_configuration=_P(),
                type=DiscoveryOperationEnum.SEARCH,
            )


def test_operator_function_conf_identifier_delegates_to_operator_metadata() -> None:
    """OperatorReference.operatorIdentifier equals explore.operators[name].operatorIdentifier."""
    from orchestrator.core.operation.config import (
        DiscoveryOperationEnum,
        OperatorReference,
    )
    from orchestrator.modules.operators.collections import explore

    for name in ["random_walk", "ray_tune"]:
        conf = OperatorReference(
            operatorName=name,
            operationType=DiscoveryOperationEnum.SEARCH,
        )
        assert conf.operatorIdentifier == explore.operators[name].operatorIdentifier


def test_explore_operation_class_decorator_registers_function() -> None:
    """@explore_operation returns the class unchanged and stores the OperatorFunction in the collection."""
    import inspect

    import pydantic

    from orchestrator.core.operation.config import (
        DiscoveryOperationEnum,
        OperatorMetadata,
    )
    from orchestrator.modules.operators.base import Explore
    from orchestrator.modules.operators.collections import explore, explore_operation

    class _Params(pydantic.BaseModel):
        pass

    @explore_operation
    class _TestOp(Explore):
        @classmethod
        def operator_metadata(cls) -> OperatorMetadata:
            return OperatorMetadata(
                name="_test_class_op",
                version="0.1.0",
                description="A test operator.",
                configuration_model=_Params,
                example_configuration=_Params(),
                type=DiscoveryOperationEnum.SEARCH,
            )

        def operationIdentifier(self) -> str:
            return "_test_class_op-run"

        async def run(self) -> None:
            pass

    # The decorator returns the class unchanged
    assert isinstance(_TestOp, type)
    assert issubclass(_TestOp, Explore)

    # The generated OperatorFunction is stored in the collection
    assert "_test_class_op" in explore.operators
    fn = explore.operators["_test_class_op"].function
    assert callable(fn)
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    assert "discoverySpace" in params
    assert "operationInfo" in params


def test_explore_operation_class_decorator_cls_stored() -> None:
    """explore.operators[name].cls is the unwrapped class after class decoration."""
    import pydantic

    from orchestrator.core.operation.config import (
        DiscoveryOperationEnum,
        OperatorMetadata,
    )
    from orchestrator.modules.operators.base import Explore
    from orchestrator.modules.operators.collections import explore, explore_operation

    class _ParamsCls(pydantic.BaseModel):
        pass

    @explore_operation
    class _TestOpCls(Explore):
        @classmethod
        def operator_metadata(cls) -> OperatorMetadata:
            return OperatorMetadata(
                name="_test_cls_stored",
                version="0.1.0",
                configuration_model=_ParamsCls,
                example_configuration=_ParamsCls(),
                type=DiscoveryOperationEnum.SEARCH,
            )

        async def run(self) -> None:
            pass

    op = explore.operators.get("_test_cls_stored")
    assert op is not None
    assert op.cls is not None


def test_explore_operation_class_decorator_metadata_from_class() -> None:
    """All OperatorMetadata fields come from the class's operator_metadata()."""
    import pydantic

    from orchestrator.core.operation.config import (
        DiscoveryOperationEnum,
        OperatorMetadata,
    )
    from orchestrator.modules.operators.base import Explore
    from orchestrator.modules.operators.collections import explore, explore_operation

    class _Params2(pydantic.BaseModel):
        x: int = 42

    @explore_operation
    class _TestOp2(Explore):
        @classmethod
        def operator_metadata(cls) -> OperatorMetadata:
            return OperatorMetadata(
                name="_test_class_op2",
                version="3.0.0",
                description="Another test operator.",
                configuration_model=_Params2,
                example_configuration=_Params2(),
                type=DiscoveryOperationEnum.SEARCH,
            )

        def operationIdentifier(self) -> str:
            return "_test_class_op2-run"

        async def run(self) -> None:
            pass

    registered = explore.operators["_test_class_op2"]
    assert registered.name == "_test_class_op2"
    assert registered.version == "3.0.0"
    assert registered.description == "Another test operator."
    assert registered.configuration_model is _Params2
    assert isinstance(registered.example_configuration, _Params2)
    assert registered.type == DiscoveryOperationEnum.SEARCH


def test_explore_operation_class_decorator_missing_operator_metadata_raises() -> None:
    """Decorating a Search subclass without operator_metadata() raises NotImplementedError."""

    from orchestrator.modules.operators.base import Explore
    from orchestrator.modules.operators.collections import explore_operation

    with pytest.raises(NotImplementedError):

        @explore_operation
        class _BadOp(Explore):
            # No operator_metadata() and no legacy classmethods — must raise.
            async def run(self) -> None:
                pass


def test_random_walk_registration() -> None:
    from orchestrator.modules.operators.collections import explore

    assert "random_walk" in explore.operators
    rw = explore.operators["random_walk"]
    assert rw.name == "random_walk"
    assert rw.cls is not None
    assert callable(rw.function)


def test_ray_tune_registration() -> None:
    from orchestrator.modules.operators.collections import explore

    assert "ray_tune" in explore.operators
    rt = explore.operators["ray_tune"]
    assert rt.name == "ray_tune"
    assert rt.cls is not None
    assert callable(rt.function)


def test_warn_if_operator_name_reused_logs_for_duplicate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Reusing an operator name logs a warning before the registry entry is replaced."""
    from orchestrator.core.operation.config import (
        DiscoveryOperationEnum,
        OperatorMetadata,
    )
    from orchestrator.modules.operators.collections import _warn_if_operator_name_reused

    class _Cfg(pydantic.BaseModel):
        pass

    placeholder = OperatorMetadata(
        name="dup",
        configuration_model=_Cfg,
        example_configuration=_Cfg(),
        type=DiscoveryOperationEnum.CHARACTERIZE,
    )
    ops: dict[str, OperatorMetadata] = {"dup": placeholder}

    with caplog.at_level(logging.WARNING):
        _warn_if_operator_name_reused("characterize", "dup", ops)

    assert any("already registered" in r.getMessage() for r in caplog.records)
