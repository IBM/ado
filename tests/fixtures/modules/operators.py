# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


import pytest
from ado_ray_tune.operator import RayTune

import orchestrator.core
from orchestrator.core.operation.config import (
    DiscoveryOperationResourceConfiguration,
    OperatorModuleConf,
)
from orchestrator.modules.operators.randomwalk import RandomWalk


@pytest.fixture
def expected_characterize_operators() -> list[str]:

    return ["profile", "detect_anomalous_series"]


@pytest.fixture
def expected_explore_operators() -> list[str]:

    return ["random_walk", "ray_tune"]


@pytest.fixture(params=["RandomWalk", "RayTune"])
def operator_module_conf(request: pytest.FixtureRequest) -> OperatorModuleConf:

    if request.param == "RandomWalk":
        return orchestrator.core.operation.config.OperatorModuleConf(
            moduleName="orchestrator.modules.operators.randomwalk",
            moduleClass=request.param,
        )
    return orchestrator.core.operation.config.OperatorModuleConf(
        moduleName="ado_ray_tune.operator",
        moduleClass=request.param,
    )


@pytest.fixture(params=["all", "value"])
def randomWalkConf(
    request: pytest.FixtureRequest,
) -> DiscoveryOperationResourceConfiguration | None:

    import yaml

    with open("examples/ml-multi-cloud/randomwalk_ml_multicloud_operation.yaml") as f:
        d = yaml.safe_load(f)
        config = DiscoveryOperationResourceConfiguration(**d)

    if request.param == "all":
        config.operation.parameters["numberEntities"] = "all"

    return config


@pytest.fixture(params=["valueGreaterThanSize", "extraField"])
def invalidRandomWalkConf(
    request: pytest.FixtureRequest,
) -> DiscoveryOperationResourceConfiguration:

    import yaml

    with open("examples/ml-multi-cloud/randomwalk_ml_multicloud_operation.yaml") as f:
        d = yaml.safe_load(f)
        config = DiscoveryOperationResourceConfiguration(**d)

    if request.param == "valueGreaterThanSize":
        config.operation.parameters["numberEntities"] = 62
    elif request.param == "extraField":
        parameters = config.operation.parameters.copy()
        parameters.pop("numberEntities")
        parameters["number-iterations"] = 10
        config.operation.parameters = parameters

    return config


@pytest.fixture
def raytuneConf() -> DiscoveryOperationResourceConfiguration:

    import yaml

    with open("examples/ml-multi-cloud/raytune_ml_multicloud_operation.yaml") as f:
        d = yaml.safe_load(f)
        return DiscoveryOperationResourceConfiguration(**d)


@pytest.fixture(params=[RandomWalk, RayTune])
def optimizer_operator(
    request: pytest.FixtureRequest,
) -> type[RandomWalk] | type[RayTune]:

    return request.param
