# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
from pathlib import Path

import pytest
import yaml
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

    config = DiscoveryOperationResourceConfiguration.model_validate(
        yaml.safe_load(
            Path(
                "examples/ml-multi-cloud/randomwalk_ml_multicloud_operation.yaml"
            ).read_text()
        )
    )

    if request.param == "all":
        config.operation.parameters["numberEntities"] = "all"

    return config


@pytest.fixture(params=["valueGreaterThanSize", "extraField"])
def invalidRandomWalkConf(
    request: pytest.FixtureRequest,
) -> DiscoveryOperationResourceConfiguration:

    config = DiscoveryOperationResourceConfiguration.model_validate(
        yaml.safe_load(
            Path(
                "examples/ml-multi-cloud/randomwalk_ml_multicloud_operation.yaml"
            ).read_text()
        )
    )

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
    return DiscoveryOperationResourceConfiguration.model_validate(
        yaml.safe_load(
            Path(
                "examples/ml-multi-cloud/raytune_ml_multicloud_operation.yaml"
            ).read_text()
        )
    )


@pytest.fixture(params=[RandomWalk, RayTune])
def optimizer_operator(
    request: pytest.FixtureRequest,
) -> type[RandomWalk] | type[RayTune]:

    return request.param
