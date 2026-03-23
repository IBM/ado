# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
import pathlib

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

    return ["profile", "detect_anomalous_series", "trim", "no_priors_characterization"]


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
        config.operation.parameters.numberEntities = "all"

    return config


@pytest.fixture
def invalidRandomWalkConf() -> DiscoveryOperationResourceConfiguration:

    config = DiscoveryOperationResourceConfiguration.model_validate(
        yaml.safe_load(
            pathlib.Path(
                "examples/ml-multi-cloud/randomwalk_ml_multicloud_operation.yaml"
            ).read_text()
        )
    )

    config.operation.parameters.numberEntities = 62

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
