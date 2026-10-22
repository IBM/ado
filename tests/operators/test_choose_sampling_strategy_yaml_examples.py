# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests that ado-choose-sampling-strategy YAML examples validate and build their samplers."""

import pathlib

import pytest
import yaml

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:  # pragma: nocover
    from yaml import SafeLoader  # type: ignore[assignment]

from ado.core.operation.config import DiscoveryOperationResourceConfiguration
from ado.modules.operators.randomwalk import RandomWalkParameters

YAML_EXAMPLES_DIR = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / ".agents/skills/ado-choose-sampling-strategy/yaml-examples"
)

RANDOM_WALK_YAMLS = [
    "random-walk-grid",
    "random-walk-extend-unmeasured",
    "random-walk-selector-new-experiment",
]

RAY_TUNE_YAMLS = {
    "ray-tune-optuna-tpe": "optuna",
    "ray-tune-optuna-qmc": "optuna",
    "ray-tune-nevergrad-ngopt": "nevergrad",
}


@pytest.mark.parametrize("yaml_name", RANDOM_WALK_YAMLS, ids=RANDOM_WALK_YAMLS)
def test_random_walk_yaml_examples(yaml_name: str) -> None:
    """Each random_walk example validates as RandomWalkParameters."""
    path = YAML_EXAMPLES_DIR / f"{yaml_name}.yaml"
    config = DiscoveryOperationResourceConfiguration.model_validate(
        yaml.load(path.read_text(), Loader=SafeLoader)
    )
    assert config.operation.module.operatorName == "random_walk"
    params = RandomWalkParameters.model_validate(config.operation.parameters)
    dumped = params.model_dump(mode="json")
    assert RandomWalkParameters.model_validate(dumped) == params


@pytest.mark.parametrize(
    ("yaml_name", "optimizer_package"),
    RAY_TUNE_YAMLS.items(),
    ids=RAY_TUNE_YAMLS.keys(),
)
def test_ray_tune_yaml_examples(yaml_name: str, optimizer_package: str) -> None:
    """Each ray_tune example validates as RayTuneConfiguration and builds its optimizer."""
    ray_tune_config = pytest.importorskip("ado_ray_tune.config")
    pytest.importorskip(optimizer_package)
    path = YAML_EXAMPLES_DIR / f"{yaml_name}.yaml"
    config = DiscoveryOperationResourceConfiguration.model_validate(
        yaml.load(path.read_text(), Loader=SafeLoader)
    )
    assert config.operation.module.operatorName == "ray_tune"
    params = ray_tune_config.RayTuneConfiguration.model_validate(
        config.operation.parameters
    )
    dumped = params.model_dump(mode="json")
    assert (
        ray_tune_config.RayTuneConfiguration.model_validate(dumped).model_dump(
            mode="json"
        )
        == dumped
    )
    search_alg = params.tuneConfig.rayTuneConfig().search_alg
    assert search_alg is not None
