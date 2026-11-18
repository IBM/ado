# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""End-to-end PKH operator integration test (example space + custom experiments)."""

import json
import pathlib
from collections.abc import Callable

import pandas as pd
import pytest
import trim_custom_experiments.experiments  # noqa: F401 — registers ideal-gas experiment
import yaml
from active_learning.pkh.parameters import (
    ActiveLearningAutoGluonParameters,
    PKHParameters,
)

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:  # pragma: nocover
    from yaml import SafeLoader  # type: ignore[assignment]
from testcontainers.community.mysql import MySqlContainer

import ado.modules.operators.randomwalk  # noqa: F401
from ado.core.discoveryspace.config import DiscoverySpaceConfiguration
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.resource import (
    OperationExitStateEnum,
    OperationResourceEventEnum,
)
from ado.core.samplestore.config import (
    SampleStoreConfiguration,
    SampleStoreModuleConf,
    SampleStoreSpecification,
)
from ado.core.samplestore.sql import SQLSampleStore
from ado.metastore.project import ProjectContext
from ado.modules.operators.collections import characterize

# The ideal-gas example space belongs to TRIM: reuse its resource file as is.
IDEAL_GAS_SPACE = pathlib.Path("tests/resources/trim/space_minimal.yaml")


@pytest.fixture
def ideal_gas_discovery_space(
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_sample_store: Callable[[SampleStoreConfiguration], SQLSampleStore],
    create_space: Callable[[DiscoverySpaceConfiguration, str], DiscoverySpace],
) -> DiscoverySpace:
    """Discovery space with the ideal-gas experiment and an empty sample store."""
    space_configuration = DiscoverySpaceConfiguration.model_validate(
        yaml.load(IDEAL_GAS_SPACE.read_text(), Loader=SafeLoader)
    )
    sample_store = create_sample_store(
        SampleStoreConfiguration(
            specification=SampleStoreSpecification(
                module=SampleStoreModuleConf(
                    moduleClass="SQLSampleStore",
                    moduleName="ado.core.samplestore.sql",
                ),
            )
        )
    )
    space = create_space(space_configuration, sample_store.identifier)
    return DiscoverySpace.from_stored_configuration(
        project_context=valid_ado_project_context,
        space_identifier=space.uri,
    )


@pytest.mark.timeout(900)
@pytest.mark.parametrize("backend", ["random_forest", "autogluon"])
def test_pkh_example_operation_succeeds(
    ideal_gas_discovery_space: DiscoverySpace,
    tmp_path: pathlib.Path,
    backend: str,
) -> None:
    """Run PKH on an empty sample store and check the operation and its output."""
    if backend == "autogluon":
        pytest.importorskip("autogluon.tabular")
    assert ideal_gas_discovery_space.matchingEntities() == []
    number_entities = 6
    parameters = PKHParameters(
        targetOutput="pressure",
        numberEntities=number_entities,
        pilotSize=3,
        epochLength=2,
        nEstimators=10,
        outputDirectory=str(tmp_path / "pkh_models"),
        finalPredictor=backend,
        autoGluon=(
            ActiveLearningAutoGluonParameters(
                timeLimit=10,
                presets="medium_quality",
                hyperparameters={"RF": {"n_estimators": 3}},
            )
            if backend == "autogluon"
            else None
        ),
    )
    pkh_function = characterize.operators["pkh"].function
    assert pkh_function is not None

    output = pkh_function(ideal_gas_discovery_space, parameters=parameters)

    assert output.exitStatus is not None
    assert output.exitStatus.exit_state == OperationExitStateEnum.SUCCESS
    assert output.exitStatus.event == OperationResourceEventEnum.FINISHED
    assert output.metadata["pkhMmdSquared"] >= 0.0

    final_directory = tmp_path / "pkh_models_finalized"
    metadata = json.loads((final_directory / "metadata.json").read_text())
    assert metadata["backend"] == backend
    assert metadata["measuredCount"] == number_entities
    assert metadata["mmdSquared"] == output.metadata["pkhMmdSquared"]

    # space_minimal.yaml has 8 entities: the table covers all of them
    predictions = pd.read_csv(final_directory / "combined_predictions.csv")
    assert len(predictions) == 8
    assert (~predictions["is_predicted"]).sum() == number_entities
    assert predictions["pressure"].notna().all()
    if backend == "random_forest":
        assert (final_directory / "model.pkl").exists()
    else:
        assert (final_directory / "autogluon").is_dir()
