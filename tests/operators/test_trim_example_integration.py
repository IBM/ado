# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""End-to-end trim operator integration test (example space + custom experiments)."""

import pathlib
from collections.abc import Callable

import pytest
import trim_custom_experiments.experiments  # noqa: F401 — registers ideal-gas experiment
import yaml
from no_priors_characterization.no_priors_pydantic import NoPriorsParameters
from testcontainers.mysql import MySqlContainer

import orchestrator.modules.operators.randomwalk  # noqa: F401
from orchestrator.core.discoveryspace.config import DiscoverySpaceConfiguration
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.resource import (
    OperationExitStateEnum,
    OperationResourceEventEnum,
)
from orchestrator.core.resources import ADOResourceEventEnum
from orchestrator.core.samplestore.config import (
    SampleStoreConfiguration,
    SampleStoreModuleConf,
    SampleStoreSpecification,
)
from orchestrator.core.samplestore.sql import SQLSampleStore
from orchestrator.metastore.project import ProjectContext
from orchestrator.modules.operators.collections import characterize

pytest.importorskip("autogluon")

from trim.trim_pydantic import (
    AutoGluonArgs,
    SamplingBudget,
    StoppingCriterion,
    TrimParameters,
)


@pytest.fixture
def trim_minimal_discovery_space(
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_sample_store: Callable[[SampleStoreConfiguration], SQLSampleStore],
    create_space: Callable[[DiscoverySpaceConfiguration, str], DiscoverySpace],
) -> DiscoverySpace:
    """Discovery space with ideal-gas experiment (empty sample store — trim may run no-priors)."""
    space_conf = DiscoverySpaceConfiguration.model_validate(
        yaml.safe_load(
            pathlib.Path("tests/resources/trim/space_minimal.yaml").read_text()
        )
    )
    sample_store = create_sample_store(
        SampleStoreConfiguration(
            specification=SampleStoreSpecification(
                module=SampleStoreModuleConf(
                    moduleClass="SQLSampleStore",
                    moduleName="orchestrator.core.samplestore.sql",
                ),
            )
        )
    )
    space = create_space(space_conf, sample_store.identifier)
    return DiscoverySpace.from_stored_configuration(
        project_context=valid_ado_project_context,
        space_identifier=space.uri,
    )


@pytest.mark.timeout(900)
def test_trim_example_operation_succeeds(
    trim_minimal_discovery_space: DiscoverySpace,
) -> None:
    """Run trim on the minimal pressure example; passes if the operation completes successfully."""
    # Trim requires >1 distinct target value before modeling. AutoGluon's internal
    # train/test split needs several rows (fails for n_samples=2). Budget must not
    # exceed tests/resources/trim/space_minimal.yaml entity count (currently 8).
    params = TrimParameters(
        targetOutput="pressure",
        samplingBudget=SamplingBudget(minPoints=8, maxPoints=8),
        iterationSize=5,
        outputDirectory="trim_integration_models",
        stoppingCriterion=StoppingCriterion(enabled=False),
        autoGluonArgs=AutoGluonArgs(
            fitArgs={
                "time_limit": 60,
                "presets": "medium",
                "excluded_model_types": ["GBM"],
            },
            tabularPredictorArgs={
                "problem_type": "regression",
                "verbosity": 0,
            },
        ),
        noPriorParameters=NoPriorsParameters(
            targetOutput="pressure",
            samples=8,
            batchSize=2,
            sampling_strategy="random",
        ),
    )
    trim_fn = characterize.operators["trim"].function
    assert trim_fn is not None

    output = trim_fn(
        trim_minimal_discovery_space,
        **params.model_dump(),
    )

    assert output.operation is not None
    assert output.exitStatus is not None
    assert output.exitStatus.exit_state == OperationExitStateEnum.SUCCESS
    assert output.exitStatus.event == OperationResourceEventEnum.FINISHED
    assert output.operation.status[-1].event == ADOResourceEventEnum.UPDATED
