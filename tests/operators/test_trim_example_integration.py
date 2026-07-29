# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""End-to-end trim operator integration test (example space + custom experiments)."""

import pathlib
from collections.abc import Callable

import pytest
import trim_custom_experiments.experiments  # noqa: F401 — registers ideal-gas experiment
import yaml
from testcontainers.mysql import MySqlContainer

import ado.modules.operators.randomwalk  # noqa: F401
from ado.core.discoveryspace.config import DiscoverySpaceConfiguration
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.config import FunctionOperationInfo
from ado.core.operation.resource import (
    OperationExitStateEnum,
    OperationResourceEventEnum,
)
from ado.core.resources import ADOResourceEventEnum
from ado.core.samplestore.config import (
    SampleStoreConfiguration,
    SampleStoreModuleConf,
    SampleStoreSpecification,
)
from ado.core.samplestore.sql import SQLSampleStore
from ado.metastore.project import ProjectContext
from ado.modules.operators.collections import characterize

pytest.importorskip("autogluon")

from trim.samplers.no_priors_parameters import NoPriorsParameters
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
                    moduleName="ado.core.samplestore.sql",
                ),
            )
        )
    )
    space = create_space(space_conf, sample_store.identifier)
    return DiscoverySpace.from_stored_configuration(
        project_context=valid_ado_project_context,
        space_identifier=space.uri,
    )


# Lightweight AutoGluon settings for CI
_TRIM_TEST_AUTOGLUON_FIT_ARGS = {
    "time_limit": 60,
    "presets": "medium_quality",
    "auto_stack": False,
    "excluded_model_types": ["CAT"],
}


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.timeout(900)
def test_trim_example_operation_succeeds(
    trim_minimal_discovery_space: DiscoverySpace,
    tmp_path: pathlib.Path,
) -> None:
    """Run trim on the minimal pressure example; passes if the operation completes successfully."""
    # Trim requires >1 distinct target value before modeling. AutoGluon's internal
    # train/test split needs several rows (fails for n_samples=2). Budget must not
    # exceed tests/resources/trim/space_minimal.yaml entity count (currently 8).
    model_dir = tmp_path / "trim_models"
    debug_dir = tmp_path / "debug_output"
    autogluon_args = AutoGluonArgs(
        fitArgs=_TRIM_TEST_AUTOGLUON_FIT_ARGS,
        tabularPredictorArgs={
            "problem_type": "regression",
            "verbosity": 0,
        },
    )
    params = TrimParameters(
        targetOutput="pressure",
        samplingBudget=SamplingBudget(minPoints=8, maxPoints=8),
        iterationSize=1,
        outputDirectory=str(model_dir),
        debugDirectory=str(debug_dir),
        stoppingCriterion=StoppingCriterion(enabled=False),
        autoGluonArgs=autogluon_args,
        finalModelAutoGluonArgs=autogluon_args,
        noPriorParameters=NoPriorsParameters(
            targetOutput="pressure",
            samples=8,
            batchSize=1,
            sampling_strategy="random",
        ),
    )
    trim_fn = characterize.operators["trim"].function
    assert trim_fn is not None

    output = trim_fn(
        trim_minimal_discovery_space,
        operationInfo=FunctionOperationInfo(
            projectContext=trim_minimal_discovery_space.project_context
        ),
        parameters=params,
    )

    assert output.operation is not None
    assert output.exitStatus is not None
    assert output.exitStatus.exit_state == OperationExitStateEnum.SUCCESS
    assert output.exitStatus.event == OperationResourceEventEnum.FINISHED
    assert output.operation.status[-1].event == ADOResourceEventEnum.UPDATED
