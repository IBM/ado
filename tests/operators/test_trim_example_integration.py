# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""End-to-end trim operator integration test (example space + custom experiments)."""

import pathlib
from collections.abc import Callable

import pytest
import trim_custom_experiments.experiments  # noqa: F401 — registers ideal-gas experiment
import yaml
from testcontainers.community.mysql import MySqlContainer

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

from trim.samplers.no_priors_parameters import NoPriorsParametersInternal
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


# Minimal AutoGluon settings — fast enough for CI and light enough to survive
# on a developer laptop.  LR only; no neural nets or tree ensembles.
_TRIM_TEST_AUTOGLUON_FIT_ARGS = {
    "time_limit": 10,
    "presets": "medium_quality",
    "auto_stack": False,
    "excluded_model_types": ["CAT", "NN_TORCH", "FASTAI", "GBM", "XGB", "RF"],
}


# @pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    ("min_points", "max_points", "expected_completed_operations"),
    [
        (8, 8, ["Characterization with no priors"]),
        (
            7,
            8,
            ["Characterization with no priors", "Iterative Modeling Operation"],
        ),
    ],
    ids=["budget-exhausted", "budget-remaining"],
)
def test_trim_example_operation_succeeds(
    trim_minimal_discovery_space: DiscoverySpace,
    tmp_path: pathlib.Path,
    min_points: int,
    max_points: int,
    expected_completed_operations: list[str],
) -> None:
    """Run TRIM with exhausted and remaining post-characterization budgets."""
    autogluon_args = AutoGluonArgs(
        fitArgs=_TRIM_TEST_AUTOGLUON_FIT_ARGS,
        tabularPredictorArgs={
            "problem_type": "regression",
            "verbosity": 0,
        },
    )
    params = TrimParameters(
        targetOutput="pressure",
        samplingBudget=SamplingBudget(
            minPoints=min_points,
            maxPoints=max_points,
        ),
        iterationSize=1,
        outputDirectory=str(tmp_path / "trim_models"),
        debugDirectory=str(tmp_path / "debug_output"),
        stoppingCriterion=StoppingCriterion(enabled=False),
        autoGluonArgs=autogluon_args,
        finalModelAutoGluonArgs=autogluon_args,
        noPriorParameters=NoPriorsParametersInternal(
            targetOutput="pressure",
            samples=min_points,
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
    assert [
        resource.config.metadata.model_dump()["completed operation"]
        for resource in output.resources
    ] == expected_completed_operations

    if max_points > min_points:
        iterative_operation = output.resources[-1]
        assert (
            iterative_operation.config.operation.parameters.numberEntities
            # VV: TRIM configure random_walk (i.e.numberEntitise above) to measure 1 point
            # more than what the user actually specified. This enables the TRIM Sampler to
            # know that it has exhausted its budget and thus finalize the model
            == (max_points - min_points) + 1
        )
        assert any(
            status.event == OperationResourceEventEnum.FINISHED
            and status.exit_state == OperationExitStateEnum.SUCCESS
            for status in iterative_operation.status
        )
        # The final model must have been persisted to outputDirectory.
        assert (tmp_path / "trim_models_finalized").is_dir(), (
            "finalize_model was never called: the trim_models_finalized directory was not created"
        )
