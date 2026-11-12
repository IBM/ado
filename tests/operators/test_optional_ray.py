# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for optional Ray initialization based on operator requiresRay."""

import os
import pathlib
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from unittest import mock

import ado.modules.operators.randomwalk  # noqa: F401 — loads explore operators
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.config import (
    DiscoveryOperationConfiguration,
    DiscoveryOperationEnum,
    DiscoveryOperationResourceConfiguration,
    FunctionOperationInfo,
    GenericOperatorParameters,
    OperatorReference,
    ScriptOperatorConf,
)
from ado.core.operation.operation import OperationOutput
from ado.core.resources import ADOResourceReference, CoreResourceKinds
from ado.metastore.project import ProjectContext
from ado.modules.operators.collections import (
    characterize,
    characterize_operation,
    explore,
)
from ado.modules.operators.orchestrate import operation_requires_ray
from ado.utilities.location import SQLiteStoreConfiguration


def _one_space(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
    *,
    parameters: GenericOperatorParameters,
) -> OperationOutput:
    """Characterize one discovery space."""
    return OperationOutput()


@contextmanager
def _temporary_characterize(
    *,
    requires_ray: bool,
) -> Iterator[str]:
    """Register a temporary characterize operator and remove it afterwards."""
    name = f"test-ray-op-{uuid.uuid4().hex[:8]}"
    characterize_operation(
        name=name,
        version="1.0.0",
        configuration_model=GenericOperatorParameters,
        example_configuration=GenericOperatorParameters(),
        description="temporary ray-requirement test operator",
        requires_ray=requires_ray,
    )(_one_space)
    try:
        yield name
    finally:
        characterize.operators.pop(name, None)


def _characterize_config(operator_name: str) -> DiscoveryOperationResourceConfiguration:
    """Build a characterize operation config for *operator_name*."""
    return DiscoveryOperationResourceConfiguration(
        operation=DiscoveryOperationConfiguration(
            module=OperatorReference(
                operatorName=operator_name,
                operationType=DiscoveryOperationEnum.CHARACTERIZE,
            ),
            parameters={},
        ),
        inputs={
            "discoverySpace": ADOResourceReference(
                identifier="space-test123",
                kind=CoreResourceKinds.DISCOVERYSPACE,
            )
        },
    )


def test_registered_explore_operators_require_ray() -> None:
    """Explore registration always sets requiresRay."""
    assert explore.operators["random_walk"].requiresRay is True


def test_trim_operator_requires_ray() -> None:
    """Trim nests random_walk explore ops, so it must opt into Ray."""
    import trim.operator  # noqa: F401 — registers the trim operator

    assert characterize.operators["trim"].requiresRay is True


def test_characterize_decorator_requires_ray_opt_in() -> None:
    """requires_ray=True is stored on OperatorMetadata."""
    with _temporary_characterize(requires_ray=True) as name:
        assert characterize.operators[name].requiresRay is True


def test_characterize_decorator_requires_ray_default_false() -> None:
    """requires_ray defaults to False for non-explore operators."""
    with _temporary_characterize(requires_ray=False) as name:
        assert characterize.operators[name].requiresRay is False


def test_operation_requires_ray_follows_operator_metadata() -> None:
    """operation_requires_ray reads requiresRay from the registered operator."""
    with _temporary_characterize(requires_ray=False) as name:
        assert operation_requires_ray(_characterize_config(name)) is False

    with _temporary_characterize(requires_ray=True) as name:
        assert operation_requires_ray(_characterize_config(name)) is True


def test_operation_requires_ray_true_for_explore_reference() -> None:
    """Explore operator references always require Ray."""
    config = DiscoveryOperationResourceConfiguration(
        operation=DiscoveryOperationConfiguration(
            module=OperatorReference(
                operatorName="random_walk",
                operationType=DiscoveryOperationEnum.EXPLORE,
            ),
            parameters={"numberEntities": 1, "batchSize": 1},
        ),
        inputs={
            "discoverySpace": ADOResourceReference(
                identifier="space-test123",
                kind=CoreResourceKinds.DISCOVERYSPACE,
            )
        },
    )
    assert operation_requires_ray(config) is True


def test_operation_requires_ray_true_for_script_operator() -> None:
    """Script operators conservatively require Ray."""
    config = DiscoveryOperationResourceConfiguration.model_construct(
        operation=DiscoveryOperationConfiguration.model_construct(
            module=ScriptOperatorConf(name="inline"),
            parameters={},
        ),
        inputs={},
    )
    assert operation_requires_ray(config) is True


def test_should_initialize_ray_local_vs_job() -> None:
    """Local non-Ray ops skip init; Ray jobs always initialize."""
    from ado.modules.operators.orchestrate import should_initialize_ray

    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("RAY_JOB_CONFIG_JSON_ENV_VAR", None)
        assert should_initialize_ray(False) is False
        assert should_initialize_ray(True) is True

    with mock.patch.dict(
        os.environ, {"RAY_JOB_CONFIG_JSON_ENV_VAR": '{"metadata": {}}'}
    ):
        assert should_initialize_ray(False) is True
        assert should_initialize_ray(True) is True


def test_orchestrate_skips_ray_init_when_not_required_locally(
    tmp_path: pathlib.Path,
) -> None:
    """Local orchestrate does not call ray.init for requiresRay=False."""
    from ado.modules.operators import orchestrate as orchestrate_module

    context = ProjectContext(
        project="no-ray",
        metadataStore=SQLiteStoreConfiguration(path=str(tmp_path / "no-ray.db")),
    )

    with _temporary_characterize(requires_ray=False) as name:
        config = _characterize_config(name)

        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch.object(orchestrate_module.ray, "init") as ray_init,
            mock.patch.object(
                orchestrate_module,
                "graceful_orchestrate_shutdown",
            ) as shutdown,
            mock.patch(
                "ado.core.operation.inputs.resource_references_to_rich_types",
                return_value={},
            ),
            mock.patch.object(
                OperatorReference,
                "operationFunction",
                return_value=lambda **kwargs: OperationOutput(),
            ),
            mock.patch.object(
                orchestrate_module.signal,
                "signal",
            ) as signal_mock,
        ):
            os.environ.pop("RAY_JOB_CONFIG_JSON_ENV_VAR", None)
            output = orchestrate_module.orchestrate(
                operation_resource_configuration=config,
                project_context=context,
            )

        assert isinstance(output, OperationOutput)
        ray_init.assert_not_called()
        shutdown.assert_not_called()
        signal_mock.assert_not_called()


def test_orchestrate_initializes_ray_when_required(
    tmp_path: pathlib.Path,
) -> None:
    """orchestrate initializes Ray for operators with requiresRay=True."""
    from ado.modules.operators import orchestrate as orchestrate_module

    context = ProjectContext(
        project="needs-ray",
        metadataStore=SQLiteStoreConfiguration(path=str(tmp_path / "needs-ray.db")),
    )

    with _temporary_characterize(requires_ray=True) as name:
        config = _characterize_config(name)

        with (
            mock.patch.object(orchestrate_module.ray, "init") as ray_init,
            mock.patch.object(
                orchestrate_module,
                "graceful_orchestrate_shutdown",
            ) as shutdown,
            mock.patch(
                "ado.core.operation.inputs.resource_references_to_rich_types",
                return_value={},
            ),
            mock.patch.object(
                OperatorReference,
                "operationFunction",
                return_value=lambda **kwargs: OperationOutput(),
            ),
            mock.patch.object(
                orchestrate_module.signal,
                "signal",
            ) as signal_mock,
        ):
            output = orchestrate_module.orchestrate(
                operation_resource_configuration=config,
                project_context=context,
            )

        assert isinstance(output, OperationOutput)
        ray_init.assert_called_once()
        shutdown.assert_called_once()
        signal_mock.assert_called_once()


def test_orchestrate_initializes_ray_for_job_even_when_not_required(
    tmp_path: pathlib.Path,
) -> None:
    """Ray job drivers still connect and install SIGTERM handlers."""
    from ado.modules.operators import orchestrate as orchestrate_module

    context = ProjectContext(
        project="job-no-ray",
        metadataStore=SQLiteStoreConfiguration(path=str(tmp_path / "job-no-ray.db")),
    )

    with _temporary_characterize(requires_ray=False) as name:
        config = _characterize_config(name)

        with (
            mock.patch.dict(
                os.environ, {"RAY_JOB_CONFIG_JSON_ENV_VAR": '{"metadata": {}}'}
            ),
            mock.patch.object(orchestrate_module.ray, "init") as ray_init,
            mock.patch.object(
                orchestrate_module,
                "graceful_orchestrate_shutdown",
            ) as shutdown,
            mock.patch(
                "ado.core.operation.inputs.resource_references_to_rich_types",
                return_value={},
            ),
            mock.patch.object(
                OperatorReference,
                "operationFunction",
                return_value=lambda **kwargs: OperationOutput(),
            ),
            mock.patch.object(
                orchestrate_module.signal,
                "signal",
            ) as signal_mock,
        ):
            output = orchestrate_module.orchestrate(
                operation_resource_configuration=config,
                project_context=context,
            )

        assert isinstance(output, OperationOutput)
        ray_init.assert_called_once()
        shutdown.assert_called_once()
        signal_mock.assert_called_once()
