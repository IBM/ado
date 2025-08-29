# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import datetime
import typing

import pydantic
import pytest
import yaml

from orchestrator.core.operation.config import (
    DiscoveryOperationConfiguration,
    DiscoveryOperationEnum,
    DiscoveryOperationResourceConfiguration,
)
from orchestrator.core.operation.resource import (
    OperationExitStateEnum,
    OperationResource,
    OperationResourceEventEnum,
    OperationResourceStatus,
)
from orchestrator.core.resources import ADOResourceEventEnum, CoreResourceKinds
from orchestrator.modules.module import load_module_class_or_function


@pytest.fixture
def operation_result() -> typing.Any:

    # Return the default
    return {}


@pytest.fixture
def operation_resource(operation_configuration) -> OperationResource:

    # This auto-generates the operation identifier
    return OperationResource(
        operatorIdentifier="test_operator",
        operationType=DiscoveryOperationEnum.SEARCH,
        config=operation_configuration,
    )


def test_operation_resource(operation_resource):

    assert operation_resource.operatorIdentifier is not None
    assert operation_resource.operatorIdentifier == "test_operator"
    assert operation_resource.identifier is not None
    x = operation_resource.identifier.split("-")
    assert "-".join(x[:2]) == "operation-test_operator"

    assert operation_resource.kind == CoreResourceKinds.OPERATION
    assert operation_resource.identifier.split("-")[0] == "operation"
    assert (
        len(operation_resource.identifier)
        == len("operation") + len("test_operator") + 2 + 8
    )
    assert operation_resource.created < datetime.datetime.now(datetime.timezone.utc)
    assert isinstance(operation_resource.metadata, dict)
    assert operation_resource.config is not None
    assert isinstance(
        operation_resource.config, DiscoveryOperationResourceConfiguration
    )
    assert operation_resource.config.operation.parameters is not None
    assert len(operation_resource.status) == 1


def test_operation_resource_event_status():
    """Test we can set additional event status for operation resources"""

    # Check we can create a resource with generic field
    status = OperationResourceStatus(event=ADOResourceEventEnum.UPDATED)
    assert status.event == ADOResourceEventEnum.UPDATED
    assert status.recorded_at
    assert not status.exit_state

    # Check we can create a resource with operation field
    status = OperationResourceStatus(event=OperationResourceEventEnum.STARTED)
    assert status.event == OperationResourceEventEnum.STARTED
    assert status.recorded_at
    assert not status.exit_state
    dump = status.model_dump()

    # check deser
    deser = OperationResourceStatus.model_validate(dump)
    assert deser.event == OperationResourceEventEnum.STARTED


def test_operation_resource_exit_state():
    """Test we can set additional event status for operation resources"""

    # Check we can create an event+exit status for operation
    status = OperationResourceStatus(
        event=OperationResourceEventEnum.FINISHED,
        exit_state=OperationExitStateEnum.FAIL,
    )
    assert status.event == OperationResourceEventEnum.FINISHED
    assert status.recorded_at
    assert status.exit_state == OperationExitStateEnum.FAIL

    # Check dumping with exit_state dumps exit_state
    dump = status.model_dump()
    assert dump.get("event")
    assert dump.get("exit_state")

    # Check deser
    deser = OperationResourceStatus.model_validate(dump)
    assert deser.event == status.event
    assert deser.recorded_at == status.recorded_at
    assert deser.exit_state == status.exit_state

    # Check we can not create an exit-code status without also setting a FINISHED event
    with pytest.raises(pydantic.ValidationError):
        OperationResourceStatus(exit_state=OperationExitStateEnum.FAIL)

    # Check we can not create an exit-code status without also setting a FINISHED event
    with pytest.raises(pydantic.ValidationError):
        OperationResourceStatus(
            event=OperationResourceEventEnum.STARTED,
            exit_state=OperationExitStateEnum.FAIL,
        )

    # Check we can create a resource with operation field
    status = OperationResourceStatus(event=OperationResourceEventEnum.STARTED)
    assert status.event == OperationResourceEventEnum.STARTED
    assert status.recorded_at


def test_operation_config_file_valid(valid_operation_config_file):

    with open(valid_operation_config_file) as f:
        content = f.read()

    op_cfg = DiscoveryOperationResourceConfiguration.model_validate(
        yaml.safe_load(content)
    )

    try:
        module = op_cfg.module
    except AttributeError:
        pass
    else:
        moduleClass = load_module_class_or_function(
            module
        )  # type: "orchestrator.modules.operators.base.DiscoveryOperationBase"
        moduleClass.validateOperationParameters(parameters=op_cfg.parameters)


def test_set_manual_operation_identifier(operation_configuration):

    test = OperationResource(
        operatorIdentifier="test",
        identifier="test-xxxdd3",
        operationType=DiscoveryOperationEnum.CHARACTERIZE,
        config=operation_configuration,
    )
    assert test.identifier == "test-xxxdd3"


def test_setting_space_id(operation_configuration):

    import pydantic

    # Test setting empty spaces raises an error
    with pytest.raises(pydantic.ValidationError):
        DiscoveryOperationResourceConfiguration(
            spaces=[], operation=DiscoveryOperationConfiguration()
        )

    # Test setting no space raises an error
    with pytest.raises(pydantic.ValidationError):
        DiscoveryOperationResourceConfiguration(
            operation=DiscoveryOperationConfiguration()
        )


def test_add_operation_result(operation_resource, operation_result):

    pass
