# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import os
import re
from collections.abc import Callable
from typing import Any, Literal

import pytest

from ado.modules.actuators import custom_experiments
from ado.modules.actuators.measurement_queue import MeasurementQueue
from ado.schema.domain import PropertyDomain, VariableTypeEnum
from ado.schema.entity import Entity
from ado.schema.observed_property import ObservedProperty, ObservedPropertyValue
from ado.schema.point import SpacePoint
from ado.schema.property import AbstractPropertyDescriptor, ConstitutiveProperty
from ado.schema.reference import ExperimentReference
from ado.schema.request import MeasurementRequest, MeasurementRequestStateEnum
from ado.schema.result import InvalidMeasurementResult, ValidMeasurementResult


def test_custom_experiment_unknown_keys_are_dropped() -> None:
    @custom_experiments.custom_experiment(output_property_identifiers=["x", "y"])
    def f_unknown_dropped(x: int, y: int):
        return {"x": 1, "y": 2, "z": 3, "foo": 4}

    exp = f_unknown_dropped._experiment
    entity = SpacePoint.model_validate({"entity": {"x": 10, "y": 20}}).to_entity()
    observed_values = custom_experiments._call_decorated_custom_experiment(
        f_unknown_dropped, exp, entity
    )
    identifiers = {v.property.targetProperty.identifier for v in observed_values}
    assert identifiers == {"x", "y"}


def test_custom_experiment_only_unknown_keys_raises_value_error() -> None:
    @custom_experiments.custom_experiment(output_property_identifiers=["x2", "y2"])
    def f_only_unknown(x2: int, y2: int):
        return {"z": 3, "foo": 4}  # none of these match the output property identifiers

    exp = f_only_unknown._experiment
    entity = SpacePoint.model_validate({"entity": {"x2": 1, "y2": 2}}).to_entity()
    with pytest.raises(ValueError, match="No valid output properties"):
        custom_experiments._call_decorated_custom_experiment(
            f_only_unknown, exp, entity
        )


def test_custom_experiment_partial_output_keys() -> None:
    @custom_experiments.custom_experiment(output_property_identifiers=["a", "b", "c"])
    def f_partial(a: int, b: int, c: int):
        return {"a": 10, "junk": 999}

    exp = f_partial._experiment
    entity = SpacePoint.model_validate({"entity": {"a": 1, "b": 2, "c": 3}}).to_entity()
    observed_values = custom_experiments._call_decorated_custom_experiment(
        f_partial, exp, entity
    )
    identifiers = {v.property.targetProperty.identifier for v in observed_values}
    assert "a" in identifiers
    assert "junk" not in identifiers


def test_custom_experiment_exact_output_keys() -> None:
    @custom_experiments.custom_experiment(output_property_identifiers=["foo", "bar"])
    def f_exact(foo: int, bar: int):
        return {"foo": 123, "bar": 456}

    exp = f_exact._experiment
    entity = SpacePoint.model_validate({"entity": {"foo": 1, "bar": 2}}).to_entity()
    observed_values = custom_experiments._call_decorated_custom_experiment(
        f_exact, exp, entity
    )
    identifiers = {v.property.targetProperty.identifier for v in observed_values}
    assert "foo" in identifiers
    assert "bar" in identifiers
    assert len(observed_values) == 2


def test_infer_domain_and_property_type() -> None:
    """Tests that given a parameter name, its type and a default the correct behaviour is observed"""

    fn = custom_experiments._infer_domain_and_property
    # int
    p = fn("a", int, 42)
    assert p.propertyDomain.variableType == VariableTypeEnum.DISCRETE_VARIABLE_TYPE
    # float
    p = fn("b", float, 3.1)
    assert p.propertyDomain.variableType == VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE
    # bool
    p = fn("c", bool, True)
    assert p.propertyDomain.variableType == VariableTypeEnum.BINARY_VARIABLE_TYPE
    # str
    p = fn("d", str, "hello")
    assert (
        p.propertyDomain.variableType == VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE
    )
    assert p.propertyDomain.values == ["hello"]
    # Literal
    p = fn("e", Literal["X", "Y"], "X")
    assert p.propertyDomain.variableType == VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE
    assert set(p.propertyDomain.values) == {"X", "Y"}
    # bytes is not supported
    with pytest.raises(ValueError, match=r"Unsupported annotation: <class 'bytes'>"):
        _ = fn("f", bytes, b"err")


def test_derive_required_properties_from_signature_basic() -> None:
    def f(a: int, b: float, c: int = 1) -> None:
        pass

    result = custom_experiments.derive_required_properties_from_signature(
        f, optional_property_identifiers=[]
    )
    # a, b expected (no-domain), c skipped as optional
    ids = {r.identifier for r in result}
    assert ids == {"a", "b"}

    # Check that missing annotation raises error
    def f(a, b: float, c: int = 1) -> None:
        pass

    with pytest.raises(
        ValueError, match=r"Unsupported annotation: <class 'inspect._empty'>"
    ):
        custom_experiments.derive_required_properties_from_signature(
            f, optional_property_identifiers=[]
        )


def test_get_parameterization_success_and_failure() -> None:
    import inspect

    from ado.schema.property import ConstitutiveProperty

    def g(x=7, y=9) -> None:
        pass

    sig = inspect.signature(g)
    ps = [ConstitutiveProperty(identifier="x"), ConstitutiveProperty(identifier="y")]
    paramz = custom_experiments.get_parameterization(ps, sig)
    assert paramz["x"] == 7
    assert paramz["y"] == 9

    def g2(x) -> None:
        pass

    sig2 = inspect.signature(g2)
    with pytest.raises(
        ValueError, match=re.escape("Parameterization missing for: ['x']")
    ):
        custom_experiments.get_parameterization(
            [ConstitutiveProperty(identifier="x")], sig2
        )


def test_derive_optional_properties_and_parameterization_basic_types_and_unsupported() -> (
    None
):
    # covers int, float, bool, str, literal, and unsupported

    def fn(
        i: int = 1,
        f: float = 2.0,
        b: bool = False,
        s: str = "abc",
        lit: Literal["A", "B"] = "A",
    ) -> None:
        pass

    optionals, _ = custom_experiments.derive_optional_properties_and_parameterization(
        fn, []
    )
    types = {p.identifier: p.propertyDomain.variableType for p in optionals}
    print(types)
    assert types["i"] == VariableTypeEnum.DISCRETE_VARIABLE_TYPE
    assert types["f"] == VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE
    assert types["b"] == VariableTypeEnum.BINARY_VARIABLE_TYPE
    assert types["s"] == VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE
    assert types["lit"] == VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE

    # bytes is not supported for inference - check fails
    def fn(
        i: int = 1,
        f: float = 2.0,
        b: bool = False,
        s: str = "abc",
        lit: Literal["A", "B"] = "A",
        n: bytes = b"foo",
    ) -> None:
        pass

    with pytest.raises(ValueError, match="Unsupported annotation: <class 'bytes'>"):
        optionals, _ = (
            custom_experiments.derive_optional_properties_and_parameterization(fn, [])
        )

    # i has no annotation - check fails
    def fn(
        i=1,
        f: float = 2.0,
        b: bool = False,
        s: str = "abc",
        lit: Literal["A", "B"] = "A",
    ) -> None:
        pass

    with pytest.raises(
        ValueError, match=r"Unsupported annotation: <class 'inspect._empty'>"
    ):
        optionals, _ = (
            custom_experiments.derive_optional_properties_and_parameterization(fn, [])
        )


def test_custom_experiment_executor_invalid_result_status_is_failed() -> None:
    """When custom experiment produces only InvalidMeasurementResult, status is FAILED."""

    @custom_experiments.custom_experiment(output_property_identifiers=["x"])
    def raises_fn(x: int):
        raise ValueError("infeasible point")

    entity = SpacePoint.model_validate({"entity": {"x": 1}}).to_entity()
    exp = raises_fn._experiment
    results: list = []

    class MockQueue:
        def put(self, item, block=False):
            results.append(item)

    request = MeasurementRequest(
        operation_id="test-op",
        requestIndex=0,
        experimentReference=exp.reference,
        entities=[entity],
    )

    custom_experiments.custom_experiment_executor(
        raises_fn,
        request,
        exp,
        MockQueue(),
    )

    assert len(results) == 1
    result = results[0]
    assert result.status == MeasurementRequestStateEnum.FAILED
    assert result.measurements is not None
    assert len(result.measurements) == 1
    from ado.schema.result import InvalidMeasurementResult

    assert isinstance(result.measurements[0], InvalidMeasurementResult)


def test_custom_experiment_executor_remote_without_ray_options() -> None:
    """A custom experiment with no ray_options is wrapped by ray.remote(fn)."""

    @custom_experiments.custom_experiment(output_property_identifiers=["y"])
    def ray_options_dispatch_unset(x: int) -> dict[str, int]:
        return {"y": x}

    remote_fn = custom_experiments._custom_experiment_executor_remote(
        ray_options_dispatch_unset
    )
    options = remote_fn._default_options
    assert not options, (
        f"Expected _custom_experiment_executor_remote to not set additional ray remote options. It set {options}."
    )


def test_custom_experiment_executor_remote_empty_ray_options() -> None:
    """An empty ray_options dict is wrapped by ray.remote(fn)."""

    @custom_experiments.custom_experiment(
        output_property_identifiers=["y"],
        ray_options={},
    )
    def ray_options_dispatch_empty(x: int) -> dict[str, int]:
        return {"y": x}

    remote_fn = custom_experiments._custom_experiment_executor_remote(
        ray_options_dispatch_empty
    )
    options = remote_fn._default_options
    assert not options, (
        f"Expected _custom_experiment_executor_remote to not set additional ray remote options. It set {options}."
    )


def test_custom_experiment_executor_remote_applies_ray_options() -> None:
    """Non-empty ray_options, including zeros, are applied to the Ray task."""

    requested_options = {
        "num_cpus": 2,
        "num_gpus": 0,
        "resources": {"CustomResource": 1},
        "runtime_env": {"env_vars": {"OMP_NUM_THREADS": "2"}},
    }

    @custom_experiments.custom_experiment(
        output_property_identifiers=["y"], ray_options=requested_options
    )
    def ray_options_dispatch_set(x: int) -> dict[str, int]:
        return {"y": x}

    remote_fn = custom_experiments._custom_experiment_executor_remote(
        ray_options_dispatch_set
    )
    options = remote_fn._default_options
    assert options["num_cpus"] == 2.0
    assert options["num_gpus"] == 0.0
    assert options["resources"] == {"CustomResource": 1}
    assert options["runtime_env"] == {"env_vars": {"OMP_NUM_THREADS": "2"}}
    assert options == requested_options, (
        "Expected set ray remote options to be identical to requested"
    )


def test_check_parameters_and_infer() -> None:

    def fn(a: int, b: float, c: int = 1) -> None:
        pass

    optionals, parameterization, required_properties = (
        custom_experiments.check_parameters_and_infer(fn, None, None, None)
    )

    assert len(optionals) == 1
    assert optionals[0].identifier == "c"
    assert (
        optionals[0].propertyDomain.variableType
        == VariableTypeEnum.DISCRETE_VARIABLE_TYPE
    )
    assert optionals[0].propertyDomain.interval == 1

    assert len(parameterization) == 1
    assert parameterization["c"] == 1

    assert len(required_properties) == 2
    assert required_properties[0].identifier == "a"
    assert (
        required_properties[0].propertyDomain.variableType
        == VariableTypeEnum.DISCRETE_VARIABLE_TYPE
    )
    assert required_properties[0].propertyDomain.interval == 1
    assert required_properties[1].identifier == "b"
    assert (
        required_properties[1].propertyDomain.variableType
        == VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE
    )
    assert required_properties[1].propertyDomain.interval is None

    # Check if we pass in optional properties the parameterization is derived from the function signature
    # and the optional property returned is the same as the one passed in
    optionals, parameterization, required_properties = (
        custom_experiments.check_parameters_and_infer(
            func=fn,
            _required_properties=None,
            _optional_properties=[
                ConstitutiveProperty(
                    identifier="c",
                    propertyDomain=PropertyDomain(
                        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
                        interval=2,
                        domainRange=[0, 10],
                    ),
                )
            ],
            _parameterization=None,
        )
    )
    assert len(optionals) == 1
    assert optionals[0].propertyDomain == PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        interval=2,
        domainRange=[0, 10],
    )
    assert parameterization["c"] == 1


def _measured_values_by_identifier(request: MeasurementRequest) -> dict[str, Any]:
    """Return the first entity's measured values keyed by target property identifier.

    Args:
        request: A completed MeasurementRequest.

    Returns:
        Dict mapping target property identifier to measured value.
    """
    return {
        value.property.targetProperty.identifier: value.value
        for value in request.measurements[0].measurements
    }


def test_runtime_env_creates_sandbox_venv() -> None:
    """Only pip/conda/uv runtime_env keys count as sandbox venv creation."""
    assert not custom_experiments.runtime_env_creates_sandbox_venv(None)
    assert not custom_experiments.runtime_env_creates_sandbox_venv({})
    assert not custom_experiments.runtime_env_creates_sandbox_venv(
        {"env_vars": {"OMP_NUM_THREADS": "2"}}
    )
    assert custom_experiments.runtime_env_creates_sandbox_venv({"pip": ["numpy"]})
    assert custom_experiments.runtime_env_creates_sandbox_venv(
        {"conda": {"dependencies": ["numpy"]}}
    )
    assert custom_experiments.runtime_env_creates_sandbox_venv(
        {"uv": {"packages": ["numpy"]}}
    )


def test_execute_no_queue_required() -> None:
    """execute() runs in the calling process when no MeasurementQueue is provided."""

    @custom_experiments.custom_experiment(output_property_identifiers=["y", "pid"])
    def execute_no_queue_exp(x: float) -> dict[str, float]:
        return {"y": x * 2, "pid": os.getpid()}

    actuator = custom_experiments.CustomExperiments()
    result = actuator.execute(
        entities=[SpacePoint.model_validate({"entity": {"x": 3.0}}).to_entity()],
        experimentReference=execute_no_queue_exp._experiment.reference,
        requesterid="test-op",
        requestIndex=0,
    )

    assert result.status == MeasurementRequestStateEnum.SUCCESS
    measured_values = _measured_values_by_identifier(result)
    assert measured_values["y"] == pytest.approx(6.0)
    assert measured_values["pid"] == os.getpid()


def test_execute_with_use_ray() -> None:
    """execute(use_ray=True) runs the experiment in a Ray worker process."""

    @custom_experiments.custom_experiment(output_property_identifiers=["y", "pid"])
    def execute_use_ray_exp(x: float) -> dict[str, float]:
        return {"y": x * 3, "pid": os.getpid()}

    actuator = custom_experiments.CustomExperiments()
    result = actuator.execute(
        entities=[SpacePoint.model_validate({"entity": {"x": 4.0}}).to_entity()],
        experimentReference=execute_use_ray_exp._experiment.reference,
        requesterid="test-op",
        requestIndex=0,
        use_ray=True,
    )

    assert result.status == MeasurementRequestStateEnum.SUCCESS
    measured_values = _measured_values_by_identifier(result)
    assert measured_values["y"] == pytest.approx(12.0)
    assert measured_values["pid"] != os.getpid()


def test_execute_with_use_ray_applies_ray_options() -> None:
    """execute(use_ray=True) reserves the CPUs given in the experiment's ray_options."""

    @custom_experiments.custom_experiment(
        output_property_identifiers=["y", "cpu"],
        ray_options={"num_cpus": 0},
    )
    def execute_ray_options_exp(x: float) -> dict[str, float]:
        import ray

        assigned_cpus = (
            ray.get_runtime_context().get_assigned_resources().get("CPU", 0.0)
        )
        return {"y": x * 2, "cpu": assigned_cpus}

    actuator = custom_experiments.CustomExperiments()
    result = actuator.execute(
        entities=[SpacePoint.model_validate({"entity": {"x": 5.0}}).to_entity()],
        experimentReference=execute_ray_options_exp._experiment.reference,
        requesterid="test-op",
        requestIndex=0,
        use_ray=True,
    )

    assert result.status == MeasurementRequestStateEnum.SUCCESS
    measured_values = _measured_values_by_identifier(result)
    assert measured_values["y"] == pytest.approx(10.0)
    assert measured_values["cpu"] == 0.0


def test_submit_use_ray_false_puts_completed_request_on_queue() -> None:
    """submit() puts the completed request on the queue before returning when use_ray is False."""

    @custom_experiments.custom_experiment(
        output_property_identifiers=["y"],
        use_ray=False,
    )
    def submit_use_ray_false_exp(x: float) -> dict[str, float]:
        return {"y": x * 2}

    queue = MeasurementQueue()
    actuator = custom_experiments.CustomExperiments(queue=queue)
    request_ids = actuator.submit(
        entities=[SpacePoint.model_validate({"entity": {"x": 3.0}}).to_entity()],
        experimentReference=submit_use_ray_false_exp._experiment.reference,
        requesterid="test-op",
        requestIndex=0,
    )

    result = queue.get(block=False)

    assert result.requestid == request_ids[0]
    assert result.status == MeasurementRequestStateEnum.SUCCESS
    assert _measured_values_by_identifier(result)["y"] == pytest.approx(6.0)


def test_execute_unknown_experiment_raises_value_error() -> None:
    """execute() raises ValueError for an experiment not in the catalog."""

    actuator = custom_experiments.CustomExperiments()
    with pytest.raises(
        ValueError, match="is not in the CustomExperiments actuator catalog"
    ):
        actuator.execute(
            entities=[SpacePoint.model_validate({"entity": {"x": 1.0}}).to_entity()],
            experimentReference=ExperimentReference(
                experimentIdentifier="execute_experiment_that_does_not_exist",
                actuatorIdentifier="custom_experiments",
            ),
            requesterid="test-op",
            requestIndex=0,
        )


def test_init_raises_when_experiment_function_is_not_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CustomExperiments() raises ValueError when a catalog experiment has no function.

    The experiment is defined inside the test and then removed from the
    function registry, so the catalog lists an experiment that cannot run.
    """

    @custom_experiments.custom_experiment(output_property_identifiers=["y"])
    def unloadable_exp(x: float) -> dict[str, float]:
        return {"y": x}

    monkeypatch.delitem(
        custom_experiments._custom_experiment_functions,
        unloadable_exp._experiment.identifier,
    )

    with pytest.raises(ValueError, match="has no registered function"):
        custom_experiments.CustomExperiments()


def test_experiment_requires_ray_sandbox_venv() -> None:
    """Sandbox detection follows use_ray and runtime_env package-manager keys."""

    @custom_experiments.custom_experiment(output_property_identifiers=["y"])
    def no_sandbox_exp(x: float) -> dict[str, float]:
        return {"y": x}

    @custom_experiments.custom_experiment(
        output_property_identifiers=["y"],
        ray_options={"runtime_env": {"pip": ["requests"]}},
    )
    def sandbox_exp(x: float) -> dict[str, float]:
        return {"y": x}

    @custom_experiments.custom_experiment(
        output_property_identifiers=["y"],
        use_ray=False,
        ray_options={"runtime_env": {"pip": ["requests"]}},
    )
    def sandbox_ignored_when_no_ray(x: float) -> dict[str, float]:
        return {"y": x}

    actuator = custom_experiments.CustomExperiments()
    assert not actuator.experiment_requires_ray_sandbox_venv(
        no_sandbox_exp._experiment.reference
    )
    assert actuator.experiment_requires_ray_sandbox_venv(
        sandbox_exp._experiment.reference
    )
    assert not actuator.experiment_requires_ray_sandbox_venv(
        sandbox_ignored_when_no_ray._experiment.reference
    )


@pytest.fixture
def observed_input_experiments() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Return an upstream experiment and a downstream experiment using its output.

    The downstream experiment requires the constitutive property ``x`` and the
    upstream experiment's ``upstream_value`` output, and returns their sum as
    ``total``.

    Returns:
        Tuple of ``(upstream, downstream)`` decorated experiment functions.
    """

    @custom_experiments.custom_experiment(
        output_property_identifiers=["upstream_value"]
    )
    def observed_input_upstream_exp(x: float) -> dict[str, float]:
        return {"upstream_value": x * 10}

    upstream_value = (
        observed_input_upstream_exp._experiment.observedPropertyForTargetIdentifier(
            "upstream_value"
        )
    )
    x = ConstitutiveProperty(
        identifier="x",
        propertyDomain=PropertyDomain(
            variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE
        ),
    )

    @custom_experiments.custom_experiment(
        output_property_identifiers=["total"],
        required_properties=[x, upstream_value],
    )
    def observed_input_downstream_exp(
        x: float, upstream_value: float
    ) -> dict[str, float]:
        return {"total": x + upstream_value}

    return observed_input_upstream_exp, observed_input_downstream_exp


def _entity_with_observed_values(
    x: float, observed_property: ObservedProperty, values: list[float]
) -> Entity:
    """Return an entity with constitutive value ``x`` and measured observed values.

    Args:
        x: Value of the constitutive property ``x``.
        observed_property: The observed property the values are measurements of.
        values: Measured values, each added as a separate measurement result.

    Returns:
        The entity.
    """
    entity = SpacePoint.model_validate({"entity": {"x": x}}).to_entity()
    for value in values:
        entity.add_measurement_result(
            ValidMeasurementResult(
                entityIdentifier=entity.identifier,
                measurements=[
                    ObservedPropertyValue(property=observed_property, value=value)
                ],
            )
        )
    return entity


def test_custom_experiment_infers_optional_properties_excluding_observed_inputs(
    observed_input_experiments: tuple[Callable[..., Any], Callable[..., Any]],
) -> None:
    """A defaulted parameter that receives an observed input is not inferred as optional."""
    upstream, _ = observed_input_experiments
    upstream_value = upstream._experiment.observedPropertyForTargetIdentifier(
        "upstream_value"
    )
    x = ConstitutiveProperty(
        identifier="x",
        propertyDomain=PropertyDomain(
            variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE
        ),
    )

    @custom_experiments.custom_experiment(
        output_property_identifiers=["total"],
        required_properties=[x, upstream_value],
    )
    def observed_input_with_default_exp(
        x: float, upstream_value: float = 0.0, scale: float = 2.0
    ) -> dict[str, float]:
        return {"total": scale * (x + upstream_value)}

    experiment = observed_input_with_default_exp._experiment
    assert [p.identifier for p in experiment.optionalProperties] == ["scale"]


def test_custom_experiment_rejects_inputs_with_same_parameter_name() -> None:
    """Decoration fails when two inputs would be passed to the same parameter."""

    def observed_value_from(experiment_identifier: str) -> ObservedProperty:
        """Return an observed property for the target ``value`` of an experiment."""
        return ObservedProperty(
            targetProperty=AbstractPropertyDescriptor(identifier="value"),
            experimentReference=ExperimentReference(
                experimentIdentifier=experiment_identifier,
                actuatorIdentifier="custom_experiments",
            ),
        )

    with pytest.raises(ValueError, match="map to the same function parameter"):

        @custom_experiments.custom_experiment(
            output_property_identifiers=["out"],
            required_properties=[
                observed_value_from("collision_upstream_a"),
                observed_value_from("collision_upstream_b"),
            ],
        )
        def two_observed_inputs_same_target(value: float) -> dict[str, float]:
            return {"out": value}

    constitutive_value = ConstitutiveProperty(
        identifier="value",
        propertyDomain=PropertyDomain(
            variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE
        ),
    )
    with pytest.raises(ValueError, match="map to the same function parameter"):

        @custom_experiments.custom_experiment(
            output_property_identifiers=["out"],
            required_properties=[
                constitutive_value,
                observed_value_from("collision_upstream_a"),
            ],
        )
        def constitutive_and_observed_same_name(value: float) -> dict[str, float]:
            return {"out": value}


def test_execute_passes_observed_value_to_target_parameter(
    observed_input_experiments: tuple[Callable[..., Any], Callable[..., Any]],
) -> None:
    """execute() passes the entity's observed value to the target-named parameter."""
    upstream, downstream = observed_input_experiments
    upstream_value = upstream._experiment.observedPropertyForTargetIdentifier(
        "upstream_value"
    )

    actuator = custom_experiments.CustomExperiments()
    result = actuator.execute(
        entities=[_entity_with_observed_values(2.0, upstream_value, [30.0])],
        experimentReference=downstream._experiment.reference,
        requesterid="test-op",
        requestIndex=0,
    )

    assert result.status == MeasurementRequestStateEnum.SUCCESS
    assert _measured_values_by_identifier(result)["total"] == pytest.approx(32.0)


def test_execute_entity_with_multiple_observed_values_is_invalid(
    observed_input_experiments: tuple[Callable[..., Any], Callable[..., Any]],
) -> None:
    """An entity with more than one value for an observed input gets an invalid result.

    Other entities in the same request are still measured.
    """
    upstream, downstream = observed_input_experiments
    upstream_value = upstream._experiment.observedPropertyForTargetIdentifier(
        "upstream_value"
    )
    single_value_entity = _entity_with_observed_values(1.0, upstream_value, [10.0])
    repeated_value_entity = _entity_with_observed_values(
        2.0, upstream_value, [20.0, 21.0]
    )

    actuator = custom_experiments.CustomExperiments()
    result = actuator.execute(
        entities=[single_value_entity, repeated_value_entity],
        experimentReference=downstream._experiment.reference,
        requesterid="test-op",
        requestIndex=0,
    )

    results_by_entity = {
        measurement_result.entityIdentifier: measurement_result
        for measurement_result in result.measurements
    }
    assert isinstance(
        results_by_entity[single_value_entity.identifier], ValidMeasurementResult
    )
    repeated_result = results_by_entity[repeated_value_entity.identifier]
    assert isinstance(repeated_result, InvalidMeasurementResult)
    assert upstream_value.identifier in repeated_result.reason


def test_execute_entity_missing_observed_value_is_invalid(
    observed_input_experiments: tuple[Callable[..., Any], Callable[..., Any]],
) -> None:
    """An entity with no value for an observed input gets an invalid result.

    Other entities in the same request are still measured.
    """
    upstream, downstream = observed_input_experiments
    upstream_value = upstream._experiment.observedPropertyForTargetIdentifier(
        "upstream_value"
    )
    measured_entity = _entity_with_observed_values(1.0, upstream_value, [10.0])
    missing_value_entity = _entity_with_observed_values(2.0, upstream_value, [])

    actuator = custom_experiments.CustomExperiments()
    result = actuator.execute(
        entities=[measured_entity, missing_value_entity],
        experimentReference=downstream._experiment.reference,
        requesterid="test-op",
        requestIndex=0,
    )

    results_by_entity = {
        measurement_result.entityIdentifier: measurement_result
        for measurement_result in result.measurements
    }
    assert isinstance(
        results_by_entity[measured_entity.identifier], ValidMeasurementResult
    )
    missing_result = results_by_entity[missing_value_entity.identifier]
    assert isinstance(missing_result, InvalidMeasurementResult)
    assert upstream_value.identifier in missing_result.reason
