# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ado.schema.experiment import Experiment

DEFAULT_MPS = str(pathlib.Path(__file__).parent / "markshare_4_0.mps.gz")

try:
    import cplex  # noqa: F401

    HAS_CPLEX = True
except ImportError:
    HAS_CPLEX = False


@pytest.fixture(scope="module")
def solve_mip_func() -> Callable[..., Any]:
    """Import and return the solve_mip decorated function."""
    from cplex_mip_experiments.solve_mip import solve_mip

    return solve_mip


@pytest.fixture(scope="module")
def experiment(solve_mip_func: Callable[..., Any]) -> Experiment:
    """Return the Experiment object attached to solve_mip."""
    return solve_mip_func._experiment


class TestExperimentRegistration:
    def test_is_custom_experiment(self, solve_mip_func: Callable[..., Any]) -> None:
        """solve_mip must be decorated as a custom experiment."""
        assert getattr(solve_mip_func, "_is_custom_experiment", False)

    def test_experiment_attached(self, solve_mip_func: Callable[..., Any]) -> None:
        """The decorated function must have an _experiment attribute."""
        assert hasattr(solve_mip_func, "_experiment")

    def test_experiment_identifier(self, experiment: Experiment) -> None:
        """Experiment identifier must be 'solve_mip'."""
        assert experiment.identifier == "solve_mip"

    def test_actuator_identifier(self, experiment: Experiment) -> None:
        """Actuator identifier must be 'custom_experiments'."""
        assert experiment.actuatorIdentifier == "custom_experiments"


class TestRequiredProperties:
    def test_mps_file_required(self, experiment: Experiment) -> None:
        """solve_mip requires mps_file."""
        assert len(experiment.requiredProperties) == 1
        assert experiment.requiredProperties[0].identifier == "mps_file"


class TestOptionalProperties:
    def test_all_optional_properties_present(self, experiment: Experiment) -> None:
        """All fourteen optional properties must be defined."""
        optional_ids = {p.identifier for p in experiment.optionalProperties}
        expected = {
            "n_seeds",
            "node_selection",
            "variable_selection",
            "heuristic_frequency",
            "time_limit_s",
            "n_threads",
            "rins_frequency",
            "cut_passes",
            "cut_passes_all",
            "mip_emphasis",
            "parallel",
            "progress_interval_s",
            "warm_start_file",
            "export_solution",
        }
        assert expected == optional_ids

    def test_node_selection_domain_values(self, experiment: Experiment) -> None:
        """node_selection domain must contain exactly [0, 1, 2, 3]."""
        prop = next(
            p for p in experiment.optionalProperties if p.identifier == "node_selection"
        )
        assert sorted(prop.propertyDomain.values) == [0, 1, 2, 3]

    def test_variable_selection_domain_values(self, experiment: Experiment) -> None:
        """variable_selection domain must contain exactly [-1, 0, 1, 2, 3, 4]."""
        prop = next(
            p
            for p in experiment.optionalProperties
            if p.identifier == "variable_selection"
        )
        assert sorted(prop.propertyDomain.values) == [-1, 0, 1, 2, 3, 4]

    def test_heuristic_frequency_domain_values(self, experiment: Experiment) -> None:
        """heuristic_frequency domain must contain exactly [-1, 0, 10, 50, 100]."""
        prop = next(
            p
            for p in experiment.optionalProperties
            if p.identifier == "heuristic_frequency"
        )
        assert sorted(prop.propertyDomain.values) == [-1, 0, 10, 50, 100]

    def test_n_seeds_domain_range(self, experiment: Experiment) -> None:
        """n_seeds domain must have range [1, 100]."""
        prop = next(
            p for p in experiment.optionalProperties if p.identifier == "n_seeds"
        )
        assert prop.propertyDomain.domainRange == [1, 100]

    def test_mps_file_example_in_required_domain(self, experiment: Experiment) -> None:
        """mps_file (required) domain lists an example path."""
        prop = next(
            p for p in experiment.requiredProperties if p.identifier == "mps_file"
        )
        assert "/path/to/instance.mps.gz" in prop.propertyDomain.values

    def test_cut_passes_all_domain_values(self, experiment: Experiment) -> None:
        """cut_passes_all domain must match categorical levels."""
        prop = next(
            p for p in experiment.optionalProperties if p.identifier == "cut_passes_all"
        )
        assert set(prop.propertyDomain.values) == {
            "cplex_default",
            -1,
            0,
            1,
            2,
            3,
        }

    def test_mip_emphasis_domain_values(self, experiment: Experiment) -> None:
        """mip_emphasis domain must be [0, 1, 2, 3, 4, 5]."""
        prop = next(
            p for p in experiment.optionalProperties if p.identifier == "mip_emphasis"
        )
        assert sorted(prop.propertyDomain.values) == [0, 1, 2, 3, 4, 5]

    def test_time_limit_domain_type(self, experiment: Experiment) -> None:
        """time_limit_s must be continuous with a nonnegative range including 1e75."""
        prop = next(
            p for p in experiment.optionalProperties if p.identifier == "time_limit_s"
        )
        from ado.schema.domain import VariableTypeEnum

        assert (
            prop.propertyDomain.variableType
            == VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE
        )
        assert prop.propertyDomain.domainRange == [0, 1e76]

    def test_n_threads_domain_values(self, experiment: Experiment) -> None:
        """n_threads domain must contain exactly [1, 2, 4, 8]."""
        prop = next(
            p for p in experiment.optionalProperties if p.identifier == "n_threads"
        )
        assert sorted(prop.propertyDomain.values) == [1, 2, 4, 8]

    def test_rins_frequency_domain_values(self, experiment: Experiment) -> None:
        """rins_frequency domain must contain exactly [-1, 0, 5, 25, 100]."""
        prop = next(
            p for p in experiment.optionalProperties if p.identifier == "rins_frequency"
        )
        assert sorted(prop.propertyDomain.values) == [-1, 0, 5, 25, 100]

    def test_cut_passes_domain_values(self, experiment: Experiment) -> None:
        """cut_passes domain must span integers -1 through 200 (exclusive upper 201)."""
        prop = next(
            p for p in experiment.optionalProperties if p.identifier == "cut_passes"
        )
        assert prop.propertyDomain.domainRange == [-1, 201]
        assert prop.propertyDomain.interval == 1


class TestDefaultParameterization:
    def test_n_seeds_default(self, experiment: Experiment) -> None:
        """Default n_seeds must be 5."""
        param_map = {
            p.property.identifier: p.value for p in experiment.defaultParameterization
        }
        assert param_map["n_seeds"] == 5

    def test_node_selection_default(self, experiment: Experiment) -> None:
        """Default node_selection must be 1 (BFS)."""
        param_map = {
            p.property.identifier: p.value for p in experiment.defaultParameterization
        }
        assert param_map["node_selection"] == 1

    def test_variable_selection_default(self, experiment: Experiment) -> None:
        """Default variable_selection must be 0 (auto)."""
        param_map = {
            p.property.identifier: p.value for p in experiment.defaultParameterization
        }
        assert param_map["variable_selection"] == 0

    def test_heuristic_frequency_default(self, experiment: Experiment) -> None:
        """Default heuristic_frequency must be 0 (auto)."""
        param_map = {
            p.property.identifier: p.value for p in experiment.defaultParameterization
        }
        assert param_map["heuristic_frequency"] == 0

    def test_time_limit_default(self, experiment: Experiment) -> None:
        """Default time_limit_s must be 1e75 (no practical limit)."""
        param_map = {
            p.property.identifier: p.value for p in experiment.defaultParameterization
        }
        assert param_map["time_limit_s"] == 1e75

    def test_n_threads_default(self, experiment: Experiment) -> None:
        """Default n_threads must be 1 (single-threaded, fully deterministic)."""
        param_map = {
            p.property.identifier: p.value for p in experiment.defaultParameterization
        }
        assert param_map["n_threads"] == 1

    def test_rins_frequency_default(self, experiment: Experiment) -> None:
        """Default rins_frequency must be 0 (auto)."""
        param_map = {
            p.property.identifier: p.value for p in experiment.defaultParameterization
        }
        assert param_map["rins_frequency"] == 0

    def test_cut_passes_default(self, experiment: Experiment) -> None:
        """Default cut_passes must be 0 (auto)."""
        param_map = {
            p.property.identifier: p.value for p in experiment.defaultParameterization
        }
        assert param_map["cut_passes"] == 0

    def test_cut_passes_all_default(self, experiment: Experiment) -> None:
        """Default cut_passes_all must be cplex_default (no overrides)."""
        param_map = {
            p.property.identifier: p.value for p in experiment.defaultParameterization
        }
        assert param_map["cut_passes_all"] == "cplex_default"

    def test_mip_emphasis_default(self, experiment: Experiment) -> None:
        """Default mip_emphasis must be 0 (balanced)."""
        param_map = {
            p.property.identifier: p.value for p in experiment.defaultParameterization
        }
        assert param_map["mip_emphasis"] == 0

    def test_parallel_default(self, experiment: Experiment) -> None:
        """Default parallel must be True (Ray remote execution)."""
        param_map = {
            p.property.identifier: p.value for p in experiment.defaultParameterization
        }
        assert param_map["parallel"]

    def test_progress_interval_default(self, experiment: Experiment) -> None:
        """Default progress_interval_s must be 0 (disabled)."""
        param_map = {
            p.property.identifier: p.value for p in experiment.defaultParameterization
        }
        assert param_map["progress_interval_s"] == 0

    def test_warm_start_file_default(self, experiment: Experiment) -> None:
        """Default warm_start_file must be empty string (disabled)."""
        param_map = {
            p.property.identifier: p.value for p in experiment.defaultParameterization
        }
        assert param_map["warm_start_file"] == ""

    def test_export_solution_default(self, experiment: Experiment) -> None:
        """Default export_solution must be False."""
        param_map = {
            p.property.identifier: p.value for p in experiment.defaultParameterization
        }
        assert not param_map["export_solution"]


class TestTargetProperties:
    def test_all_target_properties_present(self, experiment: Experiment) -> None:
        """All target (output) properties must be defined."""
        target_ids = {p.identifier for p in experiment.targetProperties}
        expected = {
            "solve_times",
            "objective_values",
            "mip_gaps",
            "nodes_explored",
            "solve_statuses",
            "best_bounds",
            "best_solution_mst",
            "progress_time_grid",
            "objective_over_time",
            "best_bound_over_time",
            "nodes_explored_over_time",
            "mip_gap_over_time",
        }
        assert expected == target_ids


class TestExperimentRoundTrip:
    def test_dump_and_reload(self, experiment: Experiment) -> None:
        """Experiment must survive a model_dump → model_validate round-trip."""
        dumped = experiment.model_dump()
        reloaded = Experiment.model_validate(dumped)
        assert reloaded.identifier == experiment.identifier
        assert reloaded.actuatorIdentifier == experiment.actuatorIdentifier
        assert len(reloaded.requiredProperties) == len(experiment.requiredProperties)
        assert len(reloaded.optionalProperties) == len(experiment.optionalProperties)
        assert len(reloaded.targetProperties) == len(experiment.targetProperties)


class TestSolveMipVectorOutput:
    def _make_seed_result(self, seed: int) -> dict[str, Any]:
        """Return a fake single-seed result dict."""
        return {
            "solve_time_s": float(seed + 1),
            "objective_value": -100.0 - seed,
            "best_bound": -90.0 - seed,
            "mip_gap": 0.01 * seed,
            "nodes_explored": 100 * (seed + 1),
            "solve_status": "optimal",
            "best_solution_mst": "",
            "progress_samples": [],
        }

    def test_returns_vectors_of_correct_length(
        self, solve_mip_func: Callable[..., Any]
    ) -> None:
        """With n_seeds=3, all output lists must have length 3."""
        with patch(
            "cplex_mip_experiments.solve_mip._run_single_seed",
            side_effect=lambda **kw: self._make_seed_result(kw["seed"]),
        ):
            result = solve_mip_func(mps_file=DEFAULT_MPS, n_seeds=3, parallel=False)

        assert len(result["solve_times"]) == 3
        assert len(result["objective_values"]) == 3
        assert len(result["mip_gaps"]) == 3
        assert len(result["nodes_explored"]) == 3
        assert len(result["solve_statuses"]) == 3
        assert len(result["best_bounds"]) == 3
        assert len(result["best_solution_mst"]) == 3

    def test_output_keys_match_target_properties(
        self, solve_mip_func: Callable[..., Any]
    ) -> None:
        """Output dict keys must match the declared target property identifiers."""
        with patch(
            "cplex_mip_experiments.solve_mip._run_single_seed",
            side_effect=lambda **kw: self._make_seed_result(kw["seed"]),
        ):
            result = solve_mip_func(mps_file=DEFAULT_MPS, n_seeds=2, parallel=False)

        assert set(result.keys()) == {
            "solve_times",
            "objective_values",
            "mip_gaps",
            "nodes_explored",
            "solve_statuses",
            "best_bounds",
            "best_solution_mst",
            "progress_time_grid",
            "objective_over_time",
            "best_bound_over_time",
            "nodes_explored_over_time",
            "mip_gap_over_time",
        }

    def test_export_solution_false_returns_empty_mst_strings(
        self, solve_mip_func: Callable[..., Any]
    ) -> None:
        """With export_solution=False, aggregated best_solution_mst must be empty strings."""

        def capture(**kw: Any) -> dict[str, Any]:  # noqa: ANN401
            assert kw["export_solution"] is False
            return self._make_seed_result(kw["seed"])

        with patch(
            "cplex_mip_experiments.solve_mip._run_single_seed",
            side_effect=capture,
        ):
            result = solve_mip_func(
                mps_file=DEFAULT_MPS,
                n_seeds=2,
                export_solution=False,
                parallel=False,
            )

        assert result["best_solution_mst"] == ["", ""]

    def test_export_solution_true_passes_flag_to_run_single_seed(
        self, solve_mip_func: Callable[..., Any]
    ) -> None:
        """export_solution=True must be forwarded to _run_single_seed."""
        seen: list[bool] = []

        def capture(**kw: Any) -> dict[str, Any]:  # noqa: ANN401
            seen.append(kw["export_solution"])
            result = self._make_seed_result(kw["seed"])
            result["best_solution_mst"] = "<CPLEXSolution/>"
            return result

        with patch(
            "cplex_mip_experiments.solve_mip._run_single_seed",
            side_effect=capture,
        ):
            solve_mip_func(
                mps_file=DEFAULT_MPS,
                n_seeds=2,
                export_solution=True,
                parallel=False,
            )

        assert seen == [True, True]

    def test_seeds_are_zero_indexed_and_sequential(
        self, solve_mip_func: Callable[..., Any]
    ) -> None:
        """_run_single_seed must be called with seeds 0, 1, …, n_seeds-1."""
        called_seeds: list[int] = []

        def capture_seed(**kw: Any) -> dict[str, Any]:  # noqa: ANN401
            called_seeds.append(kw["seed"])
            return self._make_seed_result(kw["seed"])

        with patch(
            "cplex_mip_experiments.solve_mip._run_single_seed", side_effect=capture_seed
        ):
            solve_mip_func(mps_file=DEFAULT_MPS, n_seeds=4, parallel=False)

        assert called_seeds == [0, 1, 2, 3]

    def test_default_n_seeds_is_five(self, solve_mip_func: Callable[..., Any]) -> None:
        """Calling solve_mip with only mps_file must run exactly 5 seeds."""
        called_seeds: list[int] = []

        def capture_seed(**kw: Any) -> dict[str, Any]:  # noqa: ANN401
            called_seeds.append(kw["seed"])
            return self._make_seed_result(kw["seed"])

        with patch(
            "cplex_mip_experiments.solve_mip._run_single_seed", side_effect=capture_seed
        ):
            solve_mip_func(mps_file=DEFAULT_MPS, parallel=False)

        assert len(called_seeds) == 5

    def test_parallel_true_requires_ray(
        self, solve_mip_func: Callable[..., Any]
    ) -> None:
        """parallel=True must raise if Ray is not initialized."""
        with (
            patch("ray.is_initialized", return_value=False),
            pytest.raises(ValueError, match="parallel=True requires"),
        ):
            solve_mip_func(mps_file=DEFAULT_MPS, n_seeds=2, parallel=True)

    def test_parallel_ray_seed_failure_returns_partial_results(
        self, solve_mip_func: Callable[..., Any]
    ) -> None:
        """parallel=True must return full vectors when some Ray seed tasks fail."""
        refs = [object(), object(), object()]

        def fake_ray_get(ref: object) -> dict[str, Any]:
            index = refs.index(ref)
            if index == 1:
                raise RuntimeError("Failed to set up runtime environment")
            return self._make_seed_result(index)

        with (
            patch("ray.is_initialized", return_value=True),
            patch(
                "cplex_mip_experiments.solve_mip.estimate_mip_memory_bytes",
                return_value=4096,
            ),
            patch("ray.remote") as mock_remote,
            patch("ray.get", side_effect=fake_ray_get),
        ):
            mock_remote.return_value = lambda fn: type(
                "RemoteFn", (), {"remote": lambda _self, seed: refs[seed]}
            )()
            result = solve_mip_func(mps_file=DEFAULT_MPS, n_seeds=3, parallel=True)

        assert len(result["solve_times"]) == 3
        assert result["solve_statuses"][0] == "optimal"
        assert result["solve_statuses"][1].startswith("ray_task_failed:")
        assert result["solve_statuses"][2] == "optimal"
        assert result["objective_values"][1] is None
        assert result["nodes_explored"][1] == 0

    def test_progress_interval_produces_aligned_time_series(
        self, solve_mip_func: Callable[..., Any]
    ) -> None:
        """With progress_interval_s=60, output has aligned time-series."""

        def make_result_with_progress(**kw: Any) -> dict[str, Any]:  # noqa: ANN401
            seed = kw["seed"]
            base = self._make_seed_result(seed)
            # Simulate samples at t=0 and t=60
            base["progress_samples"] = [
                {
                    "elapsed": 0.0,
                    "best_objective": -100.0 - seed,
                    "best_bound": -90.0 - seed,
                    "nodes_explored": 0,
                    "mip_gap": 0.1,
                },
                {
                    "elapsed": 60.0,
                    "best_objective": -105.0 - seed,
                    "best_bound": -95.0 - seed,
                    "nodes_explored": 100 * (seed + 1),
                    "mip_gap": 0.05,
                },
            ]
            return base

        with patch(
            "cplex_mip_experiments.solve_mip._run_single_seed",
            side_effect=make_result_with_progress,
        ):
            result = solve_mip_func(
                mps_file=DEFAULT_MPS,
                n_seeds=2,
                progress_interval_s=60.0,
                time_limit_s=120.0,
                parallel=False,
            )

        assert result["progress_time_grid"] == [0.0, 60.0, 120.0]
        assert len(result["objective_over_time"]) == 2
        assert len(result["objective_over_time"][0]) == 3
        assert result["objective_over_time"][0][0] == -100.0
        assert result["objective_over_time"][0][1] == -105.0
        assert result["objective_over_time"][0][2] == -105.0  # forward-fill


class TestCollectParallelSeedResults:
    """``_collect_parallel_seed_results`` applies partial-OK policy for Ray tasks."""

    def _success(self, seed: int) -> dict[str, Any]:
        """Return a minimal successful single-seed result."""
        return {
            "solve_time_s": float(seed + 1),
            "objective_value": -1.0,
            "mip_gap": 0.01,
            "nodes_explored": 10,
            "solve_status": "optimal",
            "progress_samples": [],
        }

    def test_all_seeds_succeed(self) -> None:
        """All Ray refs returning results must be collected in order."""
        from cplex_mip_experiments.solve_mip import _collect_parallel_seed_results

        refs = ["ref-0", "ref-1"]
        with patch("ray.get", side_effect=[self._success(0), self._success(1)]):
            results = _collect_parallel_seed_results(refs)

        assert len(results) == 2
        assert results[0]["solve_status"] == "optimal"
        assert results[1]["solve_status"] == "optimal"

    def test_one_seed_ray_failure_substitutes_structured_failure(self) -> None:
        """A failed Ray ref must not prevent collecting other seed results."""
        from cplex_mip_experiments.solve_mip import _collect_parallel_seed_results

        refs = ["ref-0", "ref-1", "ref-2"]

        def fake_get(ref: str) -> dict[str, Any]:
            if ref == "ref-1":
                raise RuntimeError("runtime env timeout")
            return self._success(refs.index(ref))

        with patch("ray.get", side_effect=fake_get):
            results = _collect_parallel_seed_results(refs)

        assert len(results) == 3
        assert results[0]["solve_status"] == "optimal"
        assert results[1]["solve_status"] == "ray_task_failed: runtime env timeout"
        assert results[1]["objective_value"] is None
        assert results[2]["solve_status"] == "optimal"

    def test_all_seeds_ray_failure_returns_failure_vectors(self) -> None:
        """When every Ray ref fails, output vectors still match n_seeds."""
        from cplex_mip_experiments.solve_mip import _collect_parallel_seed_results

        refs = ["ref-0", "ref-1"]
        with patch("ray.get", side_effect=RuntimeError("worker died")):
            results = _collect_parallel_seed_results(refs)

        assert len(results) == 2
        assert all(r["solve_status"] == "ray_task_failed: worker died" for r in results)
        assert all(r["objective_value"] is None for r in results)


class TestApplyCutPassesAll:
    """``_apply_cut_passes_all`` caps aggressiveness per CPLEX cut family."""

    def test_level_three_capped_to_two_for_gomory(self) -> None:
        """Families with max 2 must not receive level 3."""
        from unittest.mock import MagicMock

        from cplex_mip_experiments.solve_mip import _apply_cut_passes_all

        model = MagicMock()
        _apply_cut_passes_all(model, 3)
        model.parameters.mip.cuts.gomory.set.assert_called_once_with(2)
        model.parameters.mip.cuts.cliques.set.assert_called_once_with(3)

    def test_negative_one_unclipped(self) -> None:
        """-1 (off) must pass through for all families."""
        from unittest.mock import MagicMock

        from cplex_mip_experiments.solve_mip import _apply_cut_passes_all

        model = MagicMock()
        _apply_cut_passes_all(model, -1)
        model.parameters.mip.cuts.zerohalfcut.set.assert_called_once_with(-1)


class TestBuildTimeGridAndAlign:
    """``progress_time_grid`` must extend far enough for the latest sample timestamp."""

    def test_grid_last_point_covers_non_multiple_max_elapsed(self) -> None:
        """If max_elapsed is not on the interval lattice, extend past it."""
        from cplex_mip_experiments.solve_mip import _build_time_grid

        grid = _build_time_grid(60.0, 1e75, 95.0)
        assert grid[-1] >= 95.0
        assert grid == [0.0, 60.0, 120.0]

    def test_grid_extends_when_max_elapsed_exceeds_time_limit(self) -> None:
        """Wall clock can slightly exceed TILIM; grid must still cover terminal sample."""
        from cplex_mip_experiments.solve_mip import _build_time_grid

        grid = _build_time_grid(60.0, 3600.0, 3655.0)
        assert grid[-1] >= 3655.0

    def test_align_ingests_terminal_when_elapsed_not_on_grid(self) -> None:
        """Last grid point t must satisfy t >= last sample elapsed."""
        from cplex_mip_experiments.solve_mip import _align_to_grid, _build_time_grid

        samples = [
            {
                "elapsed": 0.0,
                "best_objective": 1.0,
                "best_bound": 0.0,
                "nodes_explored": 0,
                "mip_gap": None,
            },
            {
                "elapsed": 95.0,
                "best_objective": -16.0,
                "best_bound": -16.0,
                "nodes_explored": 1,
                "mip_gap": 0.0,
            },
        ]
        grid = _build_time_grid(60.0, 1e75, 95.0)
        aligned = _align_to_grid(samples, grid)
        assert aligned["best_bound"][-1] == -16.0
        assert aligned["best_objective"][-1] == -16.0

    def test_align_preserves_bound_when_terminal_sample_has_none_bound(self) -> None:
        """Terminal sample with None best_bound must not erase the last known bound."""
        from cplex_mip_experiments.solve_mip import _align_to_grid, _build_time_grid

        last_callback_bound = 0.0036462444963870537
        samples = [
            {
                "elapsed": 7140.0,
                "best_objective": 0.005212002548830863,
                "best_bound": last_callback_bound,
                "nodes_explored": 100,
                "mip_gap": 0.01,
            },
            {
                "elapsed": 7208.16956991516,
                "best_objective": 0.005212002548830863,
                "best_bound": None,
                "nodes_explored": 200,
                "mip_gap": None,
            },
        ]
        grid = _build_time_grid(60.0, 7200.0, 7208.16956991516)
        aligned = _align_to_grid(samples, grid)
        assert aligned["best_bound"][-3:] == [
            last_callback_bound,
            last_callback_bound,
            last_callback_bound,
        ]
        assert aligned["best_objective"][-1] == 0.005212002548830863


@pytest.mark.skipif(not HAS_CPLEX, reason="CPLEX Python API not installed")
class TestCommunityEditionLimits:
    """When CPLEX Community Edition limits (error 1016) are hit, the experiment must raise."""

    def test_error_1016_raises_for_invalid_result(self) -> None:
        """_run_single_seed must re-raise CplexSolverError 1016 so InvalidMeasurementResult is produced."""
        from unittest.mock import MagicMock, patch

        import cplex
        from cplex_mip_experiments.solve_mip import _run_single_seed

        mock_model = MagicMock()
        mock_model.read = MagicMock()
        mock_model.parameters.randomseed.set = MagicMock()
        mock_model.parameters.threads.set = MagicMock()
        mock_model.parameters.emphasis.mip.set = MagicMock()
        mock_model.parameters.mip.strategy.nodeselect.set = MagicMock()
        mock_model.parameters.mip.strategy.variableselect.set = MagicMock()
        mock_model.parameters.mip.strategy.heuristicfreq.set = MagicMock()
        mock_model.parameters.mip.strategy.rinsheur.set = MagicMock()
        mock_model.parameters.mip.limits.cutpasses.set = MagicMock()
        mock_model.parameters.timelimit.set = MagicMock()
        mock_model.solve.side_effect = cplex.exceptions.CplexSolverError(
            "Community Edition. Problem size limits exceeded.",
            None,
            1016,
        )

        with (
            patch.object(cplex, "Cplex", return_value=mock_model),
            pytest.raises(cplex.exceptions.CplexSolverError) as exc_info,
        ):
            _run_single_seed(
                mps_file=DEFAULT_MPS,
                seed=1,
                seed_index=1,
                n_seeds=3,
                node_selection=1,
                variable_selection=0,
                heuristic_frequency=0,
                time_limit_s=10.0,
                n_threads=1,
                rins_frequency=0,
                cut_passes=0,
                mip_emphasis=0,
            )
        assert exc_info.value.args[2] == 1016

    def test_other_cplex_errors_return_none_result(self) -> None:
        """Non-1016 CPLEX errors must return a structured result with None values, not raise."""
        from unittest.mock import MagicMock, patch

        import cplex
        from cplex_mip_experiments.solve_mip import _run_single_seed

        mock_model = MagicMock()
        mock_model.read = MagicMock()
        mock_model.parameters.randomseed.set = MagicMock()
        mock_model.parameters.threads.set = MagicMock()
        mock_model.parameters.emphasis.mip.set = MagicMock()
        mock_model.parameters.mip.strategy.nodeselect.set = MagicMock()
        mock_model.parameters.mip.strategy.variableselect.set = MagicMock()
        mock_model.parameters.mip.strategy.heuristicfreq.set = MagicMock()
        mock_model.parameters.mip.strategy.rinsheur.set = MagicMock()
        mock_model.parameters.mip.limits.cutpasses.set = MagicMock()
        mock_model.parameters.timelimit.set = MagicMock()
        mock_model.solve.side_effect = cplex.exceptions.CplexSolverError(
            "Some other CPLEX error",
            None,
            1234,
        )

        with patch.object(cplex, "Cplex", return_value=mock_model):
            result = _run_single_seed(
                mps_file=DEFAULT_MPS,
                seed=0,
                seed_index=0,
                n_seeds=1,
                node_selection=1,
                variable_selection=0,
                heuristic_frequency=0,
                time_limit_s=10.0,
                n_threads=1,
                rins_frequency=0,
                cut_passes=0,
                mip_emphasis=0,
            )
        assert result["solve_status"] == "cplex_error_1234"
        assert result["objective_value"] is None
        assert result["mip_gap"] is None
        assert result["best_bound"] is None
        assert result["best_solution_mst"] == ""


class TestBestBoundSentinel:
    def test_normalize_sentinel_values(self) -> None:
        """CPLEX sentinel magnitudes must be treated as missing bounds/objectives."""
        from cplex_mip_experiments.solve_mip import _normalize_cplex_value

        assert _normalize_cplex_value(1e75) is None
        assert _normalize_cplex_value(-1e75) is None
        assert _normalize_cplex_value(-16.0) == -16.0
        assert _normalize_cplex_value(None) is None


class TestAppendTerminalProgressSample:
    """``_append_terminal_progress_sample`` must set best_bound correctly."""

    def test_optimal_uses_objective_value(self) -> None:
        """When mip_gap == 0.0, best_bound must be set to objective_value."""
        from unittest.mock import MagicMock

        from cplex_mip_experiments.solve_mip import _append_terminal_progress_sample

        samples: list[dict] = []
        _append_terminal_progress_sample(
            model=MagicMock(),
            progress_samples=samples,
            solve_time=12.5,
            objective_value=0.005212,
            mip_gap=0.0,
            nodes_explored=500,
        )
        assert len(samples) == 1
        assert samples[-1]["best_bound"] == 0.005212
        assert samples[-1]["best_objective"] == 0.005212

    def test_non_optimal_forward_fills_last_callback_bound(self) -> None:
        """When mip_gap != 0.0, best_bound must be the last callback-recorded bound."""
        from unittest.mock import MagicMock

        from cplex_mip_experiments.solve_mip import _append_terminal_progress_sample

        last_callback_bound = 0.003623
        samples: list[dict] = [
            {
                "elapsed": 60.0,
                "best_objective": 0.005212,
                "best_bound": last_callback_bound,
                "nodes_explored": 100,
                "mip_gap": 0.305,
            }
        ]
        _append_terminal_progress_sample(
            model=MagicMock(),
            progress_samples=samples,
            solve_time=120.0,
            objective_value=0.005212,
            mip_gap=0.305,
            nodes_explored=200,
        )
        assert samples[-1]["best_bound"] == last_callback_bound

    def test_non_optimal_does_not_use_post_solve_api_when_callback_samples_exist(
        self,
    ) -> None:
        """When callback samples are present, post-solve API must not be called."""
        from unittest.mock import MagicMock

        from cplex_mip_experiments.solve_mip import _append_terminal_progress_sample

        model = MagicMock()
        samples: list[dict] = [
            {
                "elapsed": 60.0,
                "best_objective": 0.005212,
                "best_bound": 0.003623,
                "nodes_explored": 100,
                "mip_gap": 0.3,
            }
        ]
        _append_terminal_progress_sample(
            model=model,
            progress_samples=samples,
            solve_time=120.0,
            objective_value=0.005212,
            mip_gap=0.3,
            nodes_explored=200,
        )
        model.solution.MIP.get_best_objective_value.assert_not_called()

    def test_non_optimal_no_callback_samples_uses_api_fallback(self) -> None:
        """Without callback samples, post-solve API is used as best-effort fallback."""
        import sys
        from unittest.mock import MagicMock

        from cplex_mip_experiments.solve_mip import _append_terminal_progress_sample

        model = MagicMock()
        model.solution.MIP.get_best_objective_value.return_value = 0.003800

        # Provide a stub cplex module so the lazy import inside the fallback path
        # succeeds even when the real CPLEX package is not installed.
        mock_cplex = MagicMock()
        mock_cplex.exceptions.CplexSolverError = Exception
        samples: list[dict] = []
        with patch.dict(sys.modules, {"cplex": mock_cplex}):
            _append_terminal_progress_sample(
                model=model,
                progress_samples=samples,
                solve_time=120.0,
                objective_value=0.005212,
                mip_gap=0.27,
                nodes_explored=50,
            )
        assert samples[-1]["best_bound"] == 0.003800

    def test_non_optimal_none_mip_gap_forward_fills(self) -> None:
        """None mip_gap must not trigger the optimality branch."""
        from unittest.mock import MagicMock

        from cplex_mip_experiments.solve_mip import _append_terminal_progress_sample

        last_bound = 0.0036
        samples: list[dict] = [
            {
                "elapsed": 60.0,
                "best_objective": 0.005212,
                "best_bound": last_bound,
                "nodes_explored": 100,
                "mip_gap": None,
            }
        ]
        _append_terminal_progress_sample(
            model=MagicMock(),
            progress_samples=samples,
            solve_time=120.0,
            objective_value=0.005212,
            mip_gap=None,
            nodes_explored=200,
        )
        assert samples[-1]["best_bound"] == last_bound

    def test_scalar_best_bound_from_time_limited_solve(
        self, solve_mip_func: Callable[..., Any]
    ) -> None:
        """For a time-limited run, best_bounds must reflect the LP-relaxation bound
        from the last callback sample, not the incumbent objective value."""
        last_callback_bound = 0.003623

        def make_time_limited_result(**kw: object) -> dict[str, Any]:
            return {
                "solve_time_s": 120.0,
                "objective_value": 0.005212,
                "best_bound": last_callback_bound,
                "mip_gap": 0.305,
                "nodes_explored": 500,
                "solve_status": "time limit exceeded",
                "best_solution_mst": "",
                "progress_samples": [
                    {
                        "elapsed": 60.0,
                        "best_objective": 0.005212,
                        "best_bound": last_callback_bound,
                        "nodes_explored": 300,
                        "mip_gap": 0.305,
                    }
                ],
            }

        with patch(
            "cplex_mip_experiments.solve_mip._run_single_seed",
            side_effect=make_time_limited_result,
        ):
            result = solve_mip_func(
                mps_file=DEFAULT_MPS,
                n_seeds=2,
                progress_interval_s=60.0,
                time_limit_s=120.0,
                parallel=False,
            )

        for bound, obj in zip(
            result["best_bounds"], result["objective_values"], strict=True
        ):
            assert bound == last_callback_bound
            assert bound != obj


@pytest.mark.skipif(not HAS_CPLEX, reason="CPLEX Python API not installed")
class TestRunSingleSeedIntegration:
    def test_run_single_seed_returns_expected_keys(self) -> None:
        """_run_single_seed must return all expected keys including progress_samples."""
        from cplex_mip_experiments.solve_mip import _run_single_seed

        result = _run_single_seed(
            mps_file=DEFAULT_MPS,
            seed=0,
            seed_index=0,
            n_seeds=1,
            node_selection=1,
            variable_selection=0,
            heuristic_frequency=0,
            time_limit_s=10.0,
            n_threads=1,
            rins_frequency=0,
            cut_passes=0,
        )
        assert set(result.keys()) == {
            "solve_time_s",
            "objective_value",
            "best_bound",
            "mip_gap",
            "nodes_explored",
            "solve_status",
            "best_solution_mst",
            "progress_samples",
        }

    def test_run_single_seed_solve_time_is_positive(self) -> None:
        """Solve time must be a positive float."""
        from cplex_mip_experiments.solve_mip import _run_single_seed

        result = _run_single_seed(
            mps_file=DEFAULT_MPS,
            seed=0,
            seed_index=0,
            n_seeds=1,
            node_selection=1,
            variable_selection=0,
            heuristic_frequency=0,
            time_limit_s=10.0,
            n_threads=1,
            rins_frequency=0,
            cut_passes=0,
        )
        assert isinstance(result["solve_time_s"], float)
        assert result["solve_time_s"] > 0.0

    def test_run_single_seed_populates_best_bound(self) -> None:
        """_run_single_seed must return a numeric best_bound on successful solve."""
        from cplex_mip_experiments.solve_mip import _run_single_seed

        result = _run_single_seed(
            mps_file=DEFAULT_MPS,
            seed=0,
            seed_index=0,
            n_seeds=1,
            node_selection=1,
            variable_selection=0,
            heuristic_frequency=0,
            time_limit_s=10.0,
            n_threads=1,
            rins_frequency=0,
            cut_passes=0,
        )
        assert result["best_bound"] is not None
        assert isinstance(result["best_bound"], float)

    def test_export_solution_false_yields_empty_mst(self) -> None:
        """export_solution=False must not populate best_solution_mst."""
        from cplex_mip_experiments.solve_mip import _run_single_seed

        result = _run_single_seed(
            mps_file=DEFAULT_MPS,
            seed=0,
            seed_index=0,
            n_seeds=1,
            node_selection=1,
            variable_selection=0,
            heuristic_frequency=0,
            time_limit_s=10.0,
            n_threads=1,
            rins_frequency=0,
            cut_passes=0,
            export_solution=False,
        )
        assert result["best_solution_mst"] == ""

    def test_export_solution_true_yields_mst_xml(self) -> None:
        """export_solution=True must export warm-start-ready MST XML."""
        from cplex_mip_experiments.solve_mip import _run_single_seed

        result = _run_single_seed(
            mps_file=DEFAULT_MPS,
            seed=0,
            seed_index=0,
            n_seeds=1,
            node_selection=1,
            variable_selection=0,
            heuristic_frequency=0,
            time_limit_s=10.0,
            n_threads=1,
            rins_frequency=0,
            cut_passes=0,
            export_solution=True,
        )
        assert result["best_solution_mst"]
        assert "CPLEXSolution" in result["best_solution_mst"]

    def test_warm_start_round_trip(self, tmp_path: pathlib.Path) -> None:
        """Exported MST must be loadable as a warm start for a second solve."""
        from cplex_mip_experiments.solve_mip import _run_single_seed

        first = _run_single_seed(
            mps_file=DEFAULT_MPS,
            seed=0,
            seed_index=0,
            n_seeds=1,
            node_selection=1,
            variable_selection=0,
            heuristic_frequency=0,
            time_limit_s=10.0,
            n_threads=1,
            rins_frequency=0,
            cut_passes=0,
            export_solution=True,
        )
        mst_path = tmp_path / "warm.mst"
        mst_path.write_text(first["best_solution_mst"], encoding="utf-8")

        _run_single_seed(
            mps_file=DEFAULT_MPS,
            seed=1,
            seed_index=0,
            n_seeds=1,
            node_selection=1,
            variable_selection=0,
            heuristic_frequency=0,
            time_limit_s=10.0,
            n_threads=1,
            rins_frequency=0,
            cut_passes=0,
            warm_start_file=str(mst_path),
        )


class TestEstimateMemoryBytes:
    """``estimate_mip_memory_bytes`` implements the power-law formula."""

    def _call(self, file_size_bytes: int) -> int:
        from cplex_mip_experiments.solve_mip import estimate_mip_memory_bytes

        with patch("os.path.getsize", return_value=file_size_bytes):
            return estimate_mip_memory_bytes("/fake/file.mps")

    def test_returns_int(self) -> None:
        """Return value must be an int."""
        result = self._call(0)
        assert isinstance(result, int)

    def test_floor_applied_for_zero_byte_file(self) -> None:
        """A zero-byte file must still return at least floor_gb * 1.20 bytes."""
        floor_gb = 4.0
        safety = 1.20
        min_expected = int(floor_gb * safety * (1024**3))
        assert self._call(0) >= min_expected

    def test_floor_applied_for_tiny_file(self) -> None:
        """A 1-byte file is dominated by the floor; result is within 1% of zero-byte case."""
        zero_byte = self._call(0)
        one_byte = self._call(1)
        assert abs(one_byte - zero_byte) / zero_byte < 0.01

    def test_monotonic_larger_file_larger_estimate(self) -> None:
        """A 100 MB file must produce a larger estimate than a 1 MB file."""
        small = self._call(1 * 1024**2)
        large = self._call(100 * 1024**2)
        assert large > small

    def test_sub_linear_growth(self) -> None:
        """Doubling the file size must not double the estimate (power-law damping)."""
        base = self._call(10 * 1024**2)
        doubled = self._call(20 * 1024**2)
        assert doubled < base * 2

    def test_known_value_23mb(self) -> None:
        """23 MB file: estimate must be substantially less than the old linear ~46 GB."""
        result_bytes = self._call(23 * 1024**2)
        result_gb = result_bytes / (1024**3)
        # Old formula gave ~46 GB; new formula should be well below 30 GB
        assert result_gb < 30.0
        # Must still be above the 4 GB floor
        assert result_gb > 4.0


class TestRunSingleSeedCplexMemoryParams:
    """``_run_single_seed`` sets WorkMem and NodeFileInd when workmem_mb > 0."""

    def _make_mock_model(self) -> MagicMock:
        """Return a MagicMock that stands in for a cplex.Cplex() model."""
        model = MagicMock()
        model.solution.get_status_string.return_value = "optimal"
        model.solution.progress.get_num_nodes_processed.return_value = 10
        model.solution.get_objective_value.return_value = -1.0
        model.solution.MIP.get_mip_relative_gap.return_value = 0.0
        return model

    @pytest.mark.skipif(not HAS_CPLEX, reason="CPLEX Python API not installed")
    def test_workmem_set_when_workmem_mb_positive(self) -> None:
        """model.parameters.workmem.set must be called with float(workmem_mb)."""
        import cplex
        from cplex_mip_experiments.solve_mip import _run_single_seed

        mock_model = self._make_mock_model()
        with patch.object(cplex, "Cplex", return_value=mock_model):
            _run_single_seed(
                mps_file=DEFAULT_MPS,
                seed=0,
                seed_index=0,
                n_seeds=1,
                node_selection=1,
                variable_selection=0,
                heuristic_frequency=0,
                time_limit_s=10.0,
                n_threads=1,
                rins_frequency=0,
                cut_passes=0,
                workmem_mb=8192,
            )
        mock_model.parameters.workmem.set.assert_called_once_with(float(8192))

    @pytest.mark.skipif(not HAS_CPLEX, reason="CPLEX Python API not installed")
    def test_nodefile_set_when_workmem_mb_positive(self) -> None:
        """model.parameters.mip.strategy.file.set must be called with 3."""
        import cplex
        from cplex_mip_experiments.solve_mip import _run_single_seed

        mock_model = self._make_mock_model()
        with patch.object(cplex, "Cplex", return_value=mock_model):
            _run_single_seed(
                mps_file=DEFAULT_MPS,
                seed=0,
                seed_index=0,
                n_seeds=1,
                node_selection=1,
                variable_selection=0,
                heuristic_frequency=0,
                time_limit_s=10.0,
                n_threads=1,
                rins_frequency=0,
                cut_passes=0,
                workmem_mb=8192,
            )
        mock_model.parameters.mip.strategy.file.set.assert_called_once_with(3)

    @pytest.mark.skipif(not HAS_CPLEX, reason="CPLEX Python API not installed")
    def test_workmem_not_set_when_workmem_mb_zero(self) -> None:
        """Neither workmem nor nodefile must be set when workmem_mb=0 (default)."""
        import cplex
        from cplex_mip_experiments.solve_mip import _run_single_seed

        mock_model = self._make_mock_model()
        with patch.object(cplex, "Cplex", return_value=mock_model):
            _run_single_seed(
                mps_file=DEFAULT_MPS,
                seed=0,
                seed_index=0,
                n_seeds=1,
                node_selection=1,
                variable_selection=0,
                heuristic_frequency=0,
                time_limit_s=10.0,
                n_threads=1,
                rins_frequency=0,
                cut_passes=0,
                workmem_mb=0,
            )
        mock_model.parameters.workmem.set.assert_not_called()
        mock_model.parameters.mip.strategy.file.set.assert_not_called()


class TestSolveMipWorkmemPropagation:
    """``solve_mip`` derives workmem_mb from estimate and passes it to _run_single_seed."""

    def _make_seed_result(self) -> dict[str, Any]:
        return {
            "solve_time_s": 1.0,
            "objective_value": -1.0,
            "best_bound": -1.0,
            "mip_gap": 0.0,
            "nodes_explored": 10,
            "solve_status": "optimal",
            "best_solution_mst": "",
            "progress_samples": [],
        }

    def test_serial_path_passes_workmem_mb_to_run_single_seed(
        self, solve_mip_func: Callable[..., Any]
    ) -> None:
        """Serial solve_mip must pass workmem_mb=80% of estimate to _run_single_seed."""
        captured_workmem: list[int] = []

        def capture(**kw: Any) -> dict[str, Any]:  # noqa: ANN401
            captured_workmem.append(kw.get("workmem_mb", -1))
            return self._make_seed_result()

        fake_mem_bytes = 10 * 1024**3  # 10 GB

        with (
            patch(
                "cplex_mip_experiments.solve_mip.estimate_mip_memory_bytes",
                return_value=fake_mem_bytes,
            ),
            patch(
                "cplex_mip_experiments.solve_mip._run_single_seed",
                side_effect=capture,
            ),
        ):
            solve_mip_func(mps_file=DEFAULT_MPS, n_seeds=2, parallel=False)

        expected_workmem_mb = int(fake_mem_bytes / (1024**2) * 0.80)
        assert len(captured_workmem) == 2
        assert all(w == expected_workmem_mb for w in captured_workmem)
