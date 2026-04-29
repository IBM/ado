# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest

from orchestrator.schema.experiment import Experiment

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
        """All twelve optional properties must be defined."""
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
        from orchestrator.schema.domain import VariableTypeEnum

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
        """cut_passes domain must contain exactly [-1, 0, 1, 5]."""
        prop = next(
            p for p in experiment.optionalProperties if p.identifier == "cut_passes"
        )
        assert sorted(prop.propertyDomain.values) == [-1, 0, 1, 5]


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
            "mip_gap": 0.01 * seed,
            "nodes_explored": 100 * (seed + 1),
            "solve_status": "optimal",
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
            "progress_time_grid",
            "objective_over_time",
            "best_bound_over_time",
            "nodes_explored_over_time",
            "mip_gap_over_time",
        }

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
            "mip_gap",
            "nodes_explored",
            "solve_status",
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


class TestLHSSampler:
    """Unit tests for LatinHypercubeSampler and its LHS design helpers."""

    def test_lhs_points_length(self) -> None:
        """_lhs_points must return exactly n_samples dicts."""
        import numpy as np
        from cplex_mip_experiments.lhs_sampler import _lhs_points

        rng = np.random.default_rng(0)
        pts = _lhs_points(["a", "b"], {"a": [1, 2, 3], "b": [10, 20]}, 6, rng)
        assert len(pts) == 6

    def test_lhs_points_keys(self) -> None:
        """Each design point must have exactly the requested property keys."""
        import numpy as np
        from cplex_mip_experiments.lhs_sampler import _lhs_points

        names = ["x", "y"]
        rng = np.random.default_rng(1)
        pts = _lhs_points(names, {"x": [0, 1], "y": [0, 1, 2]}, 6, rng)
        for pt in pts:
            assert set(pt.keys()) == set(names)

    def test_lhs_points_values_in_domain(self) -> None:
        """All values in each design point must be drawn from the domain."""
        import numpy as np
        from cplex_mip_experiments.lhs_sampler import _lhs_points

        dim_values = {"a": [1, 2, 4, 8], "b": [-1, 0, 5, 25, 100]}
        rng = np.random.default_rng(2)
        pts = _lhs_points(list(dim_values), dim_values, 20, rng)
        for pt in pts:
            for name, val in pt.items():
                assert val in dim_values[name]

    def test_lhs_points_total_equals_n_samples(self) -> None:
        """_lhs_points must return exactly n_samples points."""
        import numpy as np
        from cplex_mip_experiments.lhs_sampler import _lhs_points

        dim_values = {"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]}
        rng = np.random.default_rng(0)
        pts = _lhs_points(list(dim_values), dim_values, 8, rng)
        assert len(pts) == 8

    def test_lhs_parameters_model(self) -> None:
        """parameters_model() must return LHSSamplerParameters."""
        from cplex_mip_experiments.lhs_sampler import (
            LatinHypercubeSampler,
            LHSSamplerParameters,
        )

        assert LatinHypercubeSampler.parameters_model() is LHSSamplerParameters
