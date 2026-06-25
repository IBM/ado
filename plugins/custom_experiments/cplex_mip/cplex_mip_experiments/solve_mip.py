# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import logging
import os
import pathlib
import sys
import tempfile
import time
from typing import Any, Literal

from orchestrator.modules.actuators.custom_experiments import custom_experiment
from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum
from orchestrator.schema.property import ConstitutiveProperty

logger = logging.getLogger(__name__)

CutPassesAllLevel = Literal["cplex_default", -1, 0, 1, 2, 3]

MpsFile = ConstitutiveProperty(
    identifier="mps_file",
    metadata={
        "description": (
            "Path to the MPS instance file to solve (.mps or .mps.gz). "
            "Example: a MIPLIB benchmark such as bab6.mps.gz."
        )
    },
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE,
        values=["/path/to/instance.mps.gz"],
    ),
)

NSeeds = ConstitutiveProperty(
    identifier="n_seeds",
    metadata={
        "description": (
            "Number of random seeds to use. CPLEX is run once per seed "
            "with seeds 0, 1, …, n_seeds-1. Output properties are vectors "
            "of length n_seeds, one element per seed run."
        )
    },
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        domainRange=[1, 100],
        interval=1,
    ),
)

NodeSelection = ConstitutiveProperty(
    identifier="node_selection",
    metadata={
        "description": (
            "CPLEX node selection strategy (CPX_PARAM_NODESEL): "
            "0=depth-first, 1=best-bound (default), "
            "2=best-estimate, 3=best-estimate-alternative."
        )
    },
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        values=[0, 1, 2, 3],
    ),
)

VariableSelection = ConstitutiveProperty(
    identifier="variable_selection",
    metadata={
        "description": (
            "CPLEX branching variable selection strategy (CPX_PARAM_VARSEL): "
            "-1=minimum infeasibility, 0=automatic (default), "
            "1=maximum infeasibility, 2=pseudo-cost, "
            "3=strong branching, 4=pseudo-reduced-cost."
        )
    },
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        values=[-1, 0, 1, 2, 3, 4],
    ),
)

HeuristicFrequency = ConstitutiveProperty(
    identifier="heuristic_frequency",
    metadata={
        "description": (
            "CPLEX MIP heuristic application frequency (CPX_PARAM_HEURFREQ): "
            "-1=none, 0=automatic (default), n=apply every n nodes."
        )
    },
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        values=[-1, 0, 10, 50, 100],
    ),
)

TimeLimit = ConstitutiveProperty(
    identifier="time_limit_s",
    metadata={
        "description": (
            "CPLEX time limit per seed run in seconds (CPX_PARAM_TILIM). "
            "Default is 1e75 (no limit); CPLEX runs until the optimal solution is found. "
            "Any positive value up to and including 1e75 is accepted."
        )
    },
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        # Upper bound is strictly above 1e75: float ULP makes (1e75 + 1) == 1e75, which
        # breaks ``value < max(domainRange)`` validation for the default 1e75.
        domainRange=[0, 1e76],
    ),
)

NThreads = ConstitutiveProperty(
    identifier="n_threads",
    metadata={
        "description": (
            "Number of parallel threads for CPLEX B&B (CPX_PARAM_THREADS). "
            "1=single-threaded (fully deterministic), 2/4/8=parallel. "
            "With n_threads>1 results may vary across runs even with the same seed."
        )
    },
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        values=[1, 2, 4, 8],
    ),
)

RinsFrequency = ConstitutiveProperty(
    identifier="rins_frequency",
    metadata={
        "description": (
            "Frequency of RINS (Relaxation Induced Neighborhood Search) heuristic "
            "(CPX_PARAM_RINSHEUR): -1=disabled, 0=automatic, n=apply every n nodes."
        )
    },
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        values=[-1, 0, 5, 25, 100],
    ),
)

CutPasses = ConstitutiveProperty(
    identifier="cut_passes",
    metadata={
        "description": (
            "Maximum number of cutting-plane passes at the root node "
            "(CPX_PARAM_CUTPASSES): -1=no cuts, 0=automatic, n=at most n passes."
        )
    },
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        domainRange=[-1, 201],
        interval=1,
    ),
)

CutPassesAll = ConstitutiveProperty(
    identifier="cut_passes_all",
    metadata={
        "description": (
            "Uniform aggressiveness for every CPLEX MIP cut family (interactive "
            "`set mip cuts all …`). Each `parameters.mip.cuts.*` value is set to "
            "this level capped by that family's maximum (-1 off, 0 automatic, "
            "1 moderate, 2 aggressive; some families support 3 very aggressive). "
            "`cplex_default` does not change cut parameters (solver defaults)."
        )
    },
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE,
        values=["cplex_default", -1, 0, 1, 2, 3],
    ),
)

MipEmphasis = ConstitutiveProperty(
    identifier="mip_emphasis",
    metadata={
        "description": (
            "MIP optimization emphasis (CPX_PARAM_MIPEMPHASIS / "
            "`set mip emphasis mip`): 0=balanced optimality and feasibility, "
            "1=integer feasibility, 2=optimality, 3=best bound, "
            "4=hidden feasible solutions, 5=heuristic."
        )
    },
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
        values=[0, 1, 2, 3, 4, 5],
    ),
)

Parallel = ConstitutiveProperty(
    identifier="parallel",
    metadata={
        "description": (
            "If True, run each of the n_seeds solver instances as a Ray remote task. "
            "If False, run all seeds in serial. Ray failures on individual seeds "
            "produce null metrics and a ray_task_failed solve_status for that seed "
            "without failing the whole measurement (partial-OK policy)."
        )
    },
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.BINARY_VARIABLE_TYPE,
        values=[False, True],
    ),
)

WarmStartFile = ConstitutiveProperty(
    identifier="warm_start_file",
    metadata={
        "description": (
            "Path to a CPLEX MIP-start file (.mst, or .sol with the same XML structure). "
            "Empty string disables warm start. Applied before solve on every seed run. "
            "For remote execution, use a bare filename and ship the file via "
            "execution context additionalFiles."
        )
    },
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE,
        values=["", "/path/to/warm_start.mst"],
    ),
)

ExportSolution = ConstitutiveProperty(
    identifier="export_solution",
    metadata={
        "description": (
            "If True, export the incumbent as MST XML in best_solution_mst. "
            "If False (default), best_solution_mst is still returned but each "
            "seed element is an empty string (no export work, no storage bloat)."
        )
    },
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.BINARY_VARIABLE_TYPE,
        values=[False, True],
    ),
)

ProgressInterval = ConstitutiveProperty(
    identifier="progress_interval_s",
    metadata={
        "description": (
            "Interval in seconds for capturing intermediate MIP progress metrics. "
            "When > 0, a callback records best_objective, best_bound, nodes_explored, "
            "and mip_gap at each interval. Outputs progress_time_grid and aligned "
            "time-series (objective_over_time, etc.). When 0, progress capture is disabled."
        )
    },
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[0, 86400],  # 0 = disabled, up to 24h
    ),
)

# Max aggressiveness level per ``parameters.mip.cuts.*`` family (CPLEX 22.1 API).
_MIP_CUT_FAMILY_MAX_LEVEL: dict[str, int] = {
    "bqp": 3,
    "cliques": 3,
    "covers": 3,
    "disjunctive": 3,
    "flowcovers": 2,
    "gomory": 2,
    "gubcovers": 2,
    "implied": 2,
    "liftproj": 3,
    "localimplied": 3,
    "mcfcut": 2,
    "mircut": 2,
    "nodecuts": 3,
    "pathcut": 2,
    "rlt": 3,
    "zerohalfcut": 2,
}


def _apply_cut_passes_all(model: object, level: int) -> None:
    """Set every MIP cut family to ``level``, capped by that family's maximum."""
    cuts = model.parameters.mip.cuts
    for name, max_level in _MIP_CUT_FAMILY_MAX_LEVEL.items():
        capped = min(max(level, -1), max_level)
        getattr(cuts, name).set(capped)


# Sentinel for "no incumbent" from CPLEX (e.g. 1e75).
_NO_INCUMBENT_SENTINEL = 1e70


def _normalize_cplex_value(value: float | None) -> float | None:
    """Return ``None`` when CPLEX uses a large sentinel for a missing value."""
    if value is None:
        return None
    if abs(value) >= _NO_INCUMBENT_SENTINEL:
        return None
    return float(value)


def _load_warm_start(model: object, warm_start_file: str) -> str | None:
    """Load a MIP start from disk. Return an error status string on failure."""
    if not warm_start_file:
        return None
    if not pathlib.Path(warm_start_file).is_file():
        return f"warm_start_file_not_found: {warm_start_file}"
    try:
        model.MIP_starts.read(warm_start_file)
        model.parameters.advance.set(1)
    except Exception as exc:  # noqa: BLE001
        return f"warm_start_read_error: {exc}"
    return None


def _export_incumbent_mst(model: object, objective_value: float | None) -> str:
    """Export the incumbent as warm-start-ready MST XML, with SOL fallback."""
    import cplex

    if _normalize_cplex_value(objective_value) is None:
        return ""

    try:
        values = model.solution.get_values()
    except cplex.exceptions.CplexSolverError:
        return ""

    if not values:
        return ""

    try:
        model.MIP_starts.delete()
        model.MIP_starts.add(values)
        with tempfile.NamedTemporaryFile(suffix=".mst", delete=False) as handle:
            temp_path = handle.name
        try:
            model.MIP_starts.write(temp_path)
            return pathlib.Path(temp_path).read_text(encoding="utf-8")
        finally:
            os.unlink(temp_path)
    except Exception:  # noqa: BLE001
        logger.debug("MST export failed; falling back to SOL format", exc_info=True)

    try:
        with tempfile.NamedTemporaryFile(suffix=".sol", delete=False) as handle:
            temp_path = handle.name
        try:
            model.solution.write(temp_path)
            return pathlib.Path(temp_path).read_text(encoding="utf-8")
        finally:
            os.unlink(temp_path)
    except Exception:  # noqa: BLE001
        logger.debug("SOL export fallback failed", exc_info=True)
        return ""


def _structured_seed_failure(
    *,
    solve_time: float,
    solve_status: str,
    progress_samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a single-seed result dict for a failed or aborted run."""
    return {
        "solve_time_s": solve_time,
        "objective_value": None,
        "best_bound": None,
        "mip_gap": None,
        "nodes_explored": 0,
        "solve_status": solve_status,
        "best_solution_mst": "",
        "progress_samples": progress_samples or [],
    }


def _make_progress_callback(
    interval_seconds: float,
    samples: list[dict[str, Any]],
) -> type:
    """Create a MIPInfoCallback subclass that records progress at fixed intervals."""
    import cplex

    class ProgressCallbackImpl(cplex.callbacks.MIPInfoCallback):
        def __init__(self, env: object) -> None:
            super().__init__(env)
            self._interval = interval_seconds
            self._samples = samples
            self._last_t: float | None = None

        def __call__(self) -> None:
            t = self.get_time() - self.get_start_time()
            if self._last_t is None or t >= self._last_t + self._interval:
                self._last_t = t
                best_obj = self.get_incumbent_objective_value()
                best_bound = self.get_best_objective_value()
                nodes = self.get_num_nodes()
                if abs(best_obj) >= _NO_INCUMBENT_SENTINEL:
                    best_obj = None
                if abs(best_bound) >= _NO_INCUMBENT_SENTINEL:
                    best_bound = None
                if best_obj is not None and best_bound is not None and best_obj != 0:
                    gap = abs(best_bound - best_obj) / abs(best_obj)
                else:
                    gap = None
                self._samples.append(
                    {
                        "elapsed": t,
                        "best_objective": best_obj,
                        "best_bound": best_bound,
                        "nodes_explored": nodes,
                        "mip_gap": gap,
                    }
                )

    return ProgressCallbackImpl


def _align_to_grid(
    samples: list[dict[str, Any]],
    time_grid: list[float],
) -> dict[str, list[float | None]]:
    """Forward-fill samples onto a fixed time grid.

    For each grid point t, use the last sample with elapsed <= t.
    Returns dict of metric -> list of values (one per grid point).
    """
    aligned: dict[str, list[float | None]] = {
        "best_objective": [],
        "best_bound": [],
        "nodes_explored": [],
        "mip_gap": [],
    }
    sample_idx = 0
    last: dict[str, float | None] = {
        "best_objective": None,
        "best_bound": None,
        "nodes_explored": None,
        "mip_gap": None,
    }
    for t in time_grid:
        while sample_idx < len(samples) and samples[sample_idx]["elapsed"] <= t:
            s = samples[sample_idx]
            # None means "not updated at this sample"; forward-fill prior values.
            if s["best_objective"] is not None:
                last["best_objective"] = s["best_objective"]
            if s["best_bound"] is not None:
                last["best_bound"] = s["best_bound"]
            nodes = s["nodes_explored"]
            if nodes is not None:
                last["nodes_explored"] = float(nodes)
            if s["mip_gap"] is not None:
                last["mip_gap"] = s["mip_gap"]
            sample_idx += 1
        aligned["best_objective"].append(last["best_objective"])
        aligned["best_bound"].append(last["best_bound"])
        aligned["nodes_explored"].append(last["nodes_explored"])
        aligned["mip_gap"].append(last["mip_gap"])
    return aligned


def _build_time_grid(
    interval_s: float,
    time_limit_s: float,
    max_elapsed: float,
) -> list[float]:
    """Build uniform grid ``[0, interval_s, 2*interval_s, ...]`` covering all samples.

    ``max_elapsed`` is the latest ``elapsed`` among callbacks and the post-solve
    terminal row (per ``solve_mip``). The grid must reach at least ``max_elapsed``
    on its last point; otherwise ``_align_to_grid`` would never apply samples with
    ``elapsed`` between the previous multiple of ``interval_s`` and ``max_elapsed``.

    When ``time_limit_s`` is finite, the initial cap is at least the limit and at
    least ``max_elapsed`` (handles small overrun past ``TILIM``).
    """
    if interval_s <= 0:
        return []
    cap = float(time_limit_s) if time_limit_s < 1e70 else float(max_elapsed)
    cap = max(cap, float(max_elapsed))
    grid: list[float] = []
    t = 0.0
    # 1e-9 is added to avoid issues with rounding errors when
    # accumulating t+interval_s
    # e.g. 0.2+0.1 in python is not 0.3
    while t <= cap + 1e-9:
        grid.append(t)
        t += interval_s
    while grid and grid[-1] + 1e-9 < float(max_elapsed):
        grid.append(grid[-1] + interval_s)
    return grid


def _append_terminal_progress_sample(
    *,
    model: object,
    progress_samples: list[dict[str, Any]],
    solve_time: float,
    objective_value: float | None,
    mip_gap: float | None,
    nodes_explored: int,
) -> None:
    """Append one sample at solve end so the aligned grid includes the final MIP state.

    Periodic MIPInfoCallback samples can omit the last jump to optimality if the
    solver finishes between two callback ticks.  This terminal sample ensures
    the aligned grid always reflects the final incumbent and best bound.

    Best bound selection:
    - ``mip_gap == 0`` (proven optimal): best bound converges to ``objective_value``.
    - Non-optimal (e.g. time limit): forward-fill the last callback-recorded bound.
      The post-solve CPLEX solution API may return the incumbent rather than the
      LP-relaxation dual bound, so it is only used as a fallback when no callback
      samples exist (``progress_interval_s == 0``).
    """
    if mip_gap is not None and mip_gap == 0.0 and objective_value is not None:
        best_bound: float | None = objective_value
    else:
        best_bound = None
        for prev in reversed(progress_samples):
            prev_bound = prev.get("best_bound")
            if prev_bound is not None:
                best_bound = prev_bound
                break
        if best_bound is None:
            # No callback samples available (progress_interval_s == 0).
            # The post-solve API may return the incumbent for non-optimal solves,
            # so this is a best-effort fallback only.
            import cplex

            try:
                api_bound = float(model.solution.MIP.get_best_objective_value())
                if abs(api_bound) < _NO_INCUMBENT_SENTINEL:
                    best_bound = api_bound
            except (
                cplex.exceptions.CplexSolverError,
                AttributeError,
                TypeError,
                ValueError,
            ):
                pass

    progress_samples.append(
        {
            "elapsed": float(solve_time),
            "best_objective": objective_value,
            "best_bound": best_bound,
            "nodes_explored": nodes_explored,
            "mip_gap": mip_gap,
        }
    )


def estimate_mip_memory_bytes(mps_file_path: str) -> int:
    """Estimate the Ray memory resource request for a single CPLEX seed task.

    Uses a power-law formula to scale the estimate with MPS file size while
    dampening growth for large instances.  The result is used both as the Ray
    task memory reservation and as the basis for the CPLEX WorkMem limit.

    Formula:
        estimated_peak_gb = floor_gb + (file_size_mb ** 0.75) * 0.5
        total_requested_gb = estimated_peak_gb * 1.20  (20% OS/Python headroom)

    The exponent 0.75 reflects that peak B&B memory grows sub-linearly with
    model size: larger models have deeper trees but also more pruning.  The
    floor of 4 GB covers solver initialisation overhead for trivial instances.

    Args:
        mps_file_path: Path to the MPS/LP instance file.

    Returns:
        Memory request in bytes for ``ray.remote(memory=...)``.
    """
    import os

    file_size_bytes = os.path.getsize(mps_file_path)
    file_size_mb = file_size_bytes / (1024**2)
    floor_gb = 4.0
    scale_factor = 0.5
    exponent = 0.75
    estimated_peak_gb = floor_gb + (file_size_mb**exponent) * scale_factor
    total_requested_gb = estimated_peak_gb * 1.20
    return int(total_requested_gb * (1024**3))


def _collect_parallel_seed_results(
    refs: list[object],
) -> list[dict[str, Any]]:
    """Collect per-seed Ray task results with partial-OK failure handling.

    If an individual seed task fails (for example runtime environment setup on a
    worker node), a structured failure entry is returned for that seed index and
    collection continues for the remaining refs.

    Args:
        refs: Ray object refs returned by ``remote_fn.remote(seed)``.

    Returns:
        One single-seed result dict per ref, in seed order.
    """
    import ray

    results: list[dict[str, Any]] = []
    for seed_index, ref in enumerate(refs):
        try:
            results.append(ray.get(ref))
        except Exception as exc:  # noqa: PERF203
            logger.warning("Ray seed task %d failed: %s", seed_index, exc)
            results.append(
                _structured_seed_failure(
                    solve_time=0.0,
                    solve_status=f"ray_task_failed: {exc}",
                )
            )
    return results


def _run_single_seed(
    *,
    mps_file: str,
    seed: int,
    seed_index: int,
    n_seeds: int,
    node_selection: int,
    variable_selection: int,
    heuristic_frequency: int,
    time_limit_s: float,
    n_threads: int,
    rins_frequency: int,
    cut_passes: int,
    cut_passes_all: CutPassesAllLevel = "cplex_default",
    mip_emphasis: int = 0,
    progress_interval_s: float = 0,
    workmem_mb: int = 0,
    warm_start_file: str = "",
    export_solution: bool = False,
) -> dict[str, Any]:
    """Run CPLEX on a single MPS instance with the given random seed and parameters.

    Args:
        mps_file: Path to the MPS instance file.
        seed: CPLEX random seed (CPX_PARAM_RANDOMSEED).
        seed_index: Index of this seed (0-based) for logging.
        n_seeds: Total number of seeds for logging.
        node_selection: Node selection strategy (CPX_PARAM_NODESEL).
        variable_selection: Variable selection strategy (CPX_PARAM_VARSEL).
        heuristic_frequency: Heuristic frequency (CPX_PARAM_HEURFREQ).
        time_limit_s: Time limit in seconds (CPX_PARAM_TILIM).
        cut_passes_all: Uniform MIP cut aggressiveness for all families, capped
            per family, or ``cplex_default`` to leave CPLEX defaults unchanged.
        mip_emphasis: MIP emphasis (CPX_PARAM_MIPEMPHASIS): 0-5, see property
            metadata; default 0 matches CPLEX balanced emphasis.
        progress_interval_s: If > 0, capture progress at this interval (seconds).
        workmem_mb: When > 0, sets CPLEX WorkMem (CPX_PARAM_WORKMEM) to this
            value in MB and enables compressed on-disk node files
            (CPX_PARAM_NODEFILEIND=3) so the solver spills to disk rather than
            crashing OOM.  Should be ~80% of the Ray task memory reservation.
        warm_start_file: Optional CPLEX MIP-start file path; empty disables warm start.
        export_solution: If True, populate best_solution_mst with MST XML.

    Returns:
        Dictionary with keys: solve_time_s, objective_value, best_bound, mip_gap,
        nodes_explored, solve_status, best_solution_mst, and progress_samples when
        progress_interval_s > 0.
    """
    import cplex

    solver_n = seed_index + 1
    logger.info("Start solver %d of %d", solver_n, n_seeds)

    model = cplex.Cplex()
    model.set_log_stream(sys.stdout)
    model.set_error_stream(sys.stderr)
    model.set_warning_stream(sys.stderr)
    model.set_results_stream(sys.stdout)

    model.read(mps_file)
    warm_start_error = _load_warm_start(model, warm_start_file)
    if warm_start_error is not None:
        logger.warning(
            "Warm start failed on seed %d: %s",
            seed,
            warm_start_error,
        )
        logger.info("End solver %d of %d", solver_n, n_seeds)
        return _structured_seed_failure(
            solve_time=0.0,
            solve_status=warm_start_error,
        )

    model.parameters.randomseed.set(seed)
    model.parameters.threads.set(n_threads)
    model.parameters.emphasis.mip.set(mip_emphasis)
    model.parameters.mip.strategy.nodeselect.set(node_selection)
    model.parameters.mip.strategy.variableselect.set(variable_selection)
    model.parameters.mip.strategy.heuristicfreq.set(heuristic_frequency)
    model.parameters.mip.strategy.rinsheur.set(rins_frequency)
    model.parameters.mip.limits.cutpasses.set(cut_passes)
    model.parameters.timelimit.set(time_limit_s)
    model.parameters.mip.display.set(4)
    model.parameters.mip.interval.set(100)
    if workmem_mb > 0:
        model.parameters.workmem.set(float(workmem_mb))
        model.parameters.mip.strategy.file.set(3)
    if cut_passes_all != "cplex_default":
        _apply_cut_passes_all(model, int(cut_passes_all))

    progress_samples: list[dict[str, Any]] = []
    if progress_interval_s > 0:
        model.register_callback(
            _make_progress_callback(progress_interval_s, progress_samples)
        )

    logger.debug(
        "Solving %s with seed=%d, n_threads=%d, node_selection=%d, "
        "variable_selection=%d, heuristic_frequency=%d, rins_frequency=%d, "
        "cut_passes=%d, cut_passes_all=%r, mip_emphasis=%d, time_limit_s=%.1g",
        mps_file,
        seed,
        n_threads,
        node_selection,
        variable_selection,
        heuristic_frequency,
        rins_frequency,
        cut_passes,
        cut_passes_all,
        mip_emphasis,
        time_limit_s,
    )

    t0 = time.perf_counter()
    try:
        model.solve()
    except cplex.exceptions.CplexSolverError as exc:
        solve_time = time.perf_counter() - t0
        # CPLEX error 1016 (CPXERR_RESTRICTED_VERSION) is the community-edition
        # size limit. Re-raise so the framework produces InvalidMeasurementResult,
        # preventing memoization from reusing this failed result.
        error_code = (
            exc.args[2] if len(exc.args) > 2 else getattr(exc, "error_code", None)
        )
        if error_code == 1016:  # CPXERR_RESTRICTED_VERSION
            logger.warning(
                "CPLEX Community Edition limits exceeded on seed %d: %s", seed, exc
            )
            raise
        # For other CPLEX errors, return a structured result.
        status = f"cplex_error_{error_code}" if error_code else f"cplex_error: {exc}"
        logger.warning("CPLEX solver error on seed %d: %s", seed, exc)
        logger.info("End solver %d of %d", solver_n, n_seeds)
        return _structured_seed_failure(
            solve_time=solve_time,
            solve_status=status,
        )
    solve_time = time.perf_counter() - t0

    status = model.solution.get_status_string()
    nodes = model.solution.progress.get_num_nodes_processed()

    try:
        obj = _normalize_cplex_value(model.solution.get_objective_value())
    except cplex.exceptions.CplexSolverError:
        obj = None

    try:
        gap = model.solution.MIP.get_mip_relative_gap()
    except cplex.exceptions.CplexSolverError:
        gap = None

    best_solution_mst = _export_incumbent_mst(model, obj) if export_solution else ""

    logger.debug(
        "Seed %d finished: status=%s, time=%.2fs, obj=%s, gap=%s, nodes=%d",
        seed,
        status,
        solve_time,
        obj,
        gap,
        nodes,
    )

    logger.info("End solver %d of %d", solver_n, n_seeds)

    # Always append the terminal sample.  When progress_interval_s > 0 it is
    # included in time-series alignment; in all cases it is the canonical source
    # for the scalar best_bound derived below.
    _append_terminal_progress_sample(
        model=model,
        progress_samples=progress_samples,
        solve_time=solve_time,
        objective_value=obj,
        mip_gap=gap,
        nodes_explored=nodes,
    )

    # Scalar best_bound: last non-None best_bound in progress_samples.
    # The terminal sample sets this to objective_value at optimality (mip_gap == 0)
    # and forward-fills the last callback-recorded LP-relaxation bound otherwise,
    # avoiding the post-solve CPLEX API which may return the incumbent.
    best_bound: float | None = None
    for prev in reversed(progress_samples):
        if prev.get("best_bound") is not None:
            best_bound = prev["best_bound"]
            break

    return {
        "solve_time_s": solve_time,
        "objective_value": obj,
        "best_bound": best_bound,
        "mip_gap": gap,
        "nodes_explored": nodes,
        "solve_status": status,
        "best_solution_mst": best_solution_mst,
        "progress_samples": progress_samples,
    }


@custom_experiment(
    required_properties=[MpsFile],
    optional_properties=[
        NSeeds,
        NodeSelection,
        VariableSelection,
        HeuristicFrequency,
        TimeLimit,
        NThreads,
        RinsFrequency,
        CutPasses,
        CutPassesAll,
        MipEmphasis,
        Parallel,
        ProgressInterval,
        WarmStartFile,
        ExportSolution,
    ],
    output_property_identifiers=[
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
    ],
    metadata={
        "description": (
            "Solves a MIP instance with CPLEX across N random seeds and reports "
            "vectors of performance metrics. Each output property is a list of "
            "length n_seeds, capturing solve time, objective value, MIP gap, "
            "nodes explored, and solver status per seed. This enables analysis "
            "of both parameter effects and seed-induced variability."
        )
    },
    parameterization={},
)
def solve_mip(
    mps_file: str,
    n_seeds: int = 5,
    node_selection: int = 1,
    variable_selection: int = 0,
    heuristic_frequency: int = 0,
    time_limit_s: float = 1e75,
    n_threads: int = 1,
    rins_frequency: int = 0,
    cut_passes: int = 0,
    cut_passes_all: CutPassesAllLevel = "cplex_default",
    mip_emphasis: int = 0,
    parallel: bool = True,
    progress_interval_s: float = 0,
    warm_start_file: str = "",
    export_solution: bool = False,
) -> dict[str, list]:
    """Solve a MIP instance with CPLEX across multiple random seeds.

    Args:
        mps_file: Path to the MPS instance file (.mps or .mps.gz).
        n_seeds: Number of random seeds to use (seeds 0 to n_seeds-1).
        node_selection: CPLEX node selection strategy (CPX_PARAM_NODESEL).
        variable_selection: CPLEX variable selection strategy (CPX_PARAM_VARSEL).
        heuristic_frequency: CPLEX heuristic frequency (CPX_PARAM_HEURFREQ).
        time_limit_s: CPLEX time limit per seed run in seconds. Default 1e75 means
            no practical limit; CPLEX runs until the optimal solution is found.
        n_threads: Number of parallel B&B threads (CPX_PARAM_THREADS). Default 1
            ensures fully deterministic execution.
        rins_frequency: RINS heuristic frequency (CPX_PARAM_RINSHEUR). -1=disabled,
            0=automatic, n=apply every n nodes.
        cut_passes: Max cutting-plane passes at root (CPX_PARAM_CUTPASSES).
            -1=no cuts, 0=automatic, n=at most n passes.
        cut_passes_all: Same aggressiveness for all ``mip.cuts`` families, capped
            per family; ``cplex_default`` leaves CPLEX defaults unchanged.
        mip_emphasis: CPLEX MIP emphasis (0=balanced through 5=heuristic); default 0.
        parallel: If True, run each seed as a Ray remote task (requires ray_remote).
            If False, run seeds in serial. With parallel=True, Ray task failures on
            individual seeds are recorded as ``ray_task_failed: ...`` in
            solve_statuses with null metrics for that seed; successful seeds are
            still returned (partial-OK policy).
        progress_interval_s: If > 0, capture intermediate progress at this interval
            (seconds). Outputs progress_time_grid and aligned time-series.
        warm_start_file: Optional CPLEX MIP-start file applied before each seed solve.
        export_solution: If True, populate best_solution_mst with MST XML per seed.

    Returns:
        Dictionary with vector-valued outputs (one element per seed):
        - solve_times: Wall-clock solve times in seconds.
        - objective_values: Best objective values found.
        - best_bounds: Final MIP best bounds.
        - best_solution_mst: MST XML strings for warm-start round-trip, or ``""``.
        - mip_gaps: Final relative MIP gaps.
        - nodes_explored: B&B nodes processed.
        - solve_statuses: CPLEX status strings.
        When progress_interval_s > 0, also:
        - progress_time_grid: Shared time points (seconds).
        - objective_over_time, best_bound_over_time, nodes_explored_over_time,
          mip_gap_over_time: list[list] of aligned values [seed][time_idx].
          A terminal sample at solve completion is appended so forward-filled
          grid values can reflect the final incumbent and best bound, not only
          the last periodic callback.
    """
    import ray

    mem_bytes = estimate_mip_memory_bytes(mps_file)
    # 80% of the Ray task reservation: CPLEX WorkMem budget in MB, leaving 20%
    # headroom for the Python interpreter and Ray worker overhead.
    workmem_mb = int(mem_bytes / (1024**2) * 0.80)

    def _run_one(seed: int) -> dict[str, Any]:
        return _run_single_seed(
            mps_file=mps_file,
            seed=seed,
            seed_index=seed,
            n_seeds=n_seeds,
            node_selection=node_selection,
            variable_selection=variable_selection,
            heuristic_frequency=heuristic_frequency,
            time_limit_s=time_limit_s,
            n_threads=n_threads,
            rins_frequency=rins_frequency,
            cut_passes=cut_passes,
            cut_passes_all=cut_passes_all,
            mip_emphasis=mip_emphasis,
            progress_interval_s=progress_interval_s,
            warm_start_file=warm_start_file,
            export_solution=export_solution,
            workmem_mb=workmem_mb,
        )

    if parallel:
        if not ray.is_initialized():
            raise ValueError(
                "parallel=True requires the experiment to run with ray_remote "
                "(use_ray=True). Ray is not initialized."
            )
        remote_fn = ray.remote(num_cpus=n_threads, memory=mem_bytes)(_run_one)
        refs = [remote_fn.remote(seed) for seed in range(n_seeds)]
        results = _collect_parallel_seed_results(refs)
    else:
        results = []
        for seed in range(n_seeds):
            result = _run_one(seed)
            results.append(result)

    out: dict[str, list] = {
        "solve_times": [r["solve_time_s"] for r in results],
        "objective_values": [r["objective_value"] for r in results],
        "best_bounds": [r["best_bound"] for r in results],
        "best_solution_mst": [r["best_solution_mst"] for r in results],
        "mip_gaps": [r["mip_gap"] for r in results],
        "nodes_explored": [r["nodes_explored"] for r in results],
        "solve_statuses": [r["solve_status"] for r in results],
    }

    if progress_interval_s > 0:
        all_samples = [r.get("progress_samples", []) for r in results]
        max_elapsed = max(
            (s["elapsed"] for samples in all_samples for s in samples),
            default=0.0,
        )
        time_grid = _build_time_grid(progress_interval_s, time_limit_s, max_elapsed)
        aligned_per_seed = [
            _align_to_grid(samples, time_grid) for samples in all_samples
        ]
        out["progress_time_grid"] = time_grid
        out["objective_over_time"] = [a["best_objective"] for a in aligned_per_seed]
        out["best_bound_over_time"] = [a["best_bound"] for a in aligned_per_seed]
        out["nodes_explored_over_time"] = [
            a["nodes_explored"] for a in aligned_per_seed
        ]
        out["mip_gap_over_time"] = [a["mip_gap"] for a in aligned_per_seed]
    else:
        out["progress_time_grid"] = []
        out["objective_over_time"] = []
        out["best_bound_over_time"] = []
        out["nodes_explored_over_time"] = []
        out["mip_gap_over_time"] = []

    return out
