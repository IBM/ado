# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import logging
import sys
import time
from typing import Any

from orchestrator.modules.actuators.custom_experiments import custom_experiment
from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum
from orchestrator.schema.property import ConstitutiveProperty

logger = logging.getLogger(__name__)

DEFAULT_MPS_FILE = "/Users/michaelj/tmp/miplib_2017_benchmarks/bab6.mps.gz"

MpsFile = ConstitutiveProperty(
    identifier="mps_file",
    metadata={
        "description": (
            "Path to the MPS instance file to solve (.mps or .mps.gz). "
            f"Defaults to the bab6 MIPLIB 2017 benchmark ({DEFAULT_MPS_FILE})."
        )
    },
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE,
        values=[DEFAULT_MPS_FILE],
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
            "0=depth-first, 1=breadth-first (default), "
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
            "No domain range is enforced — any positive value including 1e75 is accepted."
        )
    },
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
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
        values=[-1, 0, 1, 5],
    ),
)

Parallel = ConstitutiveProperty(
    identifier="parallel",
    metadata={
        "description": (
            "If True, run each of the n_seeds solver instances as a Ray remote task. "
            "If False, run all seeds in serial."
        )
    },
    propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.BINARY_VARIABLE_TYPE,
        values=[False, True],
    ),
)


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

    Returns:
        Dictionary with keys: solve_time_s, objective_value, mip_gap,
        nodes_explored, solve_status.
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
    model.parameters.randomseed.set(seed)
    model.parameters.threads.set(n_threads)
    model.parameters.mip.strategy.nodeselect.set(node_selection)
    model.parameters.mip.strategy.variableselect.set(variable_selection)
    model.parameters.mip.strategy.heuristicfreq.set(heuristic_frequency)
    model.parameters.mip.strategy.rinsheur.set(rins_frequency)
    model.parameters.mip.limits.cutpasses.set(cut_passes)
    model.parameters.timelimit.set(time_limit_s)
    model.parameters.mip.display.set(4)
    model.parameters.mip.interval.set(100)

    logger.debug(
        "Solving %s with seed=%d, n_threads=%d, node_selection=%d, "
        "variable_selection=%d, heuristic_frequency=%d, rins_frequency=%d, "
        "cut_passes=%d, time_limit_s=%.1g",
        mps_file,
        seed,
        n_threads,
        node_selection,
        variable_selection,
        heuristic_frequency,
        rins_frequency,
        cut_passes,
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
        return {
            "solve_time_s": solve_time,
            "objective_value": float("nan"),
            "mip_gap": float("nan"),
            "nodes_explored": 0,
            "solve_status": status,
        }
    solve_time = time.perf_counter() - t0

    status = model.solution.get_status_string()
    nodes = model.solution.progress.get_num_nodes_processed()

    try:
        obj = model.solution.get_objective_value()
    except cplex.exceptions.CplexSolverError:
        obj = float("nan")

    try:
        gap = model.solution.MIP.get_mip_relative_gap()
    except cplex.exceptions.CplexSolverError:
        gap = float("nan")

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

    return {
        "solve_time_s": solve_time,
        "objective_value": obj,
        "mip_gap": gap,
        "nodes_explored": nodes,
        "solve_status": status,
    }


@custom_experiment(
    required_properties=[],
    optional_properties=[
        MpsFile,
        NSeeds,
        NodeSelection,
        VariableSelection,
        HeuristicFrequency,
        TimeLimit,
        NThreads,
        RinsFrequency,
        CutPasses,
        Parallel,
    ],
    output_property_identifiers=[
        "solve_times",
        "objective_values",
        "mip_gaps",
        "nodes_explored",
        "solve_statuses",
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
    mps_file: str = DEFAULT_MPS_FILE,
    n_seeds: int = 5,
    node_selection: int = 1,
    variable_selection: int = 0,
    heuristic_frequency: int = 0,
    time_limit_s: float = 1e75,
    n_threads: int = 1,
    rins_frequency: int = 0,
    cut_passes: int = 0,
    parallel: bool = True,
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
        parallel: If True, run each seed as a Ray remote task (requires ray_remote).
            If False, run seeds in serial.

    Returns:
        Dictionary with vector-valued outputs (one element per seed):
        - solve_times: Wall-clock solve times in seconds.
        - objective_values: Best objective values found.
        - mip_gaps: Final relative MIP gaps.
        - nodes_explored: B&B nodes processed.
        - solve_statuses: CPLEX status strings.
    """
    import ray

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
        )

    if parallel:
        if not ray.is_initialized():
            raise ValueError(
                "parallel=True requires the experiment to run with ray_remote "
                "(use_ray=True). Ray is not initialized."
            )
        remote_fn = ray.remote(_run_one)
        refs = [remote_fn.remote(seed) for seed in range(n_seeds)]
        results = ray.get(refs)
    else:
        results = []
        for seed in range(n_seeds):
            result = _run_one(seed)
            results.append(result)

    return {
        "solve_times": [r["solve_time_s"] for r in results],
        "objective_values": [r["objective_value"] for r in results],
        "mip_gaps": [r["mip_gap"] for r in results],
        "nodes_explored": [r["nodes_explored"] for r in results],
        "solve_statuses": [r["solve_status"] for r in results],
    }
