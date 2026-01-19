# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
import math
import random
from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from scipy.stats.qmc import Sobol

from trim.utils.one_dimensional_sampling import get_index_list_nn


def concatenated_latin_hypercube_sampling(
    dims: list[int],
    n: int,
    seed: int | None = None,
) -> list[list[int]]:
    """
    Generates samples using a Concatenated Latin Hypercube Sampling strategy.

    For each dimension independently, this method enforces a 1D stratification
    (Latin Hypercube property) by generating random permutations of the
    possible values. If the number of requested samples 'n' exceeds the cardinality
    of a dimension, new random permutations are concatenated to the sequence.

    This guarantees that for any dimension j with size d_j, every sequence
    of d_j samples contains exactly one instance of every value in range(d_j).

    Args:
        dims (List[int]): Cardinality (size) of each dimension. Must be positive.
        n (int): Total number of points to sample.
        seed (Optional[int]): Optional PRNG seed for reproducibility.

    Returns:
        List[List[int]]: A list of n sampled points, where each point is a
        list of indices corresponding to the dimensions.

    Raises:
        ValueError: If any dimension size is less than 1.
    """
    if any(d <= 0 for d in dims):
        raise ValueError(f"All dimensions must be >= 1, received dims={dims}")

    if n <= 0:
        return []

    rng = random.Random(seed)  # noqa: S311

    # Per-dimension pools: active permutation for the current block.
    # We maintain the Latin Hypercube property by sampling without replacement.
    pools: list[list[int]] = [list(range(d)) for d in dims]
    samples: list[list[int]] = []

    for _ in range(n):
        point: list[int] = []
        for j, d in enumerate(dims):
            # If the current permutation block is exhausted, start a new one (new cycle).
            if not pools[j]:
                pools[j] = list(range(d))

            # Select a random element from the remaining pool for this block.
            k = rng.randrange(len(pools[j]))
            val = pools[j].pop(k)
            point.append(val)

        samples.append(point)

    return samples


# TODO: test this


def sobol_sampling(dims: list[int], n: int, seed: int | None = None) -> list[list[int]]:
    """
    Generates Sobol sampled points scaled to integer dimensions.

    This function uses a Sobol sequence to generate points in the unit hypercube [0, 1)^d,
    scales them to the specified integer dimensions, and checks for collisions. If collisions
    occur (duplicate points), it falls back to Concatenated Latin Hypercube Sampling.

    Args:
        dims (list[int]): A list of integers representing the size (cardinality) of each dimension.
        n (int): The number of points to sample.
        seed (int | None, optional): Random seed for the Sobol scrambler. Defaults to None.

    Returns:
        list[list[int]]: A list of n points, where each point is a list of integer coordinates.
    """
    # Sobol generates points in [0, 1). We scale them to the integer dimensions.

    sampler = Sobol(d=len(dims), scramble=True, rng=seed)
    points = sampler.random(n)

    # Scale and floor to get integer indices
    discrete_points = [
        [int(val * d) for val, d in zip(p, dims, strict=True)] for p in points
    ]

    # Check for collisions
    # Convert inner lists to tuples because lists are unhashable and cannot be used in a set
    unique_points = {tuple(p) for p in discrete_points}
    n_collisions = n - len(unique_points)

    if n_collisions > 0:
        logging.error(
            f"Sobol sampling failed, {n_collisions} collisions detected, defaulting to clhs sampling"
        )
        return concatenated_latin_hypercube_sampling(dims=dims, n=n, seed=seed)

    return discrete_points


# TODO: test this function
def distinct_sobol_sampling(
    dims: list[int], n: int, seed: int | None = None
) -> list[list[int]]:
    """
    Generates 'n' distinct points on a grid of size 'dims' using a Sobol sequence.
    Guarantees no collisions by skipping duplicates in the sequence.
    """
    # 1. Safety Check: Is the grid big enough?
    total_capacity = np.prod(dims)
    if n > total_capacity:
        raise ValueError(
            f"Cannot generate {n} distinct points: Grid only has {total_capacity} cells."
        )

    # 2. Setup Sobol
    # We scramble to get better coverage.
    sampler = Sobol(d=len(dims), scramble=True, rng=seed)

    unique_points = set()
    results = []

    # 3. Iterative Generation
    # We generate in batches to be efficient.
    # Start with a batch larger than N to account for potential rejections.
    batch_size = max(n * 2, 64)

    while len(results) < n:
        # Draw a batch of float points [0, 1)
        raw_points = sampler.random(batch_size)

        for p in raw_points:
            # Discretize: Map [0, 1) -> Integer coordinates
            # Using int(x * dim) scales it to the grid index [0, dim-1]
            coord = tuple([int(p[i] * dims[i]) for i in range(len(dims))])

            # Check Uniqueness
            if coord not in unique_points:
                unique_points.add(coord)
                results.append(list(coord))

                # Stop immediately if we have enough
                if len(results) == n:
                    break

        # If we need more points, increase batch size for next iteration
        # (helpful if the grid is nearly full and collisions are frequent)
        batch_size *= 2

    return results


def random_high_dimensional_sampling(
    dims: list[int], n: int, seed: int | None = None
) -> list[list[int]]:
    """
    Generate n unique random samples from a high-dimensional space.

    Args:
        dims: Cardinality (size) of each dimension. Must be positive.
        n: Total number of points to sample.
        seed: Optional PRNG seed for reproducibility.

    Returns:
        List of n sampled points, each point is a list of indices

    Raises:
        ValueError: If n exceeds the total number of possible configurations
    """
    import itertools
    import random
    from math import prod

    # Set the seed for the random number generator
    if seed is not None:
        random.seed(seed)

    # Check if the number of requested samples is valid
    num_configs = prod(dims)
    if n > num_configs:
        raise ValueError(
            f"Cannot generate {n} unique samples. "
            f"The sample space only contains {num_configs} possibilities."
        )

    # This still creates all combinations in memory, which is a limitation
    # for extremely large dimensional spaces.
    configs = itertools.product(*[range(d) for d in dims])

    # random.sample is highly optimized for this task.
    # It's much faster than manually choosing and removing.
    samples = random.sample(list(configs), n)

    return [list(s) for s in samples]


def get_order_list_nn_high_dimensional(
    dims: list[int],
    n: int | str = "all",
    space: dict[str, int] | None = None,
    strategy: str = "clhs",
    seed: int | None = None,
) -> list[list[int]]:
    """
    Generate sampling indices for a high-dimensional space using `get_index_list_nn` for each dimension.

    Args:
        dims (List[int]): Sizes of each dimension (e.g., [8, 5]).
        n (int | str): Number of points to sample:
            - 'all': sample all possible combinations (product of dims)
            - 'max': sample up to max(dims)
        strategy (str): sampling subroutine:
            - 'random': selects random points from the beginning
            - 'clhs': refer to concatenated_latin_hypercube_sampling
            - 'sobol': sobol sampling

        space (Optional[Dict[str, int]]): Optional mapping of dimension names to sizes (used only for logging/debug purposes).
            Example:
                space = {'batch_size': 8, 'model_name': 5}
        seed (Optional[int]): controls the randomness

    note: strategies may have an upper bound on the number of elements that respect the strategy that they can return
    if this number is exceeded, they resort to random sampling.

    Returns:
        List[List[int]]: Outer list length = n (or product of dims if n='all').
                        Each inner list contains one sampled combination across dimensions.
    """

    # Set the seed for the random number generator
    if seed is not None:
        random.seed(seed)
    else:
        seed = 123

    # Log space details if provided
    if space:
        indices_dict = {k: get_index_list_nn(v, v) for k, v in space.items()}
        if [len(indices) for indices in list(indices_dict.values())] != dims:
            logging.error(
                f"A space dict has been provided ->{space}. It is inconsistent with dims={dims}"
            )
            logging.warning(
                f"list(indices_dict.values()) = {list(indices_dict.values())}"
            )
            # raise ValueError
        logging.info(
            "Sampling indices for each named dimension (ordered low to high): %s",
            indices_dict,
        )

    # Compute sampling orders for each dimension
    orders = [get_index_list_nn(v, v) for v in dims]
    logging.debug("Dimensions: %s", dims)
    logging.debug("Sampling orders for each dimension:")
    for i, o in enumerate(orders):
        logging.debug("Dimension %d order: %s", i, o)

    # Calculate maximum possible samples
    maximum_n = 1
    for d in dims:
        maximum_n *= d
    lcm = math.lcm(*dims)

    if lcm != maximum_n:
        logging.debug(
            "Periodicity detected, the sampling subroutine will ensure that you will not sampple"
            "the same configuration more than once."
        )

    if isinstance(n, str):
        if n == "all":
            n = maximum_n
        elif n == "max":
            n = max(dims)
        else:
            raise ValueError(f"Unrecognized string for n: {n}")

    if n > maximum_n:
        logging.warning(
            f"""Maximal sample size is {maximum_n}, you requested {n} sampling presciptions.
                        Elaborating prescription for n_samples = {maximum_n}"""
        )

    logging.debug("Preparing to sample %d out of %d possible points.", n, maximum_n)

    if strategy == "random":
        return random_high_dimensional_sampling(dims, n, seed=seed)

    if strategy == "clhs":
        return concatenated_latin_hypercube_sampling(dims=dims, n=n, seed=seed)

    if strategy == "sobol":
        return sobol_sampling(dims=dims, n=n, seed=seed)

    raise NotImplementedError(f"Strategy {strategy} is unknown")


def unique_in_order_list_of_lists(
    lists: Sequence[Sequence[Any]],
) -> list[Sequence[Any]]:
    """
    Return the first occurrence of each unique row, preserving input order.

    This function deduplicates a sequence of rows (e.g., a list of lists)
    while maintaining the original order of appearance. A row's uniqueness
    is determined by converting it to a tuple and using that as a key in
    a set, which allows list rows to be compared efficiently.

    Parameters
    ----------
    lists : Sequence[Sequence[Any]]
        An iterable of rows (e.g., list of lists). Each row's elements must
        be hashable, because the row is converted to a tuple to be used as
        a set key.

    Returns
    -------
    List[Sequence[Any]]
        A list containing the first occurrence of each distinct row from
        `lists`, in the same order they first appear. Rows are returned as
        references to the original row objects (no copies are made).

    Notes
    -----
    - Order is stable: only the first occurrence of each row is kept.
    """
    seen = set()
    out = []
    for row in lists:
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def plot_grid(
    ax: Axes,
    dims: list[int] | tuple[int, int],
    points: np.ndarray | list[list[int]],
    title: str,
) -> None:
    """
    Plot a 2D grid visualization of sampled points with overlap detection.

    Args:
        ax: Matplotlib axes object to draw on
        dims: Dimensions of the grid [width, height]
        points: List of sampled points as [x, y] coordinates
        title: Title for the plot
    """
    from collections import defaultdict

    import matplotlib.patches as patches

    nx, ny = dims[0], dims[1]

    # Setup grid
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_xticks(range(nx + 1))
    ax.set_yticks(range(ny + 1))
    ax.grid(True, color="black", linewidth=1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, pad=10)

    # Track points in each cell to handle overlaps
    # Maps (x, y) -> list of time indices (1-based)
    grid_content = defaultdict(list)

    # points is a list of [x, y], enumerate gives us the time index (0-based)
    for t, p in enumerate(points):
        x, y = int(p[0]), int(p[1])  # Ensure integers
        if 0 <= x < nx and 0 <= y < ny:
            # Store t + 1 so the first sample is '1'
            grid_content[(x, y)].append(t + 1)

    # Draw squares and text
    for (x, y), indices in grid_content.items():
        count = len(indices)
        # Darker alpha if multiple points hit the same square
        alpha = min(0.4 + 0.2 * count, 1.0)
        rect = patches.Rectangle(
            (x, y), 1, 1, linewidth=0, facecolor="#ff0000", alpha=alpha
        )
        ax.add_patch(rect)

        # Label is the comma-separated list of indices
        label = ",".join(map(str, indices))

        # Add text

        ax.text(
            x + 0.52,
            y + 0.52,
            label,
            ha="center",
            va="center",
            color="#D4FF00",
            fontweight="bold",
        )
        ax.text(
            x + 0.5,
            y + 0.5,
            label,
            ha="center",
            va="center",
            color="#000000",
            fontweight="bold",
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # --- Configuration ---
    DIMS = [40, 6]  # 4 columns, 6 rows (Total 24 cells)
    N = 50  # Number of samples to draw
    SEED = 42

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Random Sampling
    pts_rnd = random_high_dimensional_sampling(DIMS, N, seed=SEED)
    plot_grid(axes[0], DIMS, pts_rnd, f"Random Sampling (N={N})\n(Clumps & Gaps)")

    # 2. Concatenated LHS
    pts_lhs = concatenated_latin_hypercube_sampling(DIMS, N, seed=SEED)
    plot_grid(axes[1], DIMS, pts_lhs, f"Concatenated LHS (N={N})\n(Uniform Rows/Cols)")

    # 3. Sobol Sequence
    pts_sobol = sobol_sampling(DIMS, N, seed=SEED)
    plot_grid(axes[2], DIMS, pts_sobol, f"Sobol Sequence (N={N})\n(Maximal Spreading)")

    plt.tight_layout()
    plt.show()
# %%
