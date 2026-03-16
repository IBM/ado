# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import logging
import math
import random
from typing import Literal

import numpy as np
from scipy.stats.qmc import Sobol

from no_priors_characterization.utils.one_dimensional_sampling import (
    get_index_list_van_der_corput,
)

logger_high_dimensional = logging.getLogger(__name__)


def concatenated_latin_hypercube_sampling(
    dimensions: list[int],
    final_sample_size: int,
    seed: int | None = None,
) -> list[list[int]]:
    """
    Generates samples using a Concatenated Latin Hypercube Sampling strategy.

    For each dimension independently, this method enforces a 1D stratification
    (Latin Hypercube property) by generating random permutations of the
    possible values. If the number of requested samples 'final_sample_size' exceeds the cardinality
    of a dimension, new random permutations are concatenated to the sequence.

    This guarantees that for any dimension j with size d_j, every sequence
    of d_j samples contains exactly one instance of every value in range(d_j).

    Args:
        dimensions (List[int]): Cardinality (size) of each dimension. Must be positive.
        final_sample_size (int): Total number of points to sample.
        seed (Optional[int]): Optional PRNG seed for reproducibility.

    Returns:
        List[List[int]]: A list of final_sample_size sampled points, where each point is a
        list of indices corresponding to the dimensions.

    Raises:
        ValueError: If any dimension size is less than 1.
    """
    if any(d <= 0 for d in dimensions):
        raise ValueError(
            f"All dimensions must be >= 1, received dimensions={dimensions}"
        )

    if final_sample_size <= 0:
        return []

    # Use default RNG when seed is not provided, otherwise create seeded instance
    rng = random.Random() if seed is None else random.Random(seed)  # noqa: S311

    # Per-dimension pools: active permutation for the current block.
    # We maintain the Latin Hypercube property by sampling without replacement.
    pools: list[list[int]] = [list(range(d)) for d in dimensions]
    samples: list[list[int]] = []

    for _ in range(final_sample_size):
        point: list[int] = []
        for j, d in enumerate(dimensions):
            # If the current permutation block is exhausted, start a new one (new cycle).
            if not pools[j]:
                pools[j] = list(range(d))

            # Select a random element from the remaining pool for this block.
            k = rng.randrange(len(pools[j]))
            value = pools[j].pop(k)
            point.append(value)

        samples.append(point)

    return samples


# NOTE: preliminary tests on collision reveal that if final_sample_size is half of the product of dimensions collisions are rare
def sobol_sampling(
    dimensions: list[int], final_sample_size: int, seed: int | None = None
) -> list[list[int]]:
    """
    Generates Sobol sampled points scaled to integer dimensions.

    This function uses a Sobol sequence to generate points in the unit hypercube [0, 1)^d,
    scales them to the specified integer dimensions, and checks for collisions. If collisions
    occur (duplicate points), it falls back to Concatenated Latin Hypercube Sampling.

    Args:
        dimensions (list[int]): A list of integers representing the size (cardinality) of each dimension.
        final_sample_size (int): The number of points to sample.
        seed (int | None, optional): Random seed for the Sobol scrambler. Defaults to None.

    Returns:
        list[list[int]]: A list of final_sample_size points, where each point is a list of integer coordinates.
    """
    # Sobol generates points in [0, 1). We scale them to the integer dimensions.

    sampler = Sobol(d=len(dimensions), scramble=True, rng=seed)
    points = sampler.random(final_sample_size)

    # Scale and floor to get integer indices
    discrete_points = [
        [int(val * d) for val, d in zip(p, dimensions, strict=True)] for p in points
    ]

    # Check for collisions
    # Convert inner lists to tuples because lists are unhashable and cannot be used in a set
    unique_points = {tuple(p) for p in discrete_points}
    n_collisions = final_sample_size - len(unique_points)

    if n_collisions > 0:
        logger_high_dimensional.error(
            f"Sobol sampling failed, {n_collisions} collisions detected, defaulting to clhs sampling"
        )
        return concatenated_latin_hypercube_sampling(
            dimensions=dimensions, final_sample_size=final_sample_size, seed=seed
        )

    return discrete_points


# TODO: test this function
def distinct_sobol_sampling(
    dimensions: list[int], final_sample_size: int, seed: int | None = None
) -> list[list[int]]:
    """
    Generates 'n' distinct points on a grid of size 'dimensions' using a Sobol sequence.
    Guarantees no collisions by skipping duplicates in the sequence.
    """
    # 1. Safety Check: Is the grid big enough?
    total_capacity = np.prod(dimensions)
    if final_sample_size > total_capacity:
        raise ValueError(
            f"Cannot generate {final_sample_size} distinct points: Grid only has {total_capacity} cells."
        )

    # 2. Setup Sobol
    # We scramble to get better coverage.
    sampler = Sobol(d=len(dimensions), scramble=True, rng=seed)

    unique_points = set()
    results = []

    # 3. Iterative Generation
    # We generate in batches to be efficient.
    # Start with a batch larger than N to account for potential rejections.
    batch_size = max(final_sample_size * 2, 64)

    while len(results) < final_sample_size:
        # Draw a batch of float points [0, 1)
        raw_points = sampler.random(batch_size)

        for p in raw_points:
            # Discretize: Map [0, 1) -> Integer coordinates
            # Using int(x * dim) scales it to the grid index [0, dim-1]
            coord = tuple([int(p[i] * dimensions[i]) for i in range(len(dimensions))])

            # Check Uniqueness
            if coord in unique_points:
                continue

            unique_points.add(coord)
            results.append(list(coord))

            # Stop immediately if we have enough
            if len(results) == final_sample_size:
                return results

        # If we need more points, increase batch size for next iteration
        # (helpful if the grid is nearly full and collisions are frequent)
        batch_size *= 2

    return results


def random_high_dimensional_sampling(
    dimensions: list[int], final_sample_size: int, seed: int | None = None
) -> list[list[int]]:
    """
    Generate n unique random samples from a high-dimensional space.

    Args:
        dimensions: Cardinality (size) of each dimension. Must be positive.
        final_sample_size: Total number of points to sample.
        seed: Optional PRNG seed for reproducibility.

    Returns:
        List of final_sample_size sampled points, each point is a list of indices

    Raises:
        ValueError: If final_sample_size exceeds the total number of possible configurations
    """
    import itertools
    import random
    from math import prod

    # Set the seed for the random number generator
    if seed is not None:
        random.seed(seed)

    # Check if the number of requested samples is valid
    num_configs = prod(dimensions)
    if final_sample_size > num_configs:
        raise ValueError(
            f"Cannot generate {final_sample_size} unique samples. "
            f"The sample space only contains {num_configs} possibilities."
        )

    # This still creates all combinations in memory, which is a limitation
    # for extremely large dimensional spaces.
    configs = list(itertools.product(*[range(d) for d in dimensions]))

    # Ensure we don't try to sample more than available
    actual_sample_size = min(final_sample_size, len(configs))
    if actual_sample_size < final_sample_size:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            f"Requested {final_sample_size} samples but only {len(configs)} unique "
            f"configurations available. Sampling {actual_sample_size} instead."
        )

    # random.sample is highly optimized for this task.
    # It's much faster than manually choosing and removing.
    samples = random.sample(configs, actual_sample_size)

    return [list(s) for s in samples]


def get_sampling_indices_multi_dimensional(
    dimensions: list[int],
    n: int | Literal["all", "max"],
    space: dict[str, int] | None = None,
    strategy: Literal["random", "clhs", "sobol"] = "clhs",
    seed: int | None = None,
) -> list[list[int]]:
    """
    Generate sampling indices for a high-dimensional space using `get_index_list_van_der_corput` for each dimension.

    Args:
        dimensions (List[int]): Sizes of each dimension (e.g., [8, 5]).
        n (int | str): Number of points to sample:
            - 'all': sample all possible combinations (product of dimensions)
            - 'max': sample up to max(dimensions)
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
        List[List[int]]: Outer list length = n (or product of dimensions if n='all').
                        Each inner list contains one sampled combination across dimensions.
    """

    # Set the seed for the random number generator
    if seed is not None:
        random.seed(seed)

    # Log space details if provided
    if space:
        indices_dict = {
            k: get_index_list_van_der_corput(v, v) for k, v in space.items()
        }
        if [len(indices) for indices in list(indices_dict.values())] != dimensions:
            logger_high_dimensional.error(
                f"A space dict has been provided ->{space}. It is inconsistent with dimensions={dimensions}"
            )
            logger_high_dimensional.warning(
                f"list(indices_dict.values()) = {list(indices_dict.values())}"
            )
            raise ValueError("Space has inconsistent dimensions!")
        logger_high_dimensional.info(
            "Sampling indices for each named dimension (ordered low to high): %s",
            indices_dict,
        )

    # Compute sampling orders for each dimension
    orders = [get_index_list_van_der_corput(v, v) for v in dimensions]

    if logger_high_dimensional.isEnabledFor(logging.DEBUG):
        logger_high_dimensional.debug("Dimensions: %s", dimensions)
        logger_high_dimensional.debug("Sampling orders for each dimension:")
        for i, o in enumerate(orders):
            logger_high_dimensional.debug("Dimension %d order: %s", i, o)

    # Calculate maximum possible samples
    maximum_n = 1
    for d in dimensions:
        maximum_n *= d
    lcm = math.lcm(*dimensions)

    if lcm != maximum_n:
        logger_high_dimensional.debug(
            "Periodicity detected, the sampling subroutine will ensure that you will not sampple"
            "the same configuration more than once."
        )

    if isinstance(n, str):
        if n == "all":
            n = maximum_n
        elif n == "max":
            n = max(dimensions)
        else:
            raise ValueError(f"Unrecognized string for n: {n}")

    if n > maximum_n:
        logger_high_dimensional.warning(
            f"Maximal sample size is {maximum_n}, you requested {n} sampling presciptions."
            f"Elaborating prescription for n_samples = {maximum_n}"
        )

    logger_high_dimensional.debug(
        "Preparing to sample %d out of %d possible points.", n, maximum_n
    )

    match strategy:
        case "random":
            return random_high_dimensional_sampling(dimensions, n, seed=seed)
        case "clhs":
            return concatenated_latin_hypercube_sampling(
                dimensions=dimensions, final_sample_size=n, seed=seed
            )
        case "sobol":
            return sobol_sampling(dimensions=dimensions, final_sample_size=n, seed=seed)
        case _:
            raise NotImplementedError(f"Strategy {strategy} is unknown")
