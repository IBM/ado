# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import itertools
import logging
import math
import random
from collections.abc import Sequence
from typing import Any

import numpy as np
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
def distinct_sobol_sampling(dims, n, seed=None):
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


# NOTE: this will be deprecated soon
def recursive_aggregation_high_dimensional_sampling(
    dims: list[int],
    n: int,
    orders: list[list[int]] | None = None,  # the already sampled ?
    random_shifts: bool = False,
) -> list[list[int]]:
    """
    Returns a list of points to sample using a recursive aggregation strategy.

    Args:
        dims (List[int]): Sizes of each dimension (e.g., [8, 5]).
        n (int): Number of points to sample
        random_shifts (bool):

    This function progressively aggregates dims to reduce a high dimensional problem to a two dimensional one.
    By default, the order of dims is NOT changed.
    Consecutively sampled points are proxies for uniformly distributed points.

    For example, say len(dims) = 2 and dims[0] = 5, this means that there are 5 possible values for this dimensions, 0 represents the smallest
    and 4 represent the biggest. From the 1d analysis I know that I will sample in this 1d, 5 points segment points in a certain position order:

        - 1st point: 0 (smallest)
        - 2nd point: 4 (biggest)
        - 3rd point: 2 (medium)
        - 4th point: 1 (continuing to fill the space)
        - 5th point: 3 (the remaining one)

    Regardless of what the other dimensions are, I want to keep this sampling order for each of them,

        e.g. `(smallest) -> (biggest) -> (medium) -> ...` or `[0,4,2,1,3]`, let this list be called "position list"

    If dims[1] = 5 (= dims[0]), there will be 25 (= dims[1] * dims[0]) combinations of points (the len of this function output with the maximal
    admissible value of n is 25).
    The combination [0,0] means that the smallest point for both dimensions, however after we pick the five [i,j], i=j combination
    We need to go to the combinations [i,j], i!=j, to keep the ordering as much as possible we can **shift** the position list
    of one of the two dimensions `[0,4,2,1,3]` (say the last one)
    by one or more element, say we shift this by one element, `[0,4,2,1,3]` becomes `[4,2,1,3,0]`.
    In this way we will sample [0,4] -> [4,2] -> [2,1] -> [1,3] -> [3,0] (-> ...).
    Note: without a shift we will sample again [i,j], i=j combinations.
    Other 15 points needs to be measured, these correspond to shifting the second list by 2, 3 and 4, as we already shifted by 1 and
    shifting by 5 corresponds to no shift at all.
    To improve distribution of points we want to order the shifts 0 to 4 of the second list at random, each shift corresponds to a batch of
    orders elements (such as [0,4], [4,2], [2,1], [1,3], [3,0], in this case they are 5).
    We select the first `n` of such elements.

    This function generalize this procedure for an arbitrary numbers of elements in the dims list.
    There is no a priori guarantee that this strategy (when plugged in `get_order_list_nn_high_dimensional`) is better than the other.
    We will benchmark different strategies across a combinations of problems and see if one emerges as the best.

    Let us sketch how this function works.
    WLOG be k = len(dims) and be the position list be 1,2,3,...,dims[mu] for each mu = 1, ..., k.
    Note however that you ultimately want to fill a high dimensional space
    as progressively uniformly as possible.
    Taking these wlog lists make difficult to sketch the real points that would be sampled in the space and thus assess how much the
    procedure is progressively uniformly filling the space.
    Obs: if the k dims are coprime we can cycle go through their lists and we are guaranteed that we will not sample the same configuration
    twice.

    Once I do lcm numbers of combinations, I have the guarantee that whatever combination I shift and whateve shift I take (different from the identity shift)
    I will then sample lcm new combinations. `number_repetitons = maximum_n / lcm` is the number of shift k-tuples I need to explore to cover the whole space.
    In the dims = [5,5] examble, number_repetitons = 25/ 5 = 5, we needed 5 shifts indeed.
    If dims = [5,5,7] we need number_repetitons = 25 * 7 / (5 * 7) = 5 shifts but the first lcm = 35 points will be sampled with no repetitions.

    ### The problem with this approach
    However these points will be of the kind [i,j,k] with i=j, and this is NOT what we ultimatively want.
    Instead we would like to maintain uniformity in the first 2-dimensional space. That is, if we ignore the k-value we should be left with a sequence of
    progressively uniform points in the 5*5 2dimensional space.

    ### The intuition behind the solution
    This calls out for a iterative procedure:
    We first set the order for the first 25 points. The third dimension will then change as we navigate through them.
    The problem of [5,5,7] is then reduced to the problem of [25, 7], they are coprime and thus easy to navigate.

    ## The solution
    We want to reduce the length of dims!
    The invariant quantity of this reduction is the product of the dims elements.
    We then need to deal with k = 2 cases only.
    Coprime situation and dims[0] = dims[1] situation have already been discussed.
    We need dims[0] and dims[1] share a factor.
    say dims[1] is p*dims[0]:
    # I AM NOT SURE ABOUT THIS, because shift should be equal to gcd
        p is coprime with dims[1]  (think about dims = [2,6])) (note here gcd !=p):
        OR: p is a factor of dims[1] (as in dims = [2,4])
            The strategy does not change: we need gcd different combinations of shifts, for the dim[0] axis, represented by 0 to gcd-1.
    say dims[1] = alpha * p  and dims[0] = beta * p (dims = [6,4]):
        alpha and beta coprime (wlog, we put factors in p = gcd), p coprime with them (dims = [6,4])
            lcm = alpha * beta * p, tot = alpha * beta * p^2. we need p different combinations of shifts, as before.
        alpha and beta coprime (wlog, we put factors in p), p not coprime with both them (dims = [6,8]), 6 = 3*2, 8= 4 * 2 p=2
            lcm = dims[0] * dims[1] / p (= 24 in this case), we need p  shifts (2, one more than the trivial one) to reach the product of dimensions.

    In all these cases, the number of shifts needed is equal to the greatest common divisor.
    We have illustrated the procedure.

    Thus, the function uses this strategy.

    ## How this works inside our function
    The easiest way to work with this is aggregating two dimensions at the time, the ones at the beginning of dims list, 2 at the time,
    the last aggregated dimension will be the last one.  Note that, the ordering of dims list matters.
    The selection of the shifts is deterministic by default.
    dims = [3,6,2,4] will become dims = [18,2,4] and then dims = [36,4] and finally we will aggregate the last two and return the results.
    The aggregation history is kept in a dict.

    NOTE: there is no guarantee that
    ::
    recursive_aggregation_high_dimensional_sampling(dims, n=4)[1] == [
            dims[0] - 1,
            dims[1] - 1,
            dims[2] - 1,
        ]
    """

    if not orders:
        orders = []

    cprod = np.cumprod(np.array(dims), dtype=int).tolist()
    maximum_n = cprod[-1]

    lcm = math.lcm(*dims)
    gcd = math.gcd(*dims)
    k = len(dims)
    if n > maximum_n:
        logging.warning(
            f"Obtaining {n} is not possible, setting it to the maximal value {maximum_n}."
        )
        n = maximum_n

    logging.info(
        f"After {lcm} points you will encounter periodicity,"
        "We use the shift strategy to avoid periodicity."
        f"If no action is taken instead, the set of {lcm} numbers will be repeated {gcd} times."
    )

    logging.info(
        f"The greatest cardinality is {max(dims)}, multiplicity is {len([d for d in dims if d == max(dims)])}."
    )

    # NOTE: we fix the shifts a priori because they need to be consistent in the for loop below
    permutations_dict = {}
    for i in range(len(dims) - 1, -1, -1):
        first = cprod[i - 1]
        last = dims[i]
        if dims[i] != cprod[i] / cprod[i - 1]:
            raise ValueError("Dimension mismatch")
        gcd = math.gcd(first, last)
        plist = list(range(gcd))
        if random_shifts:
            random.shuffle(plist)
        permutations_dict[(first, last)] = plist

    # def get_1d_sampled_points_from_orders(
    #     orders: list[list[int]], permutations_dict: dict, dims: list[int]
    # ): ...
    # sampled_points = get_1d_sampled_points_from_orders(orders, permutations_dict, dims)
    # TODO: obtain sampled_indeces from orders, to account properly for previously sampled points,
    one_d_representation_list = get_index_list_nn(
        length_segment=maximum_n, tot_points_to_sample=n
    )

    history_dict: list[dict] = []
    res = []
    for el in one_d_representation_list:
        out = [el]
        while len(out) < k:
            # i.e. if len(out) = 1, I need to divide in two pieces, one for the fictitious dimension cprod[-len(out)-1] and one for last dimension,
            # dims[-1] == dims[-len(out)]
            # these pieces replaces out. Note that the second one directly tells you which element to select in the last dimension.
            # Then:
            # if len(out) = 2, I need to obtain two pieces from out[:1], one for cprod[-len(out)-1] and one dims[-2] == dims[-len(out)].
            # Note that the second one directly tells you which element to select in the second-to-last dimension.
            # Also note  that dims[-2] == cprod[-1] / cprod[-2]
            # these two pieces will replace the element these pieces where obtained from.

            # The shifts come into play because assigning a couple to a number, given cprod[-len(out)-1] and dims[-len(out)], still has a degree of freedom, that is the
            # shift that characterizes numbers x in the interval [gcd * q , gcd * (q+1)]  (q in N). gcd here is between cprod[-len(out)-1] and dims[-len(out)].
            # given a permutation of [0,..., gcd - 1], I am able to infer the interval

            # first = cprod[-len(out)]
            first = cprod[-len(out) - 1]
            last = dims[-len(out)]
            this_iter_gcd = math.gcd(first, last)
            this_iter_lcm = math.lcm(first, last)
            this_iter_index = out[0]
            this_iter_plist = permutations_dict[(first, last)]

            try:
                this_shift = this_iter_plist[this_iter_index // this_iter_lcm]
            except IndexError:
                logging.warning(
                    f"list index out of range, this_iter_index = {this_iter_index}, lcm = {lcm}, plist = {this_iter_plist}"
                )

            index_within_shift = this_iter_index % this_iter_lcm  # remainder
            # The delta i-j of the couple i,j, capped at [first-1, last-1] is equiv this shift mod gcd.
            # Within
            i = index_within_shift % first
            j = (index_within_shift + this_shift) % last

            # Modifying the out list
            out = [i, j, *out[1:]]

            this_iter_history = {}
            this_iter_history["index_to_last"] = len(out)
            this_iter_history["first_last_tuple"] = (first, last)
            this_iter_history["ij_tuple"] = (i, j)
            this_iter_history["gcd"] = this_iter_gcd
            this_iter_history["permutation_shift_list"] = this_iter_plist
            this_iter_history["shift_at_index"] = this_shift
            this_iter_history["this_iter_index"] = this_iter_index
            logging.debug(this_iter_history)
            history_dict.append(this_iter_history)

            shift_check, index_check = identify_shift_and_position(first, last, i, j)
            if this_shift != shift_check:
                logging.error(
                    f"Shift mismatch:\ninferred={shift_check},\texpected={this_shift}"
                )
            if index_within_shift % this_iter_lcm != index_check % this_iter_lcm:
                logging.error(
                    f"Index within shift mismatch:\ninferred={index_check},\texpected={index_within_shift}"
                )
        res.append(out)
    return res


def shift_from_permutation(index: int, mcm: int, permutation: list[int]) -> int:
    """For debugging purposes"""
    gcd = len(permutation)
    tot = mcm * gcd
    if index < 0 or index > tot - 1:
        raise ValueError("Invalid index")
    div = index // gcd
    return permutation[div]


def identify_shift_and_position(
    m: int, n: int, i: int, j: int, first_index_is_0: bool = True
):
    """
    Identify the shift of the second list and the position at which the
    observed pair appears in the shifted zip.

    Parameters
    ----------
    m, n : int
        Lengths of the two lists.
    i, j : int
        Observed pair (1-based indexing).

    Returns
    -------
    s : int
        The shift of the second list modulo gcd(m, n).
    k : int
        The index (0-based) at which (i, j) appears in the shifted list.

    Mathematical setting
    --------------------
    Let m, n ∈ N and define, for a shift s ∈ Z,

        l_{c,s}[k] = ((k mod m) + 1, ((k + s) mod n) + 1),
        k = 0, …, lcm(m, n) - 1.

    Given an observed pair (i, j), we want to recover:
      (a) the shift s (as much as it is identifiable),
      (b) the index k at which (i, j) appears in l_{c,s}.

    This means solving for integers (k, s) such that

        (1)  k ≡ i - 1 (mod m),
        (2)  k + s ≡ j - 1 (mod n).

    Shift identification
    --------------------
    Subtracting (1) from (2) gives

        s ≡ (j - 1) - (i - 1) = j - i   (mod d),

    where d = gcd(m, n). Hence the shift is identifiable uniquely modulo d.
    We choose the canonical representative

        s = (j - i) mod d.

    Position identification
    -----------------------
    Fixing this s, the system becomes

        k ≡ i - 1           (mod m),
        k ≡ j - 1 - s       (mod n).

    The right-hand sides are congruent modulo d by construction, so by the
    Chinese Remainder Theorem the system admits a solution k, unique modulo
    lcm(m, n).

    An explicit solution is obtained as follows. Let

        d = gcd(m, n),
        m' = m / d,   n' = n / d,
        a = i - 1,
        b = j - 1 - s.

    Then k = a + m x, where x solves

        m' x ≡ (b - a)/d   (mod n').

    Since gcd(m', n') = 1, m' is invertible modulo n', yielding a unique x
    modulo n'.
    """
    if first_index_is_0:
        i += 1
        j += 1

    d = math.gcd(m, n)

    # shift (canonical representative modulo d)
    s = (j - i) % d

    # solve for k
    a = i - 1
    b = j - 1 - s

    m_, n_ = m // d, n // d
    rhs = (b - a) // d

    inv_m = pow(m_, -1, n_)
    x = (rhs * inv_m) % n_

    k = (a + m * x) % (m * n_)

    return s, k


# NOTE: this will be soon deprecated
def one_shift_then_random_points_high_dimensional_sampling(
    dims: list[int], orders: list[list[int]], n: int
):

    sampled_list = []
    pointer = 0  # Tracks position across dimensions

    while len(sampled_list) < n:
        logging.debug(
            f"Sampling point number (starting from 0),\t{len(sampled_list)+1}"
        )
        el = []
        inner_indeces = []
        for _i, order in enumerate(orders):
            index_inner = pointer % len(order)  # len(order) == dim by construction
            inner_indeces.append(index_inner)

        # how many inner_indexes are 0?
        zeros = inner_indeces.count(0)
        zeros_indeces = [i for i, e in enumerate(inner_indeces) if e == 0]
        if zeros > 1 and len(sampled_list) > 0:
            logging.info(f"Detected periodicity on the following {zeros} dimensions:")
            for shift, z in enumerate(zeros_indeces):
                logging.info(f"dims_{z} = {dims[z]}, shifting related order by {shift}")
                for _ in range(shift):
                    orders[z].insert(0, orders[z].pop(-1))

        # Now that I shifted the orders I continue as before
        for i, order in enumerate(orders):
            index_inner = pointer % len(order)  # len(order) == dim by construction
            logging.debug(
                f"From order_{i}={order},\tAdding order_{i}[{index_inner}]={order[index_inner]}\n"
            )
            el.append(order[index_inner])

        # NOTE: I am accumulating technical debt here because some times you do not need to shift, see 3x3x3 example on tablet
        logging.debug(f"\nAppending element {el}")
        sampled_list.append(el)
        pointer = pointer + 1

    logging.debug(
        f"Updating pointer to pointer+1={pointer}, len(sampled_list) = {len(sampled_list)}\n"
    )

    filtered_list = unique_in_order_list_of_lists(sampled_list)
    if len(filtered_list) != n:
        logging.warning(
            f"""Warning! Sample list contains duplicates, they will be filtered.
            The high-dimensional LHS-like algorithm detects {len(filtered_list)} points instead of {n}."""
        )

        points_to_add = n - len(filtered_list)
        points_added = []
        logging.info(
            f"Adding {points_to_add} points. These additions may make the sample unbalanced."
        )

        # Generate all possible configurations using Cartesian product
        # Each configuration is a tuple where element i ranges from 0 to dims[i]
        all_configs = list(itertools.product(*[range(d + 1) for d in dims]))

        # Remove configurations already present in filtered_list
        existing_set = {tuple(x) for x in filtered_list}
        available_configs = [cfg for cfg in all_configs if cfg not in existing_set]

        # Randomly sample the required number of unique configurations
        while len(points_added) < points_to_add and available_configs:
            candidate = random.choice(available_configs)  # noqa: S311
            points_added.append(list(candidate))
            available_configs.remove(candidate)

        # Merge newly added points into the filtered list
        filtered_list.extend(points_added)

    return filtered_list


def random_high_dimensional_sampling(dims: list[int], n: int, seed: int | None = None):
    """
    Generates n unique random samples from a high-dimensional space.

    Args:
        dims (List[int]): Cardinality (size) of each dimension. Must be positive.
        n (int): Total number of points to sample.
        seed (Optional[int]): Optional PRNG seed for reproducibility.
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
            - 'one_shift': refer to one_shift_then_random_points_high_dimensional_sampling
            - 'recursive_aggregation': refer to recursive_aggregation_high_dimensional_sampling
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
        return random_high_dimensional_sampling(dims, n)

    if strategy == "one_shift":
        return one_shift_then_random_points_high_dimensional_sampling(dims, orders, n)

    if strategy == "recursive_aggregation":
        return recursive_aggregation_high_dimensional_sampling(dims=dims, n=n)

    if strategy == "clhs":
        return concatenated_latin_hypercube_sampling(dims=dims, n=n, seed=SEED)

    if strategy == "sobol":
        return sobol_sampling(dims=dims, n=n, seed=SEED)

    raise NotImplementedError(f"Strategy {strategy} is unknown")


def unique_in_order_list_of_lists(lists: Sequence[Sequence[Any]]):
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


def plot_grid(ax, dims, points, title) -> None:
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
