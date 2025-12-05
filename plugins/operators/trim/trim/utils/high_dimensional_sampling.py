# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

# NOTE: At the moment 27 Nov,
# High D sampling by default adopts a naive way to limit periodicity,
# for extremely small spaces requires additional/random sampling -> this is done but can be improved
# get_index_list_nn_high_dimensional is the function.


import logging
import math

from trim.utils.one_dimensional_sampling import get_index_list_nn


def unique_in_order_list_of_lists(lists):
    seen = set()
    out = []
    for row in lists:
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def refined_high_dimensional_sampling(dims, orders, n):

    sampled_list = []
    pointer = 0  # Tracks position across dimensions

    while len(sampled_list) < n:
        logging.debug(
            f"Sampling point number (starting from 0),\t{len(sampled_list)+1}"
        )
        el = []
        inner_indeces = []
        for i, order in enumerate(orders):
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

    # TODO: think about a look up, you associate a number from 0 to (len(maximum_n), - 1)
    #  so that you can give directly the indexes of the ordered according to the dataframe
    # wrt this
    # USe
    import itertools
    import random

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
            candidate = random.choice(available_configs)
            points_added.append(list(candidate))
            available_configs.remove(candidate)

        # Merge newly added points into the filtered list
        filtered_list.extend(points_added)

    return filtered_list


def get_order_list_nn_high_dimensional(
    dims: list[int],
    n: int | str = "all",
    space: dict[str, int] | None = None,
    refined: bool = True,
) -> list[list[int]]:
    """
    Generate sampling indices for a high-dimensional space using `get_index_list_nn` for each dimension.

    Args:
        dims (List[int]): Sizes of each dimension (e.g., [8, 5]).
        n (int | str): Number of points to sample:
            - 'all': sample all possible combinations (product of dims)
            - 'max': sample up to max(dims)
        space (Optional[Dict[str, int]]): Optional mapping of dimension names to sizes for logging.
            Example:
                space = {'batch_size': 8, 'model_name': 5}

    Returns:
        List[List[int]]: Outer list length = n (or product of dims if n='all').
                        Each inner list contains one sampled combination across dimensions.
    """
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

    shift = 0
    if lcm != maximum_n:
        logging.warning(
            """Periodicity detected, you will eventually sample
                        the same configuration more than once, enabling shift"""
        )
        shift = 1

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

    # TODO: fix the function and make this the default behavior
    if refined:
        return refined_high_dimensional_sampling(dims, orders, n)

    index_of_first_max = int(dims.index(max(dims)))
    sampled_list = []

    pointer = 0  # Tracks position across dimensions

    while len(sampled_list) < n:
        logging.debug(
            f"Sampling point number (starting from 0),\t{len(sampled_list)+1}"
        )
        el = []
        if shift and pointer == max(dims):
            # the order list at the index_of_first_max undergoes a shift
            # the last elements goes at the beginning
            logging.debug("The order list at the index_of_first_max undergoes a shift")
            orders[index_of_first_max].insert(0, orders[index_of_first_max].pop(-1))

        for i, order in enumerate(orders):
            index_inner = pointer % len(order)  # len(order) == dim by construction
            logging.debug(
                f"From order_{i}={order},\tAdding order_{i}[{index_inner}]={order[index_inner]}\n"
            )
            el.append(order[index_inner])

        logging.debug(f"\nAppending element {el}")
        sampled_list.append(el)
        pointer = pointer + 1
        logging.debug(
            f"Updating pointer to pointer+1={pointer}, len(sampled_list) = {len(sampled_list)}\n"
        )

    return sampled_list


if __name__ == "__main__":
    # Example usage
    # result1 = get_index_list_nn_high_dimensional([3,2,8], n = 'max')
    # print("Result 1:", result1)
    result2 = get_order_list_nn_high_dimensional([3, 3, 3])
    print("Result 2:", result2)

# %%
