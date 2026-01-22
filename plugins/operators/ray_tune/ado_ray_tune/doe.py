# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from scipy.stats import qmc


class LatinHypercubeSampler:

    def __init__(self, dict_space: dict) -> None:
        self.dict_space = dict_space
        space_int_repr = []
        ts_orig_to_int = {}
        ts_int_to_orig = {}
        l_bounds = []
        u_bounds = []
        skipped_orig_values = {}
        total_size = 1
        i = 0
        index_to_label = {}
        for label, dim in dict_space.items():
            num_points_in_dim = len(dim)
            if num_points_in_dim < 2:
                skipped_orig_values[label] = dim
                continue
            total_size *= num_points_in_dim
            nd = list(range(num_points_in_dim))  # codespell:ignore nd
            l_bounds.append(nd[0])  # codespell:ignore nd
            u_bounds.append(nd[-1] + 1)  # codespell:ignore nd
            tti = dict(zip(dim, nd, strict=True))  # codespell:ignore nd
            tto = dict(zip(nd, dim, strict=True))  # codespell:ignore nd
            space_int_repr.append(nd)  # codespell:ignore nd
            index_to_label[i] = label
            i += 1
            ts_orig_to_int[label] = tti
            ts_int_to_orig[label] = tto
        self.total_size = total_size
        self.space_int_repr = space_int_repr
        self.ts_orig_to_int = ts_orig_to_int
        self.ts_int_to_orig = ts_int_to_orig
        self.l_bounds = l_bounds
        self.u_bounds = u_bounds
        self.skipped_orig_values = skipped_orig_values
        self.index_to_label = index_to_label

        self.d = len(space_int_repr)
        self.sampler = qmc.LatinHypercube(d=self.d)
        # sample = sampler.random(2*self.d)
        # sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
        print(
            f"[LatinHypercubeSampler] configured for {self.d} actual dimensions, total size of search space is {self.total_size}."
        )

    def generate_new_categorical_samples(
        self, n: int | None = None, n_factor: int = 2
    ) -> list[dict]:
        if n is None:
            n = n_factor * self.d
        # requires scipy>=1.11.3
        sample_ints = self.sampler.integers(
            l_bounds=self.l_bounds, u_bounds=self.u_bounds, n=n
        )
        print(f"[LatinHypercubeSampler] Generated {n} samples (by request).")
        ret_list = []
        for sample in sample_ints:
            ret = {}
            for cur_i, dim_e in enumerate(list(sample)):
                label = self.index_to_label[cur_i]
                # ret[label] = []
                # ret[label].append(self.ts_int_to_orig[label][dim_e])
                ret[label] = self.ts_int_to_orig[label][dim_e]
            for k, v in self.skipped_orig_values.items():
                ret[k] = v[0]
            ret_list.append(ret)
        return ret_list
