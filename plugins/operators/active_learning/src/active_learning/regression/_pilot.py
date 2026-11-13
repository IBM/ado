# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Structured pilot sampling over the original dimensions of a finite pool."""

from __future__ import annotations

import random
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import qmc

PilotStrategy = Literal["random", "clhs", "sobol"]


class _StructuredPilot:
    """Generate unique eligible rows from CLHS or scrambled Sobol coordinates."""

    def __init__(
        self, features: np.ndarray, strategy: PilotStrategy, seed: int
    ) -> None:
        """Encode original property levels in their finite-pool encounter order."""

        self.coordinates = np.column_stack(
            [
                pd.factorize(features[:, column], sort=False, use_na_sentinel=False)[0]
                for column in range(features.shape[1])
            ]
        )
        self.dimensions = self.coordinates.max(axis=0) + 1
        self.strategy = strategy
        self._clhs_rng = random.Random(seed)  # noqa: S311
        self._level_pools = [list(range(size)) for size in self.dimensions]
        self._fallback_rng = np.random.default_rng(seed)
        self._sobol = (
            qmc.Sobol(d=len(self.dimensions), scramble=True, rng=seed)
            if strategy == "sobol"
            else None
        )
        self._sobol_points = np.empty((0, len(self.dimensions)))
        self._sobol_offset = 0

    def _next_coordinates(self) -> tuple[int, ...]:
        """Advance the chosen design and return one discrete coordinate tuple."""

        if self._sobol is not None:
            if self._sobol_offset == len(self._sobol_points):
                generated = self._sobol.num_generated
                power = max(generated.bit_length() - 1, 0)
                self._sobol_points = self._sobol.random_base2(power)
                self._sobol_offset = 0
            point = self._sobol_points[self._sobol_offset]
            self._sobol_offset += 1
            return tuple((point * self.dimensions).astype(int))

        coordinates = []
        for column, size in enumerate(self.dimensions):
            if not self._level_pools[column]:
                self._level_pools[column] = list(range(size))
            levels = self._level_pools[column]
            coordinates.append(levels.pop(self._clhs_rng.randrange(len(levels))))
        return tuple(coordinates)

    def sample(self, candidates: np.ndarray, count: int) -> np.ndarray:
        """Draw eligible unique rows, filling sparse-grid gaps with seeded sampling.

        Proposals for unavailable coordinates are skipped. After a bounded
        number of attempts, remaining rows are drawn uniformly without
        replacement from the eligible candidates.

        Raises:
            ValueError: If count exceeds the number of eligible candidates.
        """

        if count > candidates.size:
            raise ValueError(
                f"structured pilot requested {count} candidates; "
                f"only {candidates.size} available"
            )

        available: dict[tuple[int, ...], list[int]] = {}
        for index in candidates:
            key = tuple(self.coordinates[index])
            available.setdefault(key, []).append(int(index))
        chosen: list[int] = []
        for _ in range(max(128, 32 * count)):
            key = self._next_coordinates()
            matching = available.get(key)
            if not matching:
                continue
            chosen.append(matching.pop(0))
            if not matching:
                del available[key]
            if len(chosen) == count:
                return np.asarray(chosen, dtype=np.int64)

        remaining = np.asarray(
            [index for indices in available.values() for index in indices],
            dtype=np.int64,
        )
        chosen.extend(
            self._fallback_rng.choice(
                remaining, size=count - len(chosen), replace=False
            ).tolist()
        )
        return np.asarray(chosen, dtype=np.int64)
