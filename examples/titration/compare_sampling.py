# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Compare FLORA, PKH, and random sampling on the titration example."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml
from active_learning.flora.parameters import FLORAParameters
from active_learning.flora.sampler import FLORASelectionState
from active_learning.pkh.parameters import PKHParameters
from active_learning.pkh.sampler import PKHState
from active_learning.sampler import build_forest
from titration_custom_experiments.experiments import (
    SAMPLE_VOLUME_ML,
    WATER_ION_PRODUCT,
    titrate_strong_acid,
)

from ado.schema.entityspace import EntitySpaceRepresentation
from ado.schema.property import ConstitutiveProperty

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:  # pragma: nocover
    from yaml import SafeLoader  # type: ignore[assignment]


def compare_sampling(seeds: range, budget: int) -> pd.DataFrame:
    """Return unmeasured-entity RMSE and transition coverage for each method and seed.

    The pool follows the space YAML's sequential grid order. Selection uses the
    operators' states and deterministic experiment readings. Final forests use
    candidate-pool training order and each operator's finalization procedure.
    """
    directory = Path(__file__).parent / "example_yamls"
    space = yaml.load(
        (directory / "space_titration.yaml").read_text(), Loader=SafeLoader
    )
    entity_space = EntitySpaceRepresentation(
        constitutiveProperties=[
            ConstitutiveProperty.model_validate(prop) for prop in space["entitySpace"]
        ]
    )
    frame = pd.DataFrame(
        entity_space.sequential_point_iterator(),
        columns=[prop.identifier for prop in entity_space.constitutiveProperties],
    )
    features = frame.to_numpy(dtype=float)
    if not 2 <= budget < len(frame):
        raise ValueError(f"budget must be between 2 and {len(frame) - 1}.")
    labels = np.array(
        [titrate_strong_acid(**row)["ph"] for row in frame.to_dict("records")]
    )
    acid, base, volume = (
        frame["acid_concentration"],
        frame["base_concentration"],
        frame["titrant_volume"],
    )
    net = (acid * SAMPLE_VOLUME_ML - base * volume) / (SAMPLE_VOLUME_ML + volume)
    truth = -np.log10((net + np.sqrt(net**2 + 4 * WATER_ION_PRODUCT)) / 2)
    near = (volume - acid * SAMPLE_VOLUME_ML / base).abs() < 8
    configurations = {
        name: yaml.load(
            (directory / f"operation_{name}.yaml").read_text(), Loader=SafeLoader
        )["operation"]["parameters"]
        for name in ("flora", "pkh")
    }
    results = []
    for seed in seeds:
        for name in ("flora", "pkh", "random"):
            parameter_class = PKHParameters if name == "pkh" else FLORAParameters
            parameters = parameter_class.model_validate(
                {
                    **configurations["flora" if name == "random" else name],
                    "seed": seed,
                    "numberEntities": budget,
                }
            )
            if name == "random":
                indices = np.random.default_rng(seed).choice(
                    len(features), budget, replace=False
                )
            else:
                state_class = PKHState if name == "pkh" else FLORASelectionState
                state = state_class(
                    features,
                    initial_measured_indices=[],
                    initial_labels=[],
                    params=parameters,
                )
                for _ in range(budget):
                    index = state.select_next()
                    if index is None:
                        raise RuntimeError(
                            "The sampler exhausted the pool before the budget."
                        )
                    state.record_measurement(index, float(labels[index]))
                indices = np.array(state.measured_indices)
            # PredictorExport enumerates its measured training rows in pool order.
            indices = np.sort(indices)
            if name == "pkh":
                final_state = PKHState(
                    features,
                    initial_measured_indices=indices.tolist(),
                    initial_labels=labels[indices].tolist(),
                    params=parameters,
                )
                final_state.fit_forest_and_histograms()
                model = final_state.forest
                if model is None:
                    raise RuntimeError("PKH did not fit its final forest.")
            else:
                model = build_forest(parameters, seed).fit(
                    features[indices], labels[indices]
                )
            unmeasured = np.setdiff1d(np.arange(len(features)), indices)
            error = model.predict(features[unmeasured]) - truth.to_numpy()[unmeasured]
            results.append(
                {
                    "method": name,
                    "seed": seed,
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "measured_near_equivalence": int(near.iloc[indices].sum()),
                    "measured_count": len(indices),
                }
            )
    return pd.DataFrame(results)


def main() -> None:
    """Run the comparison for consecutive seeds and optionally save per-seed results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", type=int, default=30, help="Number of consecutive seeds."
    )
    parser.add_argument("--seed", type=int, default=0, help="First seed (default: 0).")
    parser.add_argument(
        "--budget", type=int, default=60, help="Measurements per method."
    )
    parser.add_argument(
        "--output", type=Path, help="Write per-seed results to this CSV."
    )
    args = parser.parse_args()
    if args.seeds < 1 or args.seed < 0 or args.seed + args.seeds > 2**32:
        parser.error("Choose at least one seed, all between 0 and 2**32 - 1.")
    try:
        results = compare_sampling(
            range(args.seed, args.seed + args.seeds), args.budget
        )
    except ValueError as error:
        parser.error(str(error))
    print(f"NumPy {np.__version__}, scikit-learn {sklearn.__version__}")
    print(f"Budget {args.budget}; seeds {args.seed}-{args.seed + args.seeds - 1}")
    for name, rows in results.groupby("method", sort=False):
        low, high = rows["rmse"].quantile([0.25, 0.75])
        print(f"{name:7s} median {rows['rmse'].median():.2f}, IQR {low:.2f}-{high:.2f}")
    errors = results.pivot(index="seed", columns="method", values="rmse")
    wins = int((errors["flora"] < errors["pkh"]).sum())
    print(f"FLORA lower than PKH in {wins} of {args.seeds} seeds")
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"Per-seed results: {args.output}")


if __name__ == "__main__":
    main()
