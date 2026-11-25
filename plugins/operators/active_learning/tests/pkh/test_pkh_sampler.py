# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""PKH selection, histogram updates, and refitting against explicit leaf embeddings."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
import ray
from active_learning.pkh.data import (
    candidate_entities,
    encode_feature_matrix,
    measured_labels,
    validate_candidate_space,
    validate_target_output,
)
from active_learning.pkh.parameters import PKHParameters
from active_learning.pkh.sampler import PKHSampleSelector, PKHState
from scipy.sparse import issparse
from sklearn.ensemble import RandomForestRegressor

from ado.core.discoveryspace.space import DiscoverySpace
from ado.modules.actuators.measurement_queue import MeasurementQueue
from ado.modules.operators.discovery_space_manager import DiscoverySpaceManager
from ado.schema.domain import PropertyDomain, VariableTypeEnum
from ado.schema.property import ConstitutiveProperty


@contextmanager
def _remote_space(space: DiscoverySpace) -> Iterator[ray.actor.ActorHandle]:
    """Run a discovery-space manager and release its Ray actors after the test."""
    queue = MeasurementQueue()
    manager = DiscoverySpaceManager.remote(space=space, queue=queue)
    try:
        yield manager
    finally:
        ray.kill(manager)
        queue.shutdown(force=True)


def test_herding_and_mmd_match_explicit_embeddings() -> None:
    """Complete the pilot from prior labels, then check herding and MMD across refits."""
    features = np.random.default_rng(7).normal(size=(12, 3))
    initial_indices = [0, 3, 7, 9]
    state = PKHState(
        features,
        initial_measured_indices=initial_indices,
        initial_labels=[float(features[index].sum()) for index in initial_indices],
        params=PKHParameters(
            targetOutput="response",
            numberEntities=4,
            pilotSize=5,
            nEstimators=8,
            epochLength=2,
            seed=2**32 - 1,
        ),
    )
    pilot = state.select_next()
    assert pilot is not None
    assert pilot not in initial_indices
    assert state.forest is None
    assert state.mmd_squared() is None
    state.record_measurement(pilot, float(features[pilot].sum()))

    forests = []
    for _ in range(3):
        selected = state.select_next()
        forest = state.forest
        assert selected is not None
        assert forest is not None
        forests.append(forest)
        leaves = forest.apply(features)
        embeddings = np.eye(leaves.max() + 1)[leaves].reshape(len(features), -1)
        embeddings /= np.sqrt(state.params.nEstimators)
        discrepancy = embeddings.mean(axis=0) - embeddings[state.measured_indices].mean(
            axis=0
        )
        scores = embeddings @ discrepancy
        scores[state.measured_indices] = -np.inf
        assert scores[selected] == pytest.approx(scores.max())
        assert state.mmd_squared() == pytest.approx(discrepancy @ discrepancy)
        state.record_measurement(selected, float(features[selected].sum()))
        updated_discrepancy = embeddings.mean(axis=0) - embeddings[
            state.measured_indices
        ].mean(axis=0)
        assert state.mmd_squared() == pytest.approx(
            updated_discrepancy @ updated_discrepancy
        )
    assert forests[0] is forests[1]
    assert forests[1] is not forests[2]


def test_candidate_entities_rejects_colliding_identifiers(
    make_pkh_space: Callable[[list[ConstitutiveProperty]], DiscoverySpace],
) -> None:
    """PKH rejects distinct candidates that share a measurement identifier."""
    prop = ConstitutiveProperty(
        identifier="k", propertyDomain=PropertyDomain(values=[1, "1"])
    )
    space = make_pkh_space([prop])
    with pytest.raises(ValueError, match="unique entity identifiers"):
        candidate_entities(space)


@pytest.mark.asyncio
async def test_sampler_requires_current_target_before_next_entity(
    categorical_space: DiscoverySpace,
) -> None:
    """PKH waits for the selected entity's target label before choosing another."""
    with _remote_space(categorical_space) as manager:
        parameters = PKHParameters(targetOutput="self", numberEntities=2)
        sampler = PKHSampleSelector(parameters)
        assert sampler.samplerCompatibleWithDiscoverySpaceRemote(manager)
        stream = await sampler.remoteEntityIterator(manager)
        [first] = await anext(stream)
        target = validate_target_output(parameters.targetOutput, categorical_space)
        assert first.identifier not in measured_labels(categorical_space, target)
        with pytest.raises(ValueError, match="Missing finite target measurement"):
            await anext(stream)


@pytest.mark.asyncio
async def test_sampler_stops_on_fully_measured_grid(
    categorical_space: DiscoverySpace,
    make_pkh_space: Callable[[list[ConstitutiveProperty]], DiscoverySpace],
) -> None:
    """A completed grid should yield no new measurement requests."""
    kind, count = categorical_space.entitySpace.constitutiveProperties
    space = make_pkh_space(
        [
            kind.model_copy(update={"propertyDomain": PropertyDomain(values=["b"])}),
            count,
        ]
    )
    target = validate_target_output("self", space)
    assert len(measured_labels(space, target)) == validate_candidate_space(
        space.entitySpace
    )
    with _remote_space(space) as manager:
        sampler = PKHSampleSelector(
            PKHParameters(targetOutput="self", numberEntities=1)
        )
        stream = await sampler.remoteEntityIterator(manager)
        assert [batch async for batch in stream] == []


def test_sampler_rejects_unbounded_discrete_property(
    make_pkh_space: Callable[[list[ConstitutiveProperty]], DiscoverySpace],
) -> None:
    """An unbounded grid cannot supply the finite leaf distribution PKH needs."""
    prop = ConstitutiveProperty(
        identifier="temperature",
        propertyDomain=PropertyDomain(
            variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE,
            interval=1,
        ),
    )
    space = make_pkh_space([prop])
    with pytest.raises(ValueError, match=r"finite domain.*temperature"):
        validate_candidate_space(space.entitySpace)
    with _remote_space(space) as manager:
        assert not PKHSampleSelector.samplerCompatibleWithDiscoverySpaceRemote(manager)


def test_stored_candidates_match_coordinates_and_reject_ambiguity(
    make_pkh_space: Callable[[list[ConstitutiveProperty]], DiscoverySpace],
) -> None:
    """Retain custom IDs and reject multiple stored matches for a grid point."""
    prop = ConstitutiveProperty(
        identifier="code", propertyDomain=PropertyDomain(values=[1, "1", [{"key": 3}]])
    )
    space = make_pkh_space([prop])
    stored = space.entitySpace.entity_for_point({"code": 1}).model_copy(
        update={"identifier": "stored-number"}
    )
    unrelated = make_pkh_space(
        [ConstitutiveProperty(identifier="other", propertyDomain=prop.propertyDomain)]
    ).entitySpace.entity_for_point({"other": [{"key": 3}]})
    space.sample_store.addEntities([stored, unrelated])

    entities = candidate_entities(space)
    assert entities[0].identifier == stored.identifier
    assert [
        entity.valueForConstitutivePropertyIdentifier("code").value
        for entity in entities
    ] == prop.propertyDomain.values

    duplicate = stored.model_copy(update={"identifier": "another-stored-number"})
    space.sample_store.addEntities([duplicate])
    with pytest.raises(ValueError, match="Multiple stored entities"):
        candidate_entities(space)


def test_sparse_categorical_pool_preserves_forest_leaves() -> None:
    """Fit sparse one-hot features and recover the same leaves as a dense forest."""
    frame = pd.DataFrame({"code": pd.Categorical([f"code-{i}" for i in range(1000)])})
    features = encode_feature_matrix(frame)
    assert issparse(features)
    assert features.nnz == len(frame)
    indices = list(range(0, len(frame), 50))
    parameters = PKHParameters(
        targetOutput="response", numberEntities=10, nEstimators=8, minSamplesLeaf=1
    )
    states = [
        PKHState(
            matrix,
            initial_measured_indices=indices,
            initial_labels=indices,
            params=parameters,
        )
        for matrix in (features, features.toarray())
    ]
    for state in states:
        state.fit_forest_and_histograms()
    sparse_state, dense_state = states
    np.testing.assert_array_equal(
        sparse_state.forest.apply(features),
        dense_state.forest.apply(dense_state.features),
    )
    assert sparse_state.mmd_squared() == pytest.approx(dense_state.mmd_squared())


def test_forest_accepts_missing_numeric_features_with_categories() -> None:
    """A missing numeric input can share a forest with one-hot categorical inputs."""
    frame = pd.DataFrame(
        {"code": pd.Categorical(list("abcdefghij")), "reading": [np.nan, *range(9)]}
    )
    features = encode_feature_matrix(frame)
    forest = RandomForestRegressor(n_estimators=4, random_state=0)
    forest.fit(features, np.arange(len(frame)))
    assert np.isfinite(forest.predict(features)).all()
