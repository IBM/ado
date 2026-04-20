# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for the no-priors sampler in core discoveryspace."""

import pytest
from pydantic import ValidationError

from orchestrator.core.discoveryspace.no_priors_parameters import NoPriorsParameters
from orchestrator.core.discoveryspace.no_priors_sampler import NoPriorsSampleSelector


class TestNoPriorsParameters:
    """Test NoPriorsParameters model."""

    def test_default_parameters(self) -> None:
        """Test default parameter values."""
        params = NoPriorsParameters(targetOutput="test_target")
        assert params.targetOutput == "test_target"
        assert params.samples == 20
        assert params.batchSize == 1
        assert params.sampling_strategy == "clhs"

    def test_custom_parameters(self) -> None:
        """Test custom parameter values."""
        params = NoPriorsParameters(
            targetOutput="custom_target",
            samples=50,
            batchSize=5,
            sampling_strategy="sobol",
        )
        assert params.targetOutput == "custom_target"
        assert params.samples == 50
        assert params.batchSize == 5
        assert params.sampling_strategy == "sobol"

    def test_case_insensitive_strategy(self) -> None:
        """Test that sampling_strategy is case-insensitive."""
        params = NoPriorsParameters(targetOutput="test", sampling_strategy="CLHS")
        assert params.sampling_strategy == "clhs"

        params = NoPriorsParameters(targetOutput="test", sampling_strategy="Sobol")
        assert params.sampling_strategy == "sobol"

    def test_invalid_strategy(self) -> None:
        """Test that invalid strategy raises validation error."""
        with pytest.raises(ValidationError, match="sampling_strategy"):
            NoPriorsParameters(targetOutput="test", sampling_strategy="invalid")

    def test_samples_validation(self) -> None:
        """Test that samples must be >= 1."""
        with pytest.raises(ValidationError, match="samples"):
            NoPriorsParameters(targetOutput="test", samples=0)

        with pytest.raises(ValidationError, match="samples"):
            NoPriorsParameters(targetOutput="test", samples=-1)

    def test_batch_size_validation(self) -> None:
        """Test that batchSize must be >= 1."""
        with pytest.raises(ValidationError, match="batchSize"):
            NoPriorsParameters(targetOutput="test", batchSize=0)


class TestNoPriorsSampleSelector:
    """Test NoPriorsSampleSelector sampler."""

    def test_sampler_initialization(self) -> None:
        """Test sampler can be initialized with parameters."""
        params = NoPriorsParameters(targetOutput="test_target", samples=10)
        sampler = NoPriorsSampleSelector(parameters=params)
        assert sampler.params == params
        assert sampler.params.targetOutput == "test_target"
        assert sampler.params.samples == 10

    def test_parameters_model(self) -> None:
        """Test that parameters_model returns correct type."""
        assert NoPriorsSampleSelector.parameters_model() == NoPriorsParameters

    def test_sampler_compatible_with_discovery_space_remote(self) -> None:
        """Test that sampler reports compatibility with any discovery space."""
        # This is a simple compatibility check - always returns True
        # We don't need a real DiscoverySpaceManager for this test
        assert NoPriorsSampleSelector.samplerCompatibleWithDiscoverySpaceRemote(None)

    def test_entity_iterator_not_implemented(self) -> None:
        """Test that entityIterator raises NotImplementedError."""
        params = NoPriorsParameters(targetOutput="test_target")
        sampler = NoPriorsSampleSelector(parameters=params)

        # entityIterator is not implemented for this sampler
        # The NotImplementedError is raised when the iterator is called
        iterator = sampler.entityIterator(discoverySpace=None, batchsize=1)
        with pytest.raises(NotImplementedError):
            next(iterator)


# Made with Bob
