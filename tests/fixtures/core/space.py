# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT


import json
import pathlib
from collections.abc import Callable
from typing import Any

import pytest
import yaml

from ado.core import ADOResource
from ado.core.discoveryspace.config import (
    DiscoverySpaceConfiguration,
)
from ado.core.discoveryspace.resource import DiscoverySpaceResource
from ado.metastore.sqlstore import SQLStore
from ado.schema.domain import PropertyDomain
from ado.schema.property import ConstitutiveProperty


@pytest.fixture
def random_space_resource_from_file(
    random_identifier: Callable[[], str],
) -> Callable[[str | None], DiscoverySpaceResource]:

    def _random_space_resource_from_file(
        sample_store_id: str | None = None,
    ) -> DiscoverySpaceResource:
        import datetime

        file = pathlib.Path("tests/resources/space/discoveryspace_resource.json")
        random_resource_id = random_identifier()
        if not sample_store_id:
            sample_store_id = random_identifier()

        # Get the model
        space = DiscoverySpaceResource.model_validate(json.loads(file.read_text()))

        # Final touch-ups
        space.identifier = random_resource_id
        space.config.sampleStoreIdentifier = sample_store_id
        space.created = datetime.datetime.now(datetime.timezone.utc)
        return space

    return _random_space_resource_from_file


@pytest.fixture
def random_space_resource_from_db(
    random_space_resource_from_file: Callable[[str | None], DiscoverySpaceResource],
    create_resources: Callable[[list[ADOResource], SQLStore], None],
) -> Callable[[str | None], DiscoverySpaceResource]:
    def _random_space_resource_from_db(
        sample_store_id: str | None = None,
    ) -> DiscoverySpaceResource:
        space = random_space_resource_from_file(sample_store_id=sample_store_id)
        create_resources(resources=[space])
        return space

    return _random_space_resource_from_db


@pytest.fixture(params=["discrete", "continuous"])
def constitutive_property_configuration_smiles_yaml(
    request: pytest.FixtureRequest,
) -> dict[str, str]:
    """User this fixture for tests requiring matching measurement space"""
    import yaml

    confs = """
          - identifier: "smiles"
          """

    return yaml.safe_load(confs)


valid_discovery_space_configs = [
    "examples/ml-multi-cloud/ml_multicloud_space.yaml",
    "examples/ml-multi-cloud/ml_multicloud_space_with_custom.yaml",
    "examples/pfas-generative-models/space_molgx_simple.yaml",
    "examples/pfas-generative-models/space_transformer_simple.yaml",
    "examples/pfas-generative-models/space_transformer_with_objective.yaml",
    "examples/optimization_test_functions/space.yaml",
]


@pytest.fixture(params=valid_discovery_space_configs)
def valid_discovery_space_config_file(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def discovery_space_configuration() -> DiscoverySpaceConfiguration:

    with open("examples/ml-multi-cloud/ml_multicloud_space.yaml") as f:
        d = yaml.safe_load(f)

    return DiscoverySpaceConfiguration.model_validate(d)


@pytest.fixture
def discovery_space_configuration_no_replay() -> DiscoverySpaceConfiguration:
    """Returns a discovery space that doesn't use replayed experiment"""

    with open("examples/optimization_test_functions/space.yaml") as f:
        d = yaml.safe_load(f)

    return DiscoverySpaceConfiguration.model_validate(d)


@pytest.fixture
def discovery_space_resource(
    discovery_space_configuration: DiscoverySpaceConfiguration,
) -> DiscoverySpaceResource:

    # This autogenerates the operation identifier
    return DiscoverySpaceResource(
        identifier="test_space",
        config=discovery_space_configuration,
    )


@pytest.fixture
def discovery_space_resource_no_replay(
    discovery_space_configuration_no_replay: DiscoverySpaceConfiguration,
) -> DiscoverySpaceResource:

    # This autogenerates the operation identifier
    return DiscoverySpaceResource(
        identifier="test_space",
        config=discovery_space_configuration_no_replay,
    )


@pytest.fixture
def test_space_identifier() -> str:
    """Fixture providing a space id that can be used for add/update/remove tests"""

    return "space-82aecb-b48040"


@pytest.fixture
def discovery_space_resource_from_file() -> DiscoverySpaceResource:
    """Returns a DiscoverySpace resource read from a file

    For use with inactive resource-stores where read is not possible"""

    with open("examples/ml-multi-cloud/ml_multicloud_space.yaml") as f:
        conf = DiscoverySpaceConfiguration(**yaml.safe_load(f))

    return DiscoverySpaceResource(config=conf)


@pytest.fixture
def make_continuous_space_configuration() -> Callable[
    [list[float], str], DiscoverySpaceConfiguration
]:
    """Factory: build a DiscoverySpaceConfiguration with a single continuous property over a given domainRange.

    Usage::

        def test_foo(make_continuous_space_configuration):
            config = make_continuous_space_configuration([0, 10])
    """

    def _make(
        domain_range: list[float],
        identifier: str = "x",
    ) -> DiscoverySpaceConfiguration:
        return DiscoverySpaceConfiguration(
            entitySpace=[
                ConstitutiveProperty(
                    identifier=identifier,
                    propertyDomain=PropertyDomain(domainRange=domain_range),
                )
            ]
        )

    return _make


@pytest.fixture
def make_discrete_space_configuration() -> Callable[
    [list[Any], str], DiscoverySpaceConfiguration
]:
    """Factory: build a DiscoverySpaceConfiguration with a single discrete property over a given set of values.

    Usage::

        def test_foo(make_discrete_space_configuration):
            config = make_discrete_space_configuration([1, 2, 3])
    """

    def _make(
        values: list[Any],
        identifier: str = "n",
    ) -> DiscoverySpaceConfiguration:
        return DiscoverySpaceConfiguration(
            entitySpace=[
                ConstitutiveProperty(
                    identifier=identifier,
                    propertyDomain=PropertyDomain(values=values),
                )
            ]
        )

    return _make


@pytest.fixture
def make_multi_prop_space_configuration() -> Callable[
    [list[tuple[str, PropertyDomain]]],
    DiscoverySpaceConfiguration,
]:
    """Factory: build a DiscoverySpaceConfiguration from an explicit list of (identifier, PropertyDomain) pairs.

    Usage::

        def test_foo(make_multi_prop_space_configuration):
            config = make_multi_prop_space_configuration([
                ("x", PropertyDomain(domainRange=[0, 10])),
                ("y", PropertyDomain(values=["A", "B"])),
            ])
    """

    def _make(
        properties: list[tuple[str, PropertyDomain]],
    ) -> DiscoverySpaceConfiguration:
        return DiscoverySpaceConfiguration(
            entitySpace=[
                ConstitutiveProperty(identifier=ident, propertyDomain=domain)
                for ident, domain in properties
            ]
        )

    return _make


@pytest.fixture
def space_configuration_none_entity_space() -> DiscoverySpaceConfiguration:
    """DiscoverySpaceConfiguration with no entity space defined."""
    return DiscoverySpaceConfiguration(entitySpace=None)
