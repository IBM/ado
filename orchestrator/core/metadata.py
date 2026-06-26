# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT


from typing import TYPE_CHECKING, Annotated

import pydantic
from pydantic import ConfigDict
from typing_extensions import Self

from orchestrator.utilities.pydantic import Pep440VersionStr

if TYPE_CHECKING:
    from orchestrator.core.resources import ADOResource


class ConfigurationMetadata(pydantic.BaseModel):

    model_config = ConfigDict(extra="allow")

    name: Annotated[
        str | None,
        pydantic.Field(
            description="A descriptive name for this configuration. Does not have to be unique"
        ),
    ] = None
    description: Annotated[
        str | None,
        pydantic.Field(
            description="One or more sentences describing this configuration. "
        ),
    ] = None
    labels: Annotated[
        dict[str, str] | None,
        pydantic.Field(
            description="Optional labels to allow for quick filtering of this resource"
        ),
    ] = None


class PackageProvenance(pydantic.BaseModel):
    """Records the Python distribution package that provided a plugin at resource creation time.

    Captures the PyPI distribution name and installed version so that the exact
    package used when a resource was created can be identified later for
    replication or debugging.

    Attributes:
        distributionName: The PyPI distribution name (e.g. ``"ado-ray-tune"``).
        distributionVersion: The installed version of the distribution (e.g. ``"1.7.1"``).
    """

    model_config = ConfigDict(frozen=True)

    distributionName: Annotated[
        str,
        pydantic.Field(
            description="PyPI distribution name (e.g. 'ado-ray-tune', 'ado-core')."
        ),
    ]
    distributionVersion: Annotated[
        Pep440VersionStr,
        pydantic.Field(
            description="Installed version of the distribution (e.g. '1.7.1')."
        ),
    ]

    @classmethod
    def from_distribution_name(cls, distribution_name: str) -> Self | None:
        """Look up installed package provenance for a PyPI distribution name.

        Args:
            distribution_name: The PyPI distribution name (e.g. ``"ado-core"``).

        Returns:
            Package provenance for the installed distribution, or ``None`` if the
            distribution is not installed or its version could not be resolved.
        """
        import importlib.metadata

        try:
            dist = importlib.metadata.distribution(distribution_name)
            version = dist.metadata.get("Version")
            if version is None:
                return None
            return cls(
                distributionName=distribution_name,
                distributionVersion=version,
            )
        except Exception:
            return None

    @classmethod
    def from_module_name(cls, module_name: str) -> Self | None:
        """Resolve installed package provenance from a fully qualified module name.

        Modules under the ``orchestrator`` namespace package are resolved to
        ``ado-core``. For all other modules, the containing distribution is
        resolved via :func:`~orchestrator.utilities.distribution.distribution_from_module`.

        Args:
            module_name: Fully qualified module name (e.g. ``"ado_ray_tune.operator"``).

        Returns:
            Package provenance for the installed distribution, or ``None`` if it
            could not be resolved.
        """
        from orchestrator.utilities.distribution import distribution_from_module

        if module_name.startswith("orchestrator.") or module_name == "orchestrator":
            return cls.from_distribution_name("ado-core")

        try:
            dist_name = distribution_from_module(module_name)
        except Exception:
            return None
        if dist_name is None:
            return None
        return cls.from_distribution_name(dist_name)

    @classmethod
    def from_module_conf(cls, module_conf: object) -> Self | None:
        """Resolve provenance from a module configuration object or dict.

        Accepts a :class:`~orchestrator.modules.module.ModuleConf` instance or a
        dict containing ``moduleName``.

        Args:
            module_conf: Module configuration carrying ``moduleName``.

        Returns:
            Package provenance for the installed distribution, or ``None`` if
            ``moduleName`` is missing or could not be resolved.
        """
        if isinstance(module_conf, dict):
            module_name = module_conf.get("moduleName")
        else:
            module_name = getattr(module_conf, "moduleName", None)
        if not isinstance(module_name, str):
            return None
        return cls.from_module_name(module_name)


class ProvenanceInfo(pydantic.BaseModel):
    """Base model for provenance stored on ADO resources.

    Records the ``ado-core`` distribution at resource creation time and, in
    subclasses, named maps of plugin identifiers to the distribution that
    provided each plugin. Plugin map fields must be ``dict[str, PackageProvenance]``.
    """

    model_config = ConfigDict(extra="forbid")

    ado: Annotated[
        PackageProvenance | None,
        pydantic.Field(
            default=None,
            description=(
                "ado-core distribution frozen at resource creation time. "
                "None for resources created before this field existed or when "
                "the installed version could not be resolved."
            ),
        ),
    ] = None

    @classmethod
    def resolve_ado_provenance(cls) -> PackageProvenance | None:
        """Return installed ``ado-core`` package provenance, if available."""
        return PackageProvenance.from_distribution_name("ado-core")

    @classmethod
    def at_creation(cls, **fields: object) -> Self:
        """Build provenance for a newly created resource.

        Resolves ``ado-core`` provenance when ``ado`` is not supplied.

        Args:
            **fields: Provenance field values for this model or subclass.

        Returns:
            A provenance instance with ``ado`` populated when available.
        """
        if fields.get("ado") is None:
            fields["ado"] = cls.resolve_ado_provenance()
        return cls(**fields)

    @pydantic.model_validator(mode="after")
    def validate_provenance_field_values(self) -> Self:
        """Ensure plugin map fields are dicts of PackageProvenance instances."""
        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            if not isinstance(value, dict):
                continue
            for key, item in value.items():
                if not isinstance(item, PackageProvenance):
                    raise ValueError(
                        f"{field_name}[{key!r}] must be PackageProvenance, "
                        f"got {type(item)}"
                    )
        return self


def with_ado_core_provenance(resource: "ADOResource") -> "ADOResource":
    """Return a resource with ``ado-core`` provenance set when missing.

    Args:
        resource: Resource being prepared for persistence.

    Returns:
        The same resource, or a copy whose provenance was rebuilt via
        :meth:`ProvenanceInfo.at_creation` when ``provenance.ado`` was ``None``.
    """
    if resource.provenance.ado is not None:
        return resource

    provenance_type = type(resource.provenance)
    provenance_fields = resource.provenance.model_dump(exclude={"ado"})
    return resource.model_copy(
        update={"provenance": provenance_type.at_creation(**provenance_fields)}
    )
