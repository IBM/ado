# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Utilities for working with Ray-decorated classes."""

import typing

T = typing.TypeVar("T")


def extract_base_class(
    obj: typing.Any,  # noqa: ANN401
    base_class: type[T],
) -> type[T]:
    """Extract the undecorated base class from a potentially Ray-decorated ActorClass.

    When a class is decorated with ``@ray.remote`` it becomes a Ray
    ``ActorClass`` instance rather than a plain Python ``type``.  This
    function accepts either form and always returns the underlying Python
    class, which is necessary before applying ``ray.remote`` dynamically
    (applying it a second time to an already-decorated class would fail).

    Args:
        obj: Either a Ray ``ActorClass`` instance or an undecorated subclass
            of ``base_class``.
        base_class: The expected base class that the extracted type must
            be a subclass of.

    Returns:
        The undecorated Python class that is a subclass of ``base_class``.

    Raises:
        ValueError: If ``obj`` is a Ray ``ActorClass`` but the original class
            cannot be extracted or is not a subclass of ``base_class``.
        TypeError: If ``obj`` is not a type and not a Ray ``ActorClass``, or
            is a type but not a subclass of ``base_class``.
    """
    # Fast path: already an undecorated subclass.
    if isinstance(obj, type) and issubclass(obj, base_class):
        return obj  # type: ignore[return-value]

    # Try to extract the original class from a Ray ActorClass.
    try:
        import ray.actor

        if isinstance(obj, ray.actor.ActorClass):
            original = getattr(obj, "__ray_actor_class__", None)
            if isinstance(original, type) and issubclass(original, base_class):
                return original  # type: ignore[return-value]
            raise ValueError(
                f"Could not extract {base_class.__name__} from Ray ActorClass {obj}: "
                "__ray_actor_class__ is missing or is not a subclass of "
                f"{base_class.__name__}."
            )
    except ImportError:
        pass

    if not isinstance(obj, type):
        raise TypeError(
            f"Expected a {base_class.__name__} subclass or a Ray ActorClass, "
            f"got instance of {type(obj).__name__}."
        )
    raise TypeError(f"Expected a subclass of {base_class.__name__}, got {obj!r}.")
