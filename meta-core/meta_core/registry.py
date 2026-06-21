"""Plugin-style registries for encoders and modifiers.

Let third-party users plug custom encoders (e.g. `MyTransformerV2`) and custom modifiers
(e.g. `Reassembler`) into the framework without modifying core.

Usage:

    from meta_core import register_encoder, Encoder

    class MyEncoder(Encoder):
        ...

    register_encoder("my_encoder_v1", MyEncoder)

    # now available via config:
    config = MetaSpiderConfig(..., encoder_type="my_encoder_v1")

TODO: implement after finalizing the Encoder/Modifier ABC.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from meta_core.encoders.base import Encoder
    from meta_core.modifiers.base import Modifier

__all__ = [
    "register_encoder",
    "register_modifier",
    "get_encoder_class",
    "get_modifier_class",
    "list_encoders",
    "list_modifiers",
]


_ENCODER_REGISTRY: dict[str, Type["Encoder"]] = {}
_MODIFIER_REGISTRY: dict[str, Type["Modifier"]] = {}


def register_encoder(name: str, cls: Type["Encoder"]) -> None:
    """Register an encoder class under a name for use via config.

    Raises:
        ValueError: if the name is already taken (use `register_encoder(..., override=True)`
                    in the future if override is needed).
    """
    raise NotImplementedError


def register_modifier(name: str, cls: Type["Modifier"]) -> None:
    """Register a modifier class."""
    raise NotImplementedError


def get_encoder_class(name: str) -> Type["Encoder"]:
    """Get an encoder class by name.

    Raises:
        KeyError: if the name is not registered.
    """
    raise NotImplementedError


def get_modifier_class(name: str) -> Type["Modifier"]:
    """Get a modifier class by name."""
    raise NotImplementedError


def list_encoders() -> list[str]:
    """List the registered encoders."""
    return list(_ENCODER_REGISTRY.keys())


def list_modifiers() -> list[str]:
    """List the registered modifiers."""
    return list(_MODIFIER_REGISTRY.keys())
