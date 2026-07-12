"""Movement registry: look up a :class:`MovementConfig` by key.

The registry is an explicit dict populated at import time from the movement
modules in this package. Registration order is the order of ``_MODULES``
below. Adding a movement is exactly two edits: one new ``<movement>.py``
module exposing a module-level ``CONFIG``, and one line adding that module to
``_MODULES``. Nothing is discovered by scanning the filesystem -- the wiring
is deliberately explicit so the set of supported movements is greppable.
"""

from __future__ import annotations

from . import (
    back_squat,
    deadlift,
    hang_power_clean,
    power_clean,
    power_snatch,
    push_press,
)
from .base import (
    BAR_TRAVELS,
    FAMILIES,
    KNOWN_DETECTORS,
    STARTS_FROM,
    MadeCriteria,
    MovementConfig,
    PhaseDef,
)

# One entry per supported movement module. Add a new movement here.
_MODULES = (
    power_clean,
    power_snatch,
    back_squat,
    push_press,
    deadlift,
    hang_power_clean,
)


class UnknownMovementError(LookupError):
    """Raised by :func:`get` when no movement is registered under a key."""


def _build_registry() -> dict[str, MovementConfig]:
    registry: dict[str, MovementConfig] = {}
    for module in _MODULES:
        config = module.CONFIG
        if config.key in registry:
            raise ValueError(f"duplicate movement key {config.key!r} in registry")
        registry[config.key] = config
    return registry


_REGISTRY: dict[str, MovementConfig] = _build_registry()


def get(key: str) -> MovementConfig:
    """Return the config registered under ``key``.

    Raises :class:`UnknownMovementError`, whose message lists the available
    keys, if nothing is registered under ``key``.
    """
    try:
        return _REGISTRY[key]
    except KeyError:
        raise UnknownMovementError(f"unknown movement {key!r}; available: {all_keys()}") from None


def all_keys() -> list[str]:
    """Registered movement keys, in registration order."""
    return list(_REGISTRY.keys())


def all_configs() -> list[MovementConfig]:
    """All registered configs, in registration order."""
    return list(_REGISTRY.values())


__all__ = [
    "BAR_TRAVELS",
    "FAMILIES",
    "KNOWN_DETECTORS",
    "STARTS_FROM",
    "MadeCriteria",
    "MovementConfig",
    "PhaseDef",
    "UnknownMovementError",
    "all_configs",
    "all_keys",
    "get",
]
