"""Lazy plugin registry for models and energies.

Optional models/energies register a *factory* keyed by name plus the extra that
provides them. ``get_model("umetrack")`` without ``parafit[umetrack]`` installed
raises a clear, actionable ImportError instead of a cryptic traceback. Third
parties can register their own via ``register_model`` / ``register_energy`` or
setuptools entry points (group ``parafit.models`` / ``parafit.energies``).
"""
from __future__ import annotations

from importlib import import_module
from typing import Callable

_MODELS: dict[str, tuple[Callable, str]] = {}
_ENERGIES: dict[str, tuple[Callable, str]] = {}


def register_model(name: str, extra: str = ""):
    def deco(fn: Callable):
        _MODELS[name] = (fn, extra)
        return fn

    return deco


def register_energy(name: str, extra: str = ""):
    def deco(fn: Callable):
        _ENERGIES[name] = (fn, extra)
        return fn

    return deco


def _resolve(table, kind, name, extra_import_targets):
    if name not in table:
        # Trigger lazy module imports so decorators run, then retry.
        for mod in extra_import_targets:
            try:
                import_module(mod)
            except ImportError:
                pass
    if name not in table:
        raise KeyError(f"unknown {kind} '{name}'; registered: {sorted(table)}")
    fn, extra = table[name]
    return fn, extra


def get_model(name: str, *args, **kwargs):
    fn, extra = _resolve(_MODELS, "model", name, [f"parafit.models.{name}"])
    try:
        return fn(*args, **kwargs)
    except ImportError as e:  # heavy dep missing
        hint = f" -- install it with: pip install 'parafit[{extra}]'" if extra else ""
        raise ImportError(f"model '{name}' needs an optional dependency{hint}") from e


def get_energy(name: str, *args, **kwargs):
    fn, extra = _resolve(_ENERGIES, "energy", name, [f"parafit.energies.{name}"])
    try:
        return fn(*args, **kwargs)
    except ImportError as e:
        hint = f" -- install it with: pip install 'parafit[{extra}]'" if extra else ""
        raise ImportError(f"energy '{name}' needs an optional dependency{hint}") from e
