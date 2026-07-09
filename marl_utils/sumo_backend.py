#!/usr/bin/env python3
"""
Backend selector for the SUMO python API.

Selected once per process via the SUMO_MARL_BACKEND env var:
  - "libsumo" (default): in-process simulation, 5-10x faster, no GUI support.
  - "traci":  socket-based, required for sumo-gui visual debugging.

With the libsumo backend, SUMO_HOME is pointed at the pip `eclipse-sumo`
package (if installed) so that netconvert and the simulation core come from
the same SUMO version. With the traci backend, an existing SUMO_HOME is
respected.
"""
from __future__ import annotations
import os

_BACKEND = None  # (module, name)


def get_backend():
    """Return (sumo_api_module, backend_name); import happens on first call."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    name = os.environ.get("SUMO_MARL_BACKEND", "libsumo").strip().lower()
    if name not in ("libsumo", "traci"):
        raise ValueError(f"SUMO_MARL_BACKEND must be 'libsumo' or 'traci', got '{name}'")

    if name == "traci":
        import traci as mod  # type: ignore
    else:
        import libsumo as mod  # type: ignore
        _point_sumo_home_at_pip_package()

    _BACKEND = (mod, name)
    return _BACKEND


def _point_sumo_home_at_pip_package():
    try:
        import sumo  # pip package "eclipse-sumo": bundles version-matched binaries
    except ImportError:
        return  # fall back to whatever SUMO_HOME the system provides
    pip_home = os.path.dirname(os.path.abspath(sumo.__file__))
    if os.environ.get("SUMO_HOME") != pip_home:
        os.environ["SUMO_HOME"] = pip_home
