"""
Pytest path shim for the API container.

The API container (Dockerfile.api) copies apps/api/ directly into /app/,
so the FastAPI app package lives at /app/app/... With PYTHONPATH=/app the
import root for the app code is `app.*`, not `apps.api.app.*`.

Test files that import `apps.api.app.*` (written to also work in the worker
container, where apps/api/ is at /app/apps/api/) fail in the API container.
This conftest bridges the gap by registering `apps.api.app.*` as aliases of
the already-loaded `app.*` modules.
"""
from __future__ import annotations

import importlib
import sys
import types


def _install_api_aliases() -> None:
    """Register apps.api.app.* → app.* aliases in sys.modules."""
    try:
        import app  # noqa: F401 — available in API container
    except ImportError:
        return  # not in API container — nothing to do

    src, dst = "app", "apps.api.app"

    # Ensure namespace stub modules exist up the chain
    for i, part in enumerate(dst.split(".")):
        ns = ".".join(dst.split(".")[:i + 1])
        if ns not in sys.modules:
            sys.modules[ns] = types.ModuleType(ns)

    # Pre-import submodules used by tests so the alias loop can find them
    for sub in (
        "app.services.confidence",
        "app.services.answer_service",
        "app.models",
        "app.routers",
        "app.schemas",
    ):
        try:
            importlib.import_module(sub)
        except Exception:
            pass

    # Walk all currently-loaded app.* modules and register aliases
    for key in list(sys.modules):
        if key == src or key.startswith(src + "."):
            alias = dst + key[len(src):]
            if alias not in sys.modules and sys.modules[key] is not None:
                sys.modules[alias] = sys.modules[key]


_install_api_aliases()
