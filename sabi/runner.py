"""Runner - convenience entry point for building and starting the runtime.

This mirrors the structure described in the project spec. ``Runtime`` lives in
``sabi.runtime``; this module exposes simple helpers used by the CLI and by
external embedders.
"""

from __future__ import annotations

from typing import Optional

from .config import Config, load_config
from .runtime import Runtime


def build_runtime(config: Optional[Config] = None) -> Runtime:
    """Construct and fully initialise a runtime."""
    return Runtime(config or load_config()).start()


def run_once(request: str, config: Optional[Config] = None) -> dict:
    """Handle a single request and return the result dict (for scripting)."""
    return build_runtime(config).handle(request)
