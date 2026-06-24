"""SABI - The Offline AI Coworker.

An on-device AI coworker built for constrained African hardware. SABI turns
ideas into working software and structured business output locally, with no
cloud dependency, under a strict 7 GB memory ceiling (ADTC 2026).
"""

__version__ = "1.0.0"
__author__ = "Godspower Uyanga"
__license__ = "MIT"

from .config import Config, load_config  # noqa: E402,F401

__all__ = ["Config", "load_config", "__version__", "__author__", "__license__"]
