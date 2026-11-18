"""
Compatibility patches for Python 3.12+ and PyTorch 2.6+.

This module must be imported BEFORE any imports of ultralytics,
segment_anything, or label_studio_ml.
"""

from __future__ import annotations

import functools
import logging
import sys
from types import ModuleType

logger = logging.getLogger(__name__)


def apply_patches() -> None:
    """Apply all compatibility patches. Safe to call multiple times."""
    _patch_pkg_resources()
    _patch_torch_load()


def _patch_pkg_resources() -> None:
    """Mock pkg_resources for Python 3.12+ where setuptools may not be installed."""
    if "pkg_resources" in sys.modules:
        return

    m = ModuleType("pkg_resources")
    sys.modules["pkg_resources"] = m
    m.declare_namespace = lambda x: None  # type: ignore[attr-defined]
    m.get_distribution = lambda x: type("Dist", (), {"version": "0.0.0"})()  # type: ignore[attr-defined]

    try:
        from packaging.version import parse

        m.parse_version = parse  # type: ignore[attr-defined]
    except ImportError:
        m.parse_version = lambda v: v  # type: ignore[attr-defined]

    logger.debug("Applied pkg_resources mock for Python 3.12+ compatibility")


def _patch_torch_load() -> None:
    """
    Patch torch.load to use weights_only=False by default.

    WARNING: This allows arbitrary code execution during model loading.
    Only use with trusted model files.
    """
    import torch

    if hasattr(torch.load, "_patched"):
        return

    original = torch.load
    patched = functools.partial(original, weights_only=False)
    patched._patched = True  # type: ignore[attr-defined]
    torch.load = patched  # type: ignore[assignment]

    logger.debug("Applied torch.load patch (weights_only=False)")
