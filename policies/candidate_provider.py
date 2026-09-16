"""Frozen-policy candidate interface used by causal RLBench collection."""

from __future__ import annotations

import importlib
import importlib.util
from functools import partial
from pathlib import Path
from typing import Callable, Protocol
from urllib.parse import parse_qsl

import numpy as np


class CandidateProvider(Protocol):
    def __call__(
        self,
        observation_history: dict[str, np.ndarray],
        candidate_count: int,
        action_horizon: int,
        seed: int,
    ) -> np.ndarray:
        """Return absolute joint-position actions [M,L,8]."""


def load_candidate_provider(specification: str) -> Callable:
    """Load a policy callback, optionally binding URL-style keyword arguments.

    Examples:
      policies.frozen_bc:provide?checkpoint=/models/bc.pt&device=cuda
      /path/provider.py:sample
    """
    if ":" not in specification:
        raise ValueError("candidate provider must be module:function or file.py:function")
    source, target = specification.rsplit(":", 1)
    function_name, separator, query = target.partition("?")
    path = Path(source).expanduser()
    if path.suffix == ".py" or path.is_file():
        path = path.resolve()
        module_spec = importlib.util.spec_from_file_location("icra2027_candidate_provider", path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"cannot load policy provider from {path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(source)
    provider = getattr(module, function_name)
    if not callable(provider):
        raise TypeError(f"{specification} is not callable")
    if not separator:
        return provider
    # Semicolons are accepted as a shell-safe alternative to ``&`` when the
    # provider is passed through a command line on Windows or a job launcher.
    arguments = dict(parse_qsl(query.replace(";", "&"), keep_blank_values=False))
    if not arguments:
        raise ValueError("candidate provider query did not contain any keyword arguments")
    return partial(provider, **arguments)


def validate_policy_candidates(actions: np.ndarray, candidate_count: int,
                               action_horizon: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    expected = (candidate_count, action_horizon, 8)
    if actions.shape != expected:
        raise ValueError(f"policy provider returned {actions.shape}, expected {expected}")
    if not np.isfinite(actions).all():
        raise ValueError("policy candidates contain non-finite actions")
    actions[..., 7] = (actions[..., 7] >= 0.5).astype(np.float32)
    return actions
