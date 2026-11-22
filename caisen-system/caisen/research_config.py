"""Process-local settings for explicitly opt-in recognition experiments.

The production recognition rules remain the default. Research runners install
one immutable configuration around a fresh causal replay, then the recognizer
can read that configuration while creating candidates and events.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import FrozenSet, Optional


@dataclass(frozen=True)
class RecognitionExperimentConfig:
    experiment_id: str
    minimum_neckline_tests: Optional[int] = None
    admission_types: Optional[FrozenSet[str]] = None
    neckline_selection_order: str = "test_count_first"
    require_internal_rebound_coverage: bool = False
    require_minimum_anchor_span: bool = False

    def __post_init__(self):
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        if self.minimum_neckline_tests is not None and self.minimum_neckline_tests < 1:
            raise ValueError("minimum_neckline_tests must be positive")
        if self.admission_types is not None:
            object.__setattr__(self, "admission_types", frozenset(self.admission_types))
        if self.neckline_selection_order not in {"test_count_first", "span_first"}:
            raise ValueError("unsupported neckline_selection_order")
        if not isinstance(self.require_internal_rebound_coverage, bool):
            raise TypeError("require_internal_rebound_coverage must be bool")
        if not isinstance(self.require_minimum_anchor_span, bool):
            raise TypeError("require_minimum_anchor_span must be bool")

    def cache_key(self):
        return (self.experiment_id, self.minimum_neckline_tests, self.neckline_selection_order,
                tuple(sorted(self.admission_types)) if self.admission_types is not None else None,
                self.require_internal_rebound_coverage, self.require_minimum_anchor_span)


_CURRENT_EXPERIMENT = ContextVar("caisen_recognition_experiment", default=None)


def current_recognition_experiment():
    """Return the immutable experiment config active in this execution context."""
    return _CURRENT_EXPERIMENT.get()


def recognition_context_snapshot():
    """Return a stable JSON-safe description of the active recognition config."""
    config = current_recognition_experiment()
    if config is None:
        return {"recognition_experiment": None}
    return {"recognition_experiment": {
        "experiment_id": config.experiment_id,
        "minimum_neckline_tests": config.minimum_neckline_tests,
        "admission_types": sorted(config.admission_types) if config.admission_types is not None else None,
        "neckline_selection_order": config.neckline_selection_order,
        "require_internal_rebound_coverage": config.require_internal_rebound_coverage,
        "require_minimum_anchor_span": config.require_minimum_anchor_span,
        "cache_key": list(config.cache_key()),
    }}


@contextmanager
def recognition_experiment(config):
    """Temporarily install a research config; always restore the prior value."""
    if config is not None and not isinstance(config, RecognitionExperimentConfig):
        raise TypeError("config must be RecognitionExperimentConfig or None")
    token = _CURRENT_EXPERIMENT.set(config)
    try:
        yield config
    finally:
        _CURRENT_EXPERIMENT.reset(token)
