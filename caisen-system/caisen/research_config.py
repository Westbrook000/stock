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

    def __post_init__(self):
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        if self.minimum_neckline_tests is not None and self.minimum_neckline_tests < 1:
            raise ValueError("minimum_neckline_tests must be positive")
        if self.admission_types is not None:
            object.__setattr__(self, "admission_types", frozenset(self.admission_types))

    def cache_key(self):
        return (self.experiment_id, self.minimum_neckline_tests,
                tuple(sorted(self.admission_types)) if self.admission_types is not None else None)


_CURRENT_EXPERIMENT = ContextVar("caisen_recognition_experiment", default=None)


def current_recognition_experiment():
    """Return the immutable experiment config active in this execution context."""
    return _CURRENT_EXPERIMENT.get()


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
