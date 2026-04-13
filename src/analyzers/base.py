"""Abstract base class for all clip analyzers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.clip import Clip


class BaseAnalyzer(ABC):
    """Base class that every analyzer — built-in or plugin — must inherit from.

    Subclasses implement ``analyze`` to inspect a clip and return an enriched
    copy with scores, metadata, or tags added.  The ``name`` property is used
    as the key in ``Clip.scores`` and for config-driven enable/disable.

    Example:
        >>> class LoudnessAnalyzer(BaseAnalyzer):
        ...     name = "loudness"
        ...     def analyze(self, clip: Clip) -> Clip:
        ...         loudness = measure_loudness(clip.path)
        ...         return clip.with_score(self.name, loudness)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this analyzer (must match config keys)."""

    @abstractmethod
    def analyze(self, clip: Clip) -> Clip:
        """Analyze a clip and return an enriched copy.

        Args:
            clip: The clip to analyze.

        Returns:
            A new Clip instance with scores, metadata, or tags added by
            this analyzer.  The original clip must not be mutated.
        """
