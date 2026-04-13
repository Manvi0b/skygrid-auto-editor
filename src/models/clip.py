"""Clip dataclass representing a single video clip in the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Clip:
    """Immutable representation of a video clip and its analysis results.

    Attributes:
        path: Absolute path to the video file on disk.
        duration: Duration of the clip in seconds.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second of the source video.
        scores: Mapping of analyzer name to quality score (0–100).
        metadata: Arbitrary key-value metadata attached by analyzers.
        tags: Categorical labels assigned during analysis (e.g. "shaky", "loud").
    """

    path: Path
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    @property
    def resolution(self) -> str:
        """Return resolution as a ``"WxH"`` string."""
        return f"{self.width}x{self.height}"

    @property
    def composite_score(self) -> float:
        """Return the mean of all analyzer scores (0–100), or 0.0 if none."""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    def with_score(self, analyzer: str, score: float) -> Clip:
        """Return a new Clip with an additional or updated score.

        Args:
            analyzer: Name of the analyzer that produced the score.
            score: Quality score between 0 and 100.

        Returns:
            A new Clip instance with the updated scores dict.
        """
        new_scores = {**self.scores, analyzer: score}
        return Clip(
            path=self.path,
            duration=self.duration,
            width=self.width,
            height=self.height,
            fps=self.fps,
            scores=new_scores,
            metadata=self.metadata,
            tags=list(self.tags),
        )

    def with_metadata(self, key: str, value: Any) -> Clip:
        """Return a new Clip with an additional metadata entry.

        Args:
            key: Metadata key.
            value: Metadata value.

        Returns:
            A new Clip instance with the updated metadata dict.
        """
        new_metadata = {**self.metadata, key: value}
        return Clip(
            path=self.path,
            duration=self.duration,
            width=self.width,
            height=self.height,
            fps=self.fps,
            scores=self.scores,
            metadata=new_metadata,
            tags=list(self.tags),
        )

    def with_tag(self, tag: str) -> Clip:
        """Return a new Clip with an additional tag.

        Args:
            tag: Label to append (duplicates are ignored).

        Returns:
            A new Clip instance with the updated tags list.
        """
        if tag in self.tags:
            return self
        return Clip(
            path=self.path,
            duration=self.duration,
            width=self.width,
            height=self.height,
            fps=self.fps,
            scores=self.scores,
            metadata=self.metadata,
            tags=[*self.tags, tag],
        )
