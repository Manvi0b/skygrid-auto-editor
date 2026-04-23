"""Segment — a sub-range of a Clip with its own quality score.

A Clip is the whole file on disk.  A Segment is a continuous window
inside that file that survived per-window quality gating.  The
sequencer consumes segments, not clips, so it can pick the best
seconds of each source instead of whole clips.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Segment:
    """A quality-gated sub-range of a source clip.

    Attributes:
        source_path: Absolute path to the source video file.
        start: Start time inside the source (seconds).
        end: End time inside the source (seconds).
        quality_score: Mean quality score (0–100) over this window.
        clip_score: Composite score of the parent clip (tie-breaker).
        shot_type: Optional shot-type tag inherited from the parent.
        movement: Optional camera-movement tag inherited from the parent.
        orientation: Orientation of the source clip.
        source_profile: Source profile (e.g. ``"dji_mini3pro"``) of the parent.
        metrics: Per-window raw metrics (sharpness, exposure, stability, horizon).
    """

    source_path: Path
    start: float
    end: float
    quality_score: float
    clip_score: float = 0.0
    shot_type: str | None = None
    movement: str | None = None
    orientation: str = "horizontal"
    source_profile: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Segment length in seconds."""
        return self.end - self.start

    @property
    def composite_score(self) -> float:
        """Combined quality + parent-clip score (0–100)."""
        # 70 % window quality, 30 % parent clip — windows dominate.
        return 0.7 * self.quality_score + 0.3 * self.clip_score

    @property
    def metadata(self) -> dict[str, Any]:
        """Compatibility shim — sequencer reads ``metadata["shot_type"]`` etc."""
        return {
            "shot_type": self.shot_type,
            "movement": self.movement,
            "orientation": self.orientation,
            **{f"q_{k}": v for k, v in self.metrics.items()},
        }

    @property
    def path(self) -> Path:
        """Alias for ``source_path`` — matches the Clip interface."""
        return self.source_path
