"""Edit Decision List (EDL) — the cut-by-cut recipe for the final edit."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EDLEntry:
    """A single cut in the edit timeline.

    Attributes:
        source_path: Source video file.
        source_in: Start time in the source (seconds).
        source_out: End time in the source (seconds).
        timeline_in: Start time in the final timeline (seconds).
        duration: Length of this cut on the timeline (seconds).
        transition_in: Transition into this cut (``"cut"``, ``"crossfade"``,
            ``"fade_black"``, ``"match_cut"``, ``"speed_ramp"``).
        transition_out: Transition out of this cut.
        beat_aligned: True if this cut is snapped to a musical beat.
        shot_type: Optional shot-type tag (``"wide_establishing"``, ``"mid"``, ``"detail"``).
        movement: Optional camera-movement tag.
        score: Composite quality score of the source clip at this segment.
        notes: Free-form annotations (e.g. ``"peak"``, ``"intro"``).
    """

    source_path: Path
    source_in: float
    source_out: float
    timeline_in: float
    duration: float
    transition_in: str = "cut"
    transition_out: str = "cut"
    beat_aligned: bool = False
    shot_type: str | None = None
    movement: str | None = None
    score: float = 0.0
    notes: str = ""

    @property
    def timeline_out(self) -> float:
        """End timestamp on the final timeline (seconds)."""
        return self.timeline_in + self.duration


@dataclass(frozen=True, slots=True)
class EDL:
    """A full Edit Decision List — the ordered recipe for one render.

    Attributes:
        entries: Ordered list of cuts.
        target_duration: Intended total length (seconds).
        bpm: Music tempo used to align cuts (0 if no music).
        mood: Mood label applied to this edit.
        pacing: Pacing label (``"slow"``/``"medium"``/``"fast"``).
    """

    entries: list[EDLEntry] = field(default_factory=list)
    target_duration: float = 0.0
    bpm: float = 0.0
    mood: str = "neutral"
    pacing: str = "medium"

    @property
    def total_duration(self) -> float:
        """Sum of all cut durations (seconds)."""
        return sum(e.duration for e in self.entries)

    def summary(self) -> str:
        """Return a human-readable summary of the EDL."""
        lines = [
            f"EDL — {len(self.entries)} cuts  "
            f"{self.total_duration:.1f}s / {self.target_duration:.1f}s  "
            f"{self.bpm:.1f} BPM  mood={self.mood}  pacing={self.pacing}",
            f"{'#':>3}  {'t_in':>6}  {'dur':>5}  {'score':>5}  {'trans':<10}  {'notes':<15}  source",
            "  " + "─" * 78,
        ]
        for i, e in enumerate(self.entries, 1):
            lines.append(
                f"{i:>3}  {e.timeline_in:>6.2f}  {e.duration:>5.2f}  "
                f"{e.score:>5.1f}  {e.transition_in:<10}  "
                f"{e.notes:<15}  {e.source_path.name}"
            )
        return "\n".join(lines)
