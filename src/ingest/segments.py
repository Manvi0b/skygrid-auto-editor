"""Segment extraction — splits clips into quality-gated ≥1.5s windows.

Reads per-window scores produced by ``quality_windows`` and emits a flat
list of :class:`~src.models.segment.Segment` instances, one per
contiguous run of windows whose score meets the minimum threshold.

Writes a ``segments.json`` manifest alongside ``project.json`` so the
segment pool can be inspected or re-used without re-running analysis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.models.clip import Clip
from src.models.segment import Segment

logger = logging.getLogger(__name__)

_MIN_SEGMENT_DURATION = 1.5
_DEFAULT_MIN_WINDOW_SCORE = 50.0


def extract_segments(
    clips: list[Clip],
    min_window_score: float = _DEFAULT_MIN_WINDOW_SCORE,
    min_duration: float = _MIN_SEGMENT_DURATION,
) -> list[Segment]:
    """Return a flat list of usable segments across every clip.

    A segment is a contiguous run of quality windows whose score is at
    least *min_window_score*, long enough to meet *min_duration*.

    Args:
        clips: Analyzed clips (should have ``metadata["windows"]``).
        min_window_score: Gate for per-window quality score (0–100).
        min_duration: Minimum segment length in seconds.

    Returns:
        Flat list of Segment instances, ordered by descending composite score.
    """
    all_segments: list[Segment] = []
    for clip in clips:
        windows = clip.metadata.get("windows") if clip.metadata else None
        if not windows:
            # Fallback: treat the whole clip as one segment if it's long enough.
            if clip.duration >= min_duration:
                all_segments.append(_whole_clip_segment(clip))
            continue

        runs = _group_contiguous(windows, min_window_score)
        for run in runs:
            start = float(run[0]["t"])
            end = float(run[-1].get("end", run[-1]["t"] + 1.0))
            if end - start < min_duration:
                continue
            mean_score = sum(float(w["score"]) for w in run) / len(run)
            metrics = _aggregate_metrics(run)
            all_segments.append(Segment(
                source_path=clip.path,
                start=round(start, 3),
                end=round(end, 3),
                quality_score=round(mean_score, 2),
                clip_score=clip.composite_score,
                shot_type=clip.metadata.get("shot_type") if clip.metadata else None,
                movement=clip.metadata.get("movement") if clip.metadata else None,
                orientation=clip.orientation,
                source_profile=clip.source_profile,
                metrics=metrics,
            ))

    all_segments.sort(key=lambda s: s.composite_score, reverse=True)
    logger.info(
        "Extracted %d segments from %d clips (min_score=%.1f, min_dur=%.1fs)",
        len(all_segments), len(clips), min_window_score, min_duration,
    )
    return all_segments


def write_segments_manifest(
    segments: list[Segment],
    output_dir: Path,
) -> Path:
    """Serialise *segments* to ``<output_dir>/segments.json``.

    Args:
        segments: Segment list to persist.
        output_dir: Destination directory (created if needed).

    Returns:
        Path to the written manifest.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "segments.json"
    payload = {
        "segment_count": len(segments),
        "segments": [
            {
                "source_path": str(s.source_path),
                "start": s.start,
                "end": s.end,
                "duration": round(s.duration, 3),
                "quality_score": s.quality_score,
                "clip_score": round(s.clip_score, 2),
                "composite_score": round(s.composite_score, 2),
                "shot_type": s.shot_type,
                "movement": s.movement,
                "orientation": s.orientation,
                "metrics": s.metrics,
            }
            for s in segments
        ],
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Segments manifest → %s (%d segments)", path, len(segments))
    return path


# ------------------------------------------------------------------
# Internals
# ------------------------------------------------------------------

def _group_contiguous(
    windows: list[dict],
    min_score: float,
) -> list[list[dict]]:
    """Group consecutive windows whose score ≥ *min_score*."""
    runs: list[list[dict]] = []
    current: list[dict] = []
    for w in windows:
        if float(w.get("score", 0.0)) >= min_score:
            current.append(w)
        else:
            if current:
                runs.append(current)
                current = []
    if current:
        runs.append(current)
    return runs


def _aggregate_metrics(run: list[dict]) -> dict[str, float]:
    """Compute mean metric values across a window run."""
    keys = ("sharpness", "exposure", "stability", "horizon")
    out: dict[str, float] = {}
    for k in keys:
        vals = [float(w[k]) for w in run if k in w]
        if vals:
            out[k] = round(sum(vals) / len(vals), 2)
    return out


def _whole_clip_segment(clip: Clip) -> Segment:
    """Fallback segment covering the entire clip (no window data)."""
    return Segment(
        source_path=clip.path,
        start=0.0,
        end=clip.duration,
        quality_score=clip.composite_score,
        clip_score=clip.composite_score,
        shot_type=clip.metadata.get("shot_type") if clip.metadata else None,
        movement=clip.metadata.get("movement") if clip.metadata else None,
        orientation=clip.orientation,
        source_profile=clip.source_profile,
        metrics={},
    )
