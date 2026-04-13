"""Beat detection, BPM estimation, and tempo-matching utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BeatMap:
    """Holds beat-analysis results for a music track.

    Attributes:
        bpm: Estimated tempo in beats per minute.
        beat_times: Sorted list of beat-onset timestamps in seconds.
        downbeat_times: Subset of beat_times on the downbeat (every 4th beat).
        duration: Total duration of the audio in seconds.
    """

    bpm: float
    beat_times: list[float]
    downbeat_times: list[float]
    duration: float


def analyze_music(audio_path: Path) -> BeatMap:
    """Load a music file and return a full BeatMap.

    Args:
        audio_path: Path to a WAV, MP3, OGG, or FLAC file.

    Returns:
        A ``BeatMap`` with tempo, beat timestamps, and downbeats.

    Raises:
        FileNotFoundError: If *audio_path* does not exist.
        RuntimeError: If librosa fails to load or analyse the file.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Music file not found: {audio_path}")

    try:
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to load {audio_path}: {exc}") from exc

    duration = float(len(y)) / sr

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.asarray(tempo).flat[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    # Every 4th beat as a downbeat (approximation — works for 4/4 time).
    downbeat_times = [bt for i, bt in enumerate(beat_times) if i % 4 == 0]

    logger.info(
        "Music analysis: %s  %.1f BPM  %d beats  %d downbeats  %.1fs",
        audio_path.name, bpm, len(beat_times), len(downbeat_times), duration,
    )
    return BeatMap(
        bpm=round(bpm, 1),
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        duration=duration,
    )


def detect_beats(audio_path: Path) -> list[float]:
    """Return a list of beat timestamps (in seconds) for an audio file.

    Convenience wrapper around ``analyze_music`` for callers that only
    need the beat list.

    Args:
        audio_path: Path to a WAV or MP3 file.

    Returns:
        Sorted list of beat onset times in seconds.
    """
    return analyze_music(audio_path).beat_times


def snap_cuts_to_beats(
    cut_points: list[float],
    beat_times: list[float],
    tolerance: float = 0.15,
) -> list[float]:
    """Snap each cut point to the nearest beat within *tolerance*.

    Args:
        cut_points: Desired cut timestamps in seconds.
        beat_times: Beat timestamps from ``detect_beats``.
        tolerance: Maximum distance (seconds) to snap a cut.

    Returns:
        Adjusted cut points, each snapped to its nearest beat when
        the distance is within *tolerance*.
    """
    if not beat_times:
        return cut_points

    beats = np.asarray(beat_times)
    snapped: list[float] = []
    for cut in cut_points:
        idx = int(np.argmin(np.abs(beats - cut)))
        if abs(beats[idx] - cut) <= tolerance:
            snapped.append(float(beats[idx]))
        else:
            snapped.append(cut)
    return snapped


def compute_cut_points(
    clip_durations: list[float],
    beat_map: BeatMap,
    use_downbeats: bool = False,
) -> list[float]:
    """Compute ideal cut timestamps by aligning cumulative clip boundaries to beats.

    Args:
        clip_durations: Durations of each clip in playback order.
        beat_map: Beat analysis of the music track.
        use_downbeats: If True, snap to downbeats only (fewer, stronger cuts).

    Returns:
        List of cut timestamps (in seconds) for each clip boundary.
    """
    targets = beat_map.downbeat_times if use_downbeats else beat_map.beat_times
    cumulative = 0.0
    raw_cuts: list[float] = []
    for dur in clip_durations[:-1]:  # no cut after the last clip
        cumulative += dur
        raw_cuts.append(cumulative)
    return snap_cuts_to_beats(raw_cuts, targets, tolerance=0.25)
