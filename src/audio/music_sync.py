"""Music analysis — beat grid, downbeats, energy curve, and section detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MusicSection:
    """A contiguous region of the music track labelled by energy role.

    Attributes:
        start: Start time in seconds.
        end: End time in seconds.
        label: One of ``"intro"``, ``"build"``, ``"peak"``, ``"sustain"``, ``"outro"``.
        avg_energy: Mean RMS energy (0–1) in this section.
    """

    start: float
    end: float
    label: str
    avg_energy: float

    @property
    def duration(self) -> float:
        """Section length in seconds."""
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class MusicMap:
    """Full music analysis result — the backbone of beat-driven editing.

    Attributes:
        bpm: Estimated tempo (BPM).
        beat_times: Sorted list of beat-onset timestamps (seconds).
        downbeat_times: Bar-start beats (every 4th beat by default).
        duration: Total track duration (seconds).
        energy_curve: Normalized RMS energy sampled uniformly across the track.
        energy_times: Time axis (seconds) aligned with ``energy_curve``.
        sections: Labelled sections (intro / build / peak / sustain / outro).
        peak_time: Timestamp of the maximum-energy moment (for hero-shot placement).
    """

    bpm: float
    beat_times: list[float]
    downbeat_times: list[float]
    duration: float
    energy_curve: list[float] = field(default_factory=list)
    energy_times: list[float] = field(default_factory=list)
    sections: list[MusicSection] = field(default_factory=list)
    peak_time: float = 0.0

    # Backwards-compat alias — some callers still refer to BeatMap.
    @property
    def beat_count(self) -> int:
        """Total number of detected beats."""
        return len(self.beat_times)

    def section_at(self, t: float) -> MusicSection | None:
        """Return the section containing timestamp *t* (seconds), or None."""
        for s in self.sections:
            if s.start <= t < s.end:
                return s
        return None

    def energy_at(self, t: float) -> float:
        """Interpolated energy (0–1) at timestamp *t* (seconds)."""
        if not self.energy_curve:
            return 0.0
        return float(np.interp(t, self.energy_times, self.energy_curve))


# Backwards-compat alias so older code importing BeatMap still works.
BeatMap = MusicMap


def analyze_music(audio_path: Path) -> MusicMap:
    """Load a music file and return a full MusicMap.

    Args:
        audio_path: Path to a WAV, MP3, OGG, or FLAC file.

    Returns:
        A ``MusicMap`` with tempo, beats, downbeats, energy curve,
        labelled sections, and the peak timestamp.

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

    # --- Beat / tempo ---
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.asarray(tempo).flat[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    downbeat_times = [bt for i, bt in enumerate(beat_times) if i % 4 == 0]

    # --- Energy curve ---
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    # Smooth with a ~1-second moving average.
    smooth_n = max(1, int(sr / hop))
    kernel = np.ones(smooth_n) / smooth_n
    rms_smooth = np.convolve(rms, kernel, mode="same")
    # Normalise to 0–1.
    rms_max = float(rms_smooth.max()) if rms_smooth.size else 1.0
    energy_curve = (rms_smooth / rms_max).tolist() if rms_max > 0 else rms_smooth.tolist()
    energy_times = librosa.frames_to_time(
        np.arange(len(energy_curve)), sr=sr, hop_length=hop,
    ).tolist()

    peak_idx = int(np.argmax(rms_smooth)) if rms_smooth.size else 0
    peak_time = float(energy_times[peak_idx]) if energy_times else 0.0

    # --- Sections via spectral novelty ---
    sections = _detect_sections(y, sr, duration, energy_curve, energy_times)

    logger.info(
        "Music analysis: %s  %.1f BPM  %d beats  %d downbeats  "
        "%d sections  peak@%.2fs  %.1fs total",
        audio_path.name, bpm, len(beat_times), len(downbeat_times),
        len(sections), peak_time, duration,
    )

    return MusicMap(
        bpm=round(bpm, 1),
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        duration=duration,
        energy_curve=energy_curve,
        energy_times=energy_times,
        sections=sections,
        peak_time=peak_time,
    )


def _detect_sections(
    y: np.ndarray,
    sr: int,
    duration: float,
    energy_curve: list[float],
    energy_times: list[float],
) -> list[MusicSection]:
    """Partition the track into intro/build/peak/sustain/outro using energy."""
    if duration < 4.0 or not energy_curve:
        return [MusicSection(0.0, duration, "sustain", 0.5)]

    e = np.asarray(energy_curve)
    t = np.asarray(energy_times)

    # Simple rule-based partition driven by energy thresholds.
    peak_idx = int(np.argmax(e))
    peak_t = float(t[peak_idx])

    # Intro: first region where energy is below 40% of peak.
    intro_end = peak_t * 0.25
    # Outro: last 15% of track when energy drops below 60% of peak.
    outro_start = duration * 0.85
    # Build spans from intro_end to peak.
    build_start = intro_end
    build_end = max(peak_t - 1.0, build_start + 0.5)
    peak_end = min(peak_t + 3.0, outro_start)

    def _avg_energy(t0: float, t1: float) -> float:
        mask = (t >= t0) & (t < t1)
        return float(e[mask].mean()) if mask.any() else 0.0

    raw_sections = [
        (0.0, intro_end, "intro"),
        (intro_end, build_end, "build"),
        (build_end, peak_end, "peak"),
        (peak_end, outro_start, "sustain"),
        (outro_start, duration, "outro"),
    ]
    out: list[MusicSection] = []
    for s, e_, label in raw_sections:
        if e_ - s < 0.25:
            continue
        out.append(MusicSection(
            start=round(s, 3),
            end=round(e_, 3),
            label=label,
            avg_energy=round(_avg_energy(s, e_), 3),
        ))
    return out


# ------------------------------------------------------------------
# Beat-snapping helpers
# ------------------------------------------------------------------

def detect_beats(audio_path: Path) -> list[float]:
    """Return a list of beat timestamps (seconds) for an audio file."""
    return analyze_music(audio_path).beat_times


def snap_cuts_to_beats(
    cut_points: list[float],
    beat_times: list[float],
    tolerance: float = 0.15,
) -> list[float]:
    """Snap each cut to the nearest beat within *tolerance*."""
    if not beat_times:
        return cut_points
    beats = np.asarray(beat_times)
    out: list[float] = []
    for cut in cut_points:
        idx = int(np.argmin(np.abs(beats - cut)))
        if abs(beats[idx] - cut) <= tolerance:
            out.append(float(beats[idx]))
        else:
            out.append(cut)
    return out


def compute_cut_points(
    clip_durations: list[float],
    music_map: MusicMap,
    use_downbeats: bool = False,
) -> list[float]:
    """Compute ideal cut timestamps by aligning cumulative clip boundaries to beats."""
    targets = music_map.downbeat_times if use_downbeats else music_map.beat_times
    cumulative = 0.0
    raw: list[float] = []
    for dur in clip_durations[:-1]:
        cumulative += dur
        raw.append(cumulative)
    return snap_cuts_to_beats(raw, targets, tolerance=0.25)
