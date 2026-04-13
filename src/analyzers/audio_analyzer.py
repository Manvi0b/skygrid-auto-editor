"""Analyzer that evaluates audio quality, volume, and silence in clips."""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np

from src.analyzers.base import BaseAnalyzer
from src.models.clip import Clip
from src.utils.ffmpeg_helpers import extract_audio_to_wav, has_audio_stream

logger = logging.getLogger(__name__)

# RMS below this threshold (in linear amplitude) is considered silence.
_SILENCE_THRESHOLD = 0.005
# Minimum silence duration in seconds to count as a segment.
_MIN_SILENCE_DURATION = 0.5


class AudioAnalyzer(BaseAnalyzer):
    """Scores clips by audio presence, volume levels, and silence ratio.

    Clips with clear, well-levelled audio score higher.  Clips with no
    audio track or that are entirely silent score 0.
    """

    @property
    def name(self) -> str:
        return "audio_analyzer"

    def analyze(self, clip: Clip) -> Clip:
        """Measure audio quality: presence, RMS, peak, and silence segments.

        Args:
            clip: Clip to analyze.

        Returns:
            Clip with an ``audio_analyzer`` score (0–100) and metadata for
            ``has_audio``, ``rms_db``, ``peak_db``, ``silence_ratio``, and
            ``silence_segments``.
        """
        # 1. Check for audio stream at all.
        if not has_audio_stream(clip.path):
            logger.info("%s — no audio stream", clip.path.name)
            result = clip.with_score(self.name, 0.0)
            result = result.with_metadata("has_audio", False)
            return result.with_tag("no_audio")

        # 2. Extract and load audio.
        wav_path: Path | None = None
        try:
            wav_path = extract_audio_to_wav(clip.path)
            y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
        except Exception:
            logger.exception("Audio extraction failed for %s", clip.path)
            result = clip.with_score(self.name, 50.0)
            return result.with_metadata("has_audio", True)
        finally:
            if wav_path and wav_path.exists():
                wav_path.unlink()

        if len(y) == 0:
            result = clip.with_score(self.name, 0.0)
            result = result.with_metadata("has_audio", True)
            return result.with_tag("silent")

        # 3. Compute RMS and peak in dB.
        rms_linear = float(np.sqrt(np.mean(y ** 2)))
        peak_linear = float(np.max(np.abs(y)))
        rms_db = 20.0 * np.log10(max(rms_linear, 1e-10))
        peak_db = 20.0 * np.log10(max(peak_linear, 1e-10))

        # 4. Detect silence segments via frame-level RMS.
        hop_length = 512
        frame_rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        frame_times = librosa.frames_to_time(
            np.arange(len(frame_rms)), sr=sr, hop_length=hop_length,
        )
        is_silent = frame_rms < _SILENCE_THRESHOLD

        silence_segments = _find_silence_segments(
            frame_times, is_silent, _MIN_SILENCE_DURATION,
        )
        total_silence = sum(end - start for start, end in silence_segments)
        audio_duration = float(len(y)) / sr
        silence_ratio = total_silence / max(audio_duration, 0.01)

        # 5. Score: combine volume level and silence ratio.
        #    Target RMS ≈ -20 dB (well-recorded outdoor audio).
        #    Heavily penalise near-silent clips and clips with >50 % silence.
        volume_score = _volume_score(rms_db)
        silence_penalty = max(0.0, silence_ratio - 0.3) * 100.0  # free up to 30 %
        score = max(0.0, min(100.0, volume_score - silence_penalty))
        score = round(score, 1)

        result = clip.with_score(self.name, score)
        result = result.with_metadata("has_audio", True)
        result = result.with_metadata("rms_db", round(float(rms_db), 1))
        result = result.with_metadata("peak_db", round(float(peak_db), 1))
        result = result.with_metadata("silence_ratio", round(silence_ratio, 3))
        result = result.with_metadata("silence_segments", silence_segments)

        if silence_ratio > 0.8:
            result = result.with_tag("silent")
        elif rms_db > -6:
            result = result.with_tag("loud")
        elif rms_db < -40:
            result = result.with_tag("quiet")

        logger.info(
            "%s — rms=%.1f dB  peak=%.1f dB  silence=%.0f%%  score=%s",
            clip.path.name, rms_db, peak_db, silence_ratio * 100, score,
        )
        return result


def _volume_score(rms_db: float) -> float:
    """Map RMS dB to a 0–100 score, peaking around -20 dB.

    Args:
        rms_db: RMS volume in decibels.

    Returns:
        Score between 0 and 100.
    """
    # Ideal range: -25 to -15 dB → 100.
    if -25.0 <= rms_db <= -15.0:
        return 100.0
    if rms_db < -60.0:
        return 0.0
    if rms_db < -25.0:
        # Linear ramp from 0 at -60 dB to 100 at -25 dB.
        return (rms_db + 60.0) / 35.0 * 100.0
    # Above -15 dB: penalise clipping risk.
    return max(0.0, 100.0 - (rms_db + 15.0) * 8.0)


def _find_silence_segments(
    frame_times: np.ndarray,
    is_silent: np.ndarray,
    min_duration: float,
) -> list[tuple[float, float]]:
    """Find contiguous silence segments above *min_duration*.

    Args:
        frame_times: Timestamp of each analysis frame.
        is_silent: Boolean mask — True where frame is silent.
        min_duration: Minimum segment length in seconds.

    Returns:
        List of ``(start_s, end_s)`` tuples.
    """
    segments: list[tuple[float, float]] = []
    in_silence = False
    start = 0.0

    for i, silent in enumerate(is_silent):
        t = float(frame_times[i])
        if silent and not in_silence:
            start = t
            in_silence = True
        elif not silent and in_silence:
            if t - start >= min_duration:
                segments.append((round(start, 3), round(t, 3)))
            in_silence = False

    # Handle trailing silence.
    if in_silence and len(frame_times) > 0:
        end = float(frame_times[-1])
        if end - start >= min_duration:
            segments.append((round(start, 3), round(end, 3)))

    return segments
