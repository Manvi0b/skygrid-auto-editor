"""Assembles scored clips into a timeline with transitions and dead-air trimming."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from moviepy import (
    VideoFileClip,
    concatenate_videoclips,
)
from moviepy.video.fx import CrossFadeIn, CrossFadeOut, FadeIn, FadeOut
from moviepy.audio.fx import AudioFadeIn, AudioFadeOut

from src.config import Config
from src.models.clip import Clip

logger = logging.getLogger(__name__)

# Minimum RMS (linear) to consider a frame as "non-silent".
_SILENCE_FLOOR = 0.01


def assemble(clips: list[Clip], config: Config) -> Path:
    """Stitch *clips* into a single video with transitions and trimming.

    Steps for each clip:
        1. Load with moviepy.
        2. Trim leading/trailing dead air (silence) if audio is present.
        3. Apply fade-in / fade-out based on the configured transition style.

    All processed clips are concatenated and written to an intermediate file.

    Args:
        clips: Ordered list of clips to concatenate.
        config: Pipeline configuration.

    Returns:
        Path to the assembled (intermediate) video file.

    Raises:
        ValueError: If *clips* is empty.
    """
    if not clips:
        raise ValueError("No clips to assemble")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / "assembled.mp4"

    video_clips: list[VideoFileClip] = []
    total_duration = 0.0
    xfade = config.transition_duration

    try:
        for clip in clips:
            if total_duration >= config.max_duration:
                break

            vc = VideoFileClip(str(clip.path))

            # Trim dead air from start and end.
            vc = _trim_dead_air(vc)

            # Enforce max duration budget.
            remaining = config.max_duration - total_duration
            if vc.duration > remaining:
                vc = vc.subclipped(0, remaining)

            if vc.duration < 0.5:
                logger.debug("Skipping very short clip %s (%.2fs)", clip.path.name, vc.duration)
                vc.close()
                continue

            # Apply transitions.
            vc = _apply_transition(vc, config.transition_style, xfade)
            video_clips.append(vc)
            total_duration += vc.duration

        if not video_clips:
            raise ValueError("No usable video clips after trimming")

        method = "compose" if config.transition_style != "cut" else "chain"
        padding = -xfade if config.transition_style == "crossfade" and len(video_clips) > 1 else 0
        final = concatenate_videoclips(video_clips, method=method, padding=padding)

        final.write_videofile(
            str(output_path),
            fps=config.fps,
            codec=config.codec,
            audio_codec=config.audio_codec,
            bitrate=config.bitrate,
            preset=config.preset,
            logger=None,
        )
    finally:
        for vc in video_clips:
            vc.close()

    logger.info(
        "Assembled %d clips → %s (%.1fs)",
        len(video_clips), output_path, total_duration,
    )
    return output_path


def _trim_dead_air(vc: VideoFileClip, margin: float = 0.1) -> VideoFileClip:
    """Trim leading and trailing silence from a video clip.

    Args:
        vc: The moviepy VideoFileClip to trim.
        margin: Extra padding in seconds to keep around the content boundary.

    Returns:
        A (possibly sub-clipped) VideoFileClip with dead air removed.
    """
    if vc.audio is None:
        return vc

    try:
        sr = 22050
        audio_array = vc.audio.to_soundarray(fps=sr)
        if audio_array.ndim > 1:
            mono = audio_array.mean(axis=1)
        else:
            mono = audio_array

        abs_signal = np.abs(mono)

        # Find first and last sample above the silence floor.
        above = np.where(abs_signal > _SILENCE_FLOOR)[0]
        if len(above) == 0:
            return vc  # entirely silent — don't trim

        first_sample = above[0]
        last_sample = above[-1]
        start_s = max(0.0, first_sample / sr - margin)
        end_s = min(vc.duration, last_sample / sr + margin)

        if end_s - start_s < 0.5:
            return vc  # too short after trim — keep original

        if start_s > 0.2 or (vc.duration - end_s) > 0.2:
            logger.debug(
                "Trimming dead air: %.2fs–%.2fs → %.2fs–%.2fs",
                0.0, vc.duration, start_s, end_s,
            )
            return vc.subclipped(start_s, end_s)

    except Exception:
        logger.debug("Dead-air trimming failed — keeping original clip", exc_info=True)

    return vc


def _apply_transition(
    vc: VideoFileClip,
    style: str,
    duration: float,
) -> VideoFileClip:
    """Apply fade-in/fade-out effects to a clip based on the transition style.

    Args:
        vc: The video clip.
        style: One of ``"cut"``, ``"crossfade"``, ``"fade_black"``.
        duration: Transition duration in seconds.

    Returns:
        The clip with transition effects applied.
    """
    if style == "cut" or duration <= 0 or vc.duration <= duration * 2:
        return vc

    effects: list = []
    audio_effects: list = []

    match style:
        case "crossfade":
            effects = [CrossFadeIn(duration), CrossFadeOut(duration)]
            audio_effects = [AudioFadeIn(duration), AudioFadeOut(duration)]
        case "fade_black":
            effects = [FadeIn(duration), FadeOut(duration)]
            audio_effects = [AudioFadeIn(duration), AudioFadeOut(duration)]
        case _:
            return vc

    vc = vc.with_effects(effects)
    if vc.audio is not None:
        vc = vc.with_audio(vc.audio.with_effects(audio_effects))

    return vc
