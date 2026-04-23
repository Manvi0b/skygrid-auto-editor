"""Assembles scored clips into a timeline with transitions, dead-air trimming,
and aspect-ratio adaptation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from moviepy import (
    VideoFileClip,
    concatenate_videoclips,
)
from moviepy.video.fx import CrossFadeIn, CrossFadeOut, FadeIn, FadeOut, Resize
from moviepy.audio.fx import AudioFadeIn, AudioFadeOut

from src.config import Config
from src.models.clip import Clip
from src.models.output_profile import OutputProfile

logger = logging.getLogger(__name__)

_SILENCE_FLOOR = 0.01


def assemble(
    clips: list[Clip],
    config: Config,
    profile: OutputProfile | None = None,
) -> Path:
    """Stitch *clips* into a single video with transitions and trimming.

    Steps for each clip:
        1. Load with moviepy.
        2. Adapt aspect ratio to match the target output profile.
        3. Trim leading/trailing dead air (silence) if audio is present.
        4. Apply fade-in / fade-out based on the configured transition style.

    Args:
        clips: Ordered list of clips to concatenate.
        config: Pipeline configuration.
        profile: Output profile override.  Falls back to ``config.output_profile``.

    Returns:
        Path to the assembled (intermediate) video file.

    Raises:
        ValueError: If *clips* is empty.
    """
    if not clips:
        raise ValueError("No clips to assemble")

    target = profile or config.output_profile
    ar_mode = config.aspect_ratio_mode

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

            # Aspect-ratio adaptation.
            vc = _adapt_aspect_ratio(vc, target, ar_mode)

            # Trim dead air.
            vc = _trim_dead_air(vc)

            # Duration budget.
            remaining = config.max_duration - total_duration
            if vc.duration > remaining:
                vc = vc.subclipped(0, remaining)

            if vc.duration < 0.5:
                logger.debug("Skipping very short clip %s (%.2fs)", clip.path.name, vc.duration)
                vc.close()
                continue

            # Transitions.
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
            fps=target.fps,
            codec=target.codec,
            audio_codec=target.audio_codec,
            bitrate=target.bitrate,
            preset=target.preset,
            logger=None,
        )
    finally:
        for vc in video_clips:
            vc.close()

    logger.info(
        "Assembled %d clips → %s (%.1fs) [profile=%s]",
        len(video_clips), output_path, total_duration, target.name,
    )
    return output_path


# ------------------------------------------------------------------
# Aspect-ratio adaptation
# ------------------------------------------------------------------

def _adapt_aspect_ratio(
    vc: VideoFileClip,
    target: OutputProfile,
    mode: str,
) -> VideoFileClip:
    """Adapt a clip's aspect ratio to match the target output profile.

    Args:
        vc: Source video clip.
        target: Desired output profile.
        mode: Adaptation strategy — ``"crop"``, ``"blur_bars"``, or
            ``"letterbox"``.

    Returns:
        A new VideoFileClip conforming to the target dimensions.
    """
    src_w, src_h = vc.w, vc.h
    tgt_w, tgt_h = target.width, target.height

    src_ar = src_w / max(src_h, 1)
    tgt_ar = tgt_w / max(tgt_h, 1)

    # Already close enough (within 2 %).
    if abs(src_ar - tgt_ar) / max(tgt_ar, 0.01) < 0.02:
        return vc.with_effects([Resize((tgt_w, tgt_h))])

    match mode:
        case "crop":
            return _center_crop(vc, tgt_w, tgt_h)
        case "blur_bars":
            return _blur_bars(vc, tgt_w, tgt_h)
        case "letterbox":
            return _letterbox(vc, tgt_w, tgt_h)
        case "smart_crop":
            from src.reframe.smart import smart_reframe
            return smart_reframe(vc, tgt_w, tgt_h)
        case _:
            logger.warning("Unknown aspect_ratio_mode '%s' — using letterbox", mode)
            return _letterbox(vc, tgt_w, tgt_h)


def _center_crop(vc: VideoFileClip, tgt_w: int, tgt_h: int) -> VideoFileClip:
    """Crop from the centre to fill the target dimensions exactly.

    Args:
        vc: Source clip.
        tgt_w: Target width.
        tgt_h: Target height.

    Returns:
        Cropped and resized clip.
    """
    src_ar = vc.w / max(vc.h, 1)
    tgt_ar = tgt_w / max(tgt_h, 1)

    if src_ar > tgt_ar:
        # Source is wider — crop sides.
        new_w = int(vc.h * tgt_ar)
        x1 = (vc.w - new_w) // 2
        vc = vc.cropped(x1=x1, x2=x1 + new_w)
    else:
        # Source is taller — crop top/bottom.
        new_h = int(vc.w / tgt_ar)
        y1 = (vc.h - new_h) // 2
        vc = vc.cropped(y1=y1, y2=y1 + new_h)

    return vc.with_effects([Resize((tgt_w, tgt_h))])


def _blur_bars(vc: VideoFileClip, tgt_w: int, tgt_h: int) -> VideoFileClip:
    """Place the clip over a blurred, scaled-up version of itself.

    The foreground is scaled to fit within the target dimensions
    (preserving aspect ratio), and the background is a heavily blurred
    version stretched to fill the frame.

    Args:
        vc: Source clip.
        tgt_w: Target width.
        tgt_h: Target height.

    Returns:
        Composite clip with blurred background bars.
    """
    from moviepy import CompositeVideoClip

    # Background: stretch to fill, then blur.
    bg = vc.with_effects([Resize((tgt_w, tgt_h))])

    def blur_frame(get_frame, t):
        """Apply a heavy Gaussian blur to each frame."""
        import cv2
        frame = get_frame(t)
        return cv2.GaussianBlur(frame, (0, 0), sigmaX=40, sigmaY=40)

    bg = bg.transform(blur_frame)

    # Foreground: scale to fit inside target, centred.
    scale = min(tgt_w / vc.w, tgt_h / vc.h)
    fg_w = int(vc.w * scale)
    fg_h = int(vc.h * scale)
    fg = vc.with_effects([Resize((fg_w, fg_h))])
    fg = fg.with_position(("center", "center"))

    composite = CompositeVideoClip([bg, fg], size=(tgt_w, tgt_h))
    # Preserve audio from the foreground clip.
    if vc.audio is not None:
        composite = composite.with_audio(vc.audio)
    return composite


def _letterbox(vc: VideoFileClip, tgt_w: int, tgt_h: int) -> VideoFileClip:
    """Scale to fit and pad with black bars (letterbox / pillarbox).

    Args:
        vc: Source clip.
        tgt_w: Target width.
        tgt_h: Target height.

    Returns:
        Padded clip at exactly (tgt_w, tgt_h).
    """
    from moviepy import CompositeVideoClip, ColorClip

    scale = min(tgt_w / vc.w, tgt_h / vc.h)
    fg_w = int(vc.w * scale)
    fg_h = int(vc.h * scale)
    fg = vc.with_effects([Resize((fg_w, fg_h))])
    fg = fg.with_position(("center", "center"))

    bg = ColorClip(size=(tgt_w, tgt_h), color=(0, 0, 0)).with_duration(vc.duration)

    composite = CompositeVideoClip([bg, fg], size=(tgt_w, tgt_h))
    if vc.audio is not None:
        composite = composite.with_audio(vc.audio)
    return composite


# ------------------------------------------------------------------
# Dead-air trimming
# ------------------------------------------------------------------

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

        above = np.where(abs_signal > _SILENCE_FLOOR)[0]
        if len(above) == 0:
            return vc

        first_sample = above[0]
        last_sample = above[-1]
        start_s = max(0.0, first_sample / sr - margin)
        end_s = min(vc.duration, last_sample / sr + margin)

        if end_s - start_s < 0.5:
            return vc

        if start_s > 0.2 or (vc.duration - end_s) > 0.2:
            logger.debug(
                "Trimming dead air: %.2fs–%.2fs → %.2fs–%.2fs",
                0.0, vc.duration, start_s, end_s,
            )
            return vc.subclipped(start_s, end_s)

    except Exception:
        logger.debug("Dead-air trimming failed — keeping original clip", exc_info=True)

    return vc


# ------------------------------------------------------------------
# Transitions
# ------------------------------------------------------------------

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
