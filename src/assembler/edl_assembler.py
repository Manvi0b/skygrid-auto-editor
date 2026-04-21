"""EDL-driven assembler — renders an Edit Decision List to a video file.

Unlike ``editor.assemble()`` (which concatenates whole clips with transitions),
this module consumes an EDL and honours each cut's exact in/out points on
the source clip.  It is the path used when music-driven, beat-aligned
sequencing is active.
"""

from __future__ import annotations

import logging
from pathlib import Path

from moviepy import VideoFileClip, concatenate_videoclips
from moviepy.video.fx import CrossFadeIn, CrossFadeOut, FadeIn, FadeOut, Resize

from src.config import Config
from src.models.edl import EDL
from src.models.output_profile import OutputProfile
from src.assembler.editor import _adapt_aspect_ratio  # reuse aspect logic

logger = logging.getLogger(__name__)


def assemble_from_edl(
    edl: EDL,
    config: Config,
    profile: OutputProfile | None = None,
) -> Path:
    """Render *edl* to an intermediate video file.

    Args:
        edl: Edit Decision List produced by the sequencer.
        config: Pipeline configuration.
        profile: Output profile override (falls back to ``config.output_profile``).

    Returns:
        Path to the assembled (intermediate) video.

    Raises:
        ValueError: If *edl* is empty.
    """
    if not edl.entries:
        raise ValueError("EDL is empty — nothing to assemble")

    target = profile or config.output_profile
    ar_mode = config.aspect_ratio_mode
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / "assembled.mp4"

    # Cache source clips so we only decode each file once.
    source_cache: dict[str, VideoFileClip] = {}
    sub_clips: list = []

    try:
        for i, entry in enumerate(edl.entries):
            key = str(entry.source_path)
            if key not in source_cache:
                source_cache[key] = VideoFileClip(str(entry.source_path))
            src = source_cache[key]

            s_in = max(0.0, min(entry.source_in, src.duration))
            s_out = max(s_in + 0.1, min(entry.source_out, src.duration))
            sub = src.subclipped(s_in, s_out)

            # Adapt aspect ratio to target profile.
            sub = _adapt_aspect_ratio(sub, target, ar_mode)

            # Apply transitions at cut boundaries.
            sub = _apply_edl_transition(
                sub,
                entry.transition_in,
                entry.transition_out,
                config.transition_duration,
                is_first=(i == 0),
                is_last=(i == len(edl.entries) - 1),
            )
            sub_clips.append(sub)

        # Concatenate.  If any entries use crossfade, use compose method.
        uses_xfade = any(
            e.transition_in == "crossfade" or e.transition_out == "crossfade"
            for e in edl.entries
        )
        method = "compose" if uses_xfade else "chain"
        padding = -config.transition_duration if uses_xfade and len(sub_clips) > 1 else 0
        final = concatenate_videoclips(sub_clips, method=method, padding=padding)

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
        for sc in source_cache.values():
            try:
                sc.close()
            except Exception:
                pass

    logger.info(
        "Assembled EDL → %s  (%d cuts, %.1fs)",
        output_path, len(edl.entries), edl.total_duration,
    )
    return output_path


def _apply_edl_transition(
    vc,
    t_in: str,
    t_out: str,
    duration: float,
    is_first: bool,
    is_last: bool,
):
    """Apply per-cut transitions honouring the EDL spec."""
    if duration <= 0 or vc.duration <= duration * 2:
        return vc

    effects = []
    match t_in:
        case "crossfade" if not is_first:
            effects.append(CrossFadeIn(duration))
        case "fade_black" if is_first:
            effects.append(FadeIn(duration))
    match t_out:
        case "crossfade" if not is_last:
            effects.append(CrossFadeOut(duration))
        case "fade_black" if is_last:
            effects.append(FadeOut(duration))

    if effects:
        vc = vc.with_effects(effects)
    return vc
