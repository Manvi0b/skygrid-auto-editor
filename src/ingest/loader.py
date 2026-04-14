"""Scans an input directory and produces Clip objects for the pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from src.models.clip import Clip
from src.models.output_profile import OutputProfile
from src.utils.ffmpeg_helpers import (
    detect_orientation,
    detect_source_from_tags,
    probe_video_info,
)

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mts",
})


def load_clips(
    input_dir: Path,
    target_profile: OutputProfile | None = None,
) -> list[Clip]:
    """Recursively scan *input_dir* for video files and return Clip objects.

    Each clip is probed for duration, resolution, frame rate, orientation,
    and source device.  When *target_profile* is provided the returned list
    is sorted so that clips matching the target orientation come first,
    followed by the rest — all sub-sorted alphabetically.

    Args:
        input_dir: Root directory to scan.
        target_profile: Optional output profile used to prioritise clips
            whose orientation matches the target aspect ratio.

    Returns:
        A list of Clip instances.

    Raises:
        FileNotFoundError: If *input_dir* does not exist.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    clips: list[Clip] = []
    for file_path in sorted(input_dir.rglob("*")):
        if file_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        info = probe_video_info(file_path)
        orientation = detect_orientation(info["width"], info["height"])
        source = detect_source_from_tags(info.get("format_tags", {}))

        clip = Clip(
            path=file_path.resolve(),
            duration=info["duration"],
            width=info["width"],
            height=info["height"],
            fps=info["fps"],
            orientation=orientation,
            source_profile=source,
        )
        clips.append(clip)
        logger.info(
            "Loaded clip: %s  %.2fs  %dx%d @ %.1f fps  %s  src=%s",
            file_path.name,
            clip.duration,
            clip.width,
            clip.height,
            clip.fps,
            orientation,
            source or "unknown",
        )

    logger.info("Found %d clip(s) in %s", len(clips), input_dir)

    # Auto-sort: matching orientation first.
    if target_profile is not None:
        clips = _sort_by_orientation_match(clips, target_profile)

    return clips


def _sort_by_orientation_match(
    clips: list[Clip],
    profile: OutputProfile,
) -> list[Clip]:
    """Sort clips so those matching the target orientation come first.

    Args:
        clips: Unsorted clip list.
        profile: Target output profile.

    Returns:
        Re-ordered list with matching clips first, then others.
    """
    if profile.is_portrait:
        target = "vertical"
    elif profile.is_square:
        target = "square"
    else:
        target = "horizontal"

    matching = [c for c in clips if c.orientation == target]
    other = [c for c in clips if c.orientation != target]

    if matching and other:
        logger.info(
            "Orientation sort: %d matching '%s', %d non-matching",
            len(matching), target, len(other),
        )

    return matching + other
