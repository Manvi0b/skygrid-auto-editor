"""Scans an input directory and produces Clip objects for the pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from src.models.clip import Clip
from src.utils.ffmpeg_helpers import probe_video_info

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mts",
})


def load_clips(input_dir: Path) -> list[Clip]:
    """Recursively scan *input_dir* for video files and return Clip objects.

    Each clip is probed for duration, resolution, and frame rate via ffprobe.

    Args:
        input_dir: Root directory to scan.

    Returns:
        A list of Clip instances sorted alphabetically by filename.

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
        clip = Clip(
            path=file_path.resolve(),
            duration=info["duration"],
            width=info["width"],
            height=info["height"],
            fps=info["fps"],
        )
        clips.append(clip)
        logger.info(
            "Loaded clip: %s  %.2fs  %dx%d @ %.1f fps",
            file_path.name,
            clip.duration,
            clip.width,
            clip.height,
            clip.fps,
        )

    logger.info("Found %d clip(s) in %s", len(clips), input_dir)
    return clips
