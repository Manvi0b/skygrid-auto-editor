"""FFmpeg-based final renderer for the assembled video."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from src.config import Config

logger = logging.getLogger(__name__)


def render(
    assembled_path: Path,
    output_path: Path,
    config: Config,
) -> Path:
    """Re-encode *assembled_path* to the final output with configured settings.

    Optionally mixes in a background music track at the configured volume
    levels.

    Args:
        assembled_path: Path to the intermediate assembled video.
        output_path: Desired path for the final rendered file.
        config: Pipeline configuration.

    Returns:
        Path to the rendered output file.

    Raises:
        RuntimeError: If ffmpeg encoding fails.
    """
    width, height = config.resolution.split("x")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the ffmpeg command.
    inputs = ["-y", "-i", str(assembled_path)]
    filter_parts: list[str] = []

    # Video scaling / padding.
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )

    has_music = config.music_track is not None and config.music_track.exists()

    if has_music:
        inputs.extend(["-i", str(config.music_track)])
        orig_vol = config.original_audio_volume
        music_vol = config.music_volume
        # Mix original audio (stream 0:a) with music (stream 1:a).
        filter_parts.append(
            f"[0:a]volume={orig_vol}[a0];"
            f"[1:a]volume={music_vol}[a1];"
            f"[a0][a1]amix=inputs=2:duration=shortest[aout]"
        )
        audio_map = ["-map", "0:v", "-map", "[aout]"]
    else:
        audio_map = []

    cmd: list[str] = ["ffmpeg", *inputs]

    if filter_parts:
        # Combine video and audio filters.
        full_filter = f"[0:v]{vf}[vout];{';'.join(filter_parts)}"
        cmd.extend(["-filter_complex", full_filter])
        cmd.extend(["-map", "[vout]"])
        cmd.extend(audio_map)
    else:
        cmd.extend(["-vf", vf])

    cmd.extend([
        "-r", str(config.fps),
        "-c:v", config.codec,
        "-b:v", config.bitrate,
        "-preset", config.preset,
        "-c:a", config.audio_codec,
        "-b:a", config.audio_bitrate,
        "-movflags", "+faststart",
        str(output_path),
    ])

    logger.info("Rendering final output → %s", output_path)
    logger.debug("ffmpeg command: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"FFmpeg render failed: {exc.stderr}") from exc

    logger.info("Render complete: %s", output_path)
    return output_path
