"""Thin wrappers around FFmpeg / ffprobe for common operations."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def probe_video_info(video_path: Path) -> dict[str, Any]:
    """Return duration, resolution, and fps for a video file via ffprobe.

    Args:
        video_path: Path to the video file.

    Returns:
        Dict with keys ``duration`` (float, seconds), ``width`` (int),
        ``height`` (int), ``fps`` (float).  Missing values default to 0.
    """
    info: dict[str, Any] = {
        "duration": 0.0,
        "width": 0,
        "height": 0,
        "fps": 0.0,
    }
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)

        # Duration from format container.
        fmt = data.get("format", {})
        info["duration"] = float(fmt.get("duration", 0))

        # Resolution and fps from the first video stream.
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                info["width"] = int(stream.get("width", 0))
                info["height"] = int(stream.get("height", 0))
                # r_frame_rate is typically "30/1" or "30000/1001".
                rfr = stream.get("r_frame_rate", "0/1")
                num, den = rfr.split("/")
                info["fps"] = round(int(num) / max(int(den), 1), 3)
                break

    except Exception:
        logger.exception("ffprobe failed for %s", video_path)

    return info


def probe_duration(video_path: Path) -> float:
    """Return the duration of a video file in seconds via ffprobe.

    Args:
        video_path: Path to the video file.

    Returns:
        Duration in seconds, or 0.0 on failure.
    """
    return probe_video_info(video_path)["duration"]


def extract_audio_to_wav(video_path: Path) -> Path:
    """Extract the audio track from a video file as a temporary WAV.

    Args:
        video_path: Path to the source video.

    Returns:
        Path to the extracted WAV file.  Caller is responsible for cleanup.

    Raises:
        RuntimeError: If ffmpeg extraction fails.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    wav_path = Path(tmp.name)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(video_path),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "22050",
                "-ac", "1",
                str(wav_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        wav_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Audio extraction failed for {video_path}: {exc.stderr}"
        ) from exc

    return wav_path


def has_audio_stream(video_path: Path) -> bool:
    """Check whether a video file contains at least one audio stream.

    Args:
        video_path: Path to the video file.

    Returns:
        True if an audio stream exists, False otherwise.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-select_streams", "a",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        return len(data.get("streams", [])) > 0
    except Exception:
        return False
