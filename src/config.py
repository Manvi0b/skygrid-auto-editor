"""Configuration loader for the editing pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass(frozen=True, slots=True)
class Config:
    """Parsed, validated pipeline configuration.

    Attributes:
        input_dir: Directory containing source clips.
        output_dir: Directory for rendered output.
        plugins_dir: Directory for user-supplied analyzer plugins.
        max_duration: Maximum output video duration in seconds.
        min_clip_score: Minimum composite score to include a clip.
        sort_by: Ordering strategy for accepted clips.
        enabled_analyzers: List of analyzer names to run.
        transition_style: Transition type between clips.
        transition_duration: Transition length in seconds.
        music_track: Optional path to a background music file.
        sync_to_beat: Whether to align cuts to musical beats.
        original_audio_volume: Volume multiplier for clip audio.
        music_volume: Volume multiplier for background music.
        resolution: Output resolution as ``"WIDTHxHEIGHT"``.
        fps: Output frames per second.
        codec: FFmpeg video codec.
        bitrate: FFmpeg video bitrate string.
        audio_codec: FFmpeg audio codec.
        audio_bitrate: FFmpeg audio bitrate string.
        preset: FFmpeg encoding preset.
    """

    input_dir: Path = Path("./input")
    output_dir: Path = Path("./output")
    plugins_dir: Path = Path("./plugins")
    max_duration: float = 120.0
    min_clip_score: float = 0.3
    sort_by: str = "score"
    enabled_analyzers: list[str] = field(default_factory=lambda: [
        "shake_detector", "scene_detector", "audio_analyzer",
    ])
    transition_style: str = "crossfade"
    transition_duration: float = 0.5
    music_track: Path | None = None
    sync_to_beat: bool = False
    original_audio_volume: float = 0.7
    music_volume: float = 0.3
    resolution: str = "1920x1080"
    fps: int = 30
    codec: str = "libx264"
    bitrate: str = "8M"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    preset: str = "medium"


def load_config(path: Path | None = None) -> Config:
    """Load and validate a YAML configuration file.

    Args:
        path: Path to the config file.  Falls back to the repo-root
            ``config.yaml`` when *None*.

    Returns:
        A fully populated Config instance.
    """
    config_path = path or _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        logger.warning("Config file not found at %s — using defaults", config_path)
        return Config()

    with open(config_path, "r") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    paths = raw.get("paths", {})
    pipeline = raw.get("pipeline", {})
    transitions = raw.get("transitions", {})
    audio = raw.get("audio", {})
    export = raw.get("export", {})

    music = audio.get("music_track")

    return Config(
        input_dir=Path(paths.get("input", "./input")),
        output_dir=Path(paths.get("output", "./output")),
        plugins_dir=Path(paths.get("plugins", "./plugins")),
        max_duration=float(pipeline.get("max_duration", 120)),
        min_clip_score=float(pipeline.get("min_clip_score", 0.3)),
        sort_by=str(pipeline.get("sort_by", "score")),
        enabled_analyzers=list(pipeline.get("enabled_analyzers", [])),
        transition_style=str(transitions.get("style", "crossfade")),
        transition_duration=float(transitions.get("duration", 0.5)),
        music_track=Path(music) if music else None,
        sync_to_beat=bool(audio.get("sync_to_beat", False)),
        original_audio_volume=float(audio.get("original_audio_volume", 0.7)),
        music_volume=float(audio.get("music_volume", 0.3)),
        resolution=str(export.get("resolution", "1920x1080")),
        fps=int(export.get("fps", 30)),
        codec=str(export.get("codec", "libx264")),
        bitrate=str(export.get("bitrate", "8M")),
        audio_codec=str(export.get("audio_codec", "aac")),
        audio_bitrate=str(export.get("audio_bitrate", "192k")),
        preset=str(export.get("preset", "medium")),
    )
