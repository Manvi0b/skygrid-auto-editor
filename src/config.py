"""Configuration loader for the editing pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.models.output_profile import BUILTIN_OUTPUT_PROFILES, OutputProfile
from src.models.source import BUILTIN_SOURCE_PROFILES, SourceProfile

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
        output_profile: Active output profile (platform target).
        source_profiles: Available source device profiles.
        aspect_ratio_mode: How to handle mismatched aspect ratios.
    """

    input_dir: Path = Path("./input")
    output_dir: Path = Path("./output")
    plugins_dir: Path = Path("./plugins")
    max_duration: float = 120.0
    min_clip_score: float = 30.0
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
    bitrate: str = "20M"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    preset: str = "medium"
    output_profile: OutputProfile = field(
        default_factory=lambda: BUILTIN_OUTPUT_PROFILES["youtube"],
    )
    source_profiles: dict[str, SourceProfile] = field(
        default_factory=lambda: dict(BUILTIN_SOURCE_PROFILES),
    )
    aspect_ratio_mode: str = "blur_bars"

    # --- v0.3.0 — music-driven editing ---
    target_length: float = 0.0   # 0 = use max_duration; else overrides it
    pacing: str = "medium"        # slow | medium | fast | cinematic | energetic
    mood: str = "neutral"         # luxury | energetic | moody | bright | dramatic | neutral
    property_type: str = "generic"  # real_estate | event | commercial | landscape | generic
    hero_clips: list[str] = field(default_factory=list)     # must-include clip names
    must_include: list[str] = field(default_factory=list)   # must-include segments
    beat_aligned: bool = True     # when music_track is set, use beat-aligned sequencer
    project_name: str = "skygrid_project"
    outputs: list[dict] = field(default_factory=list)  # multi-output spec
    client_name: str = ""         # used in filename when provided

    # --- v0.10.0 — loudness + music ducking ---
    target_lufs: float = -14.0       # integrated loudness target (YouTube ≈ -14)
    loudnorm_tp: float = -1.5        # true-peak ceiling (dBTP)
    loudnorm_lra: float = 11.0       # loudness range target
    duck_music: bool = True          # sidechain-duck music under speech/nat-sound
    duck_threshold: float = 0.05     # sidechaincompress threshold (linear)
    duck_ratio: float = 8.0          # compression ratio while ducking
    duck_attack: float = 20.0        # attack in ms
    duck_release: float = 250.0      # release in ms

    # --- v0.11.0 — branding / title / outro / lower-third ---
    # Stored as a dict to avoid ballooning Config with 20+ nested fields.
    # Shape documented in config.yaml under `branding:`.
    branding: dict = field(default_factory=dict)

    # --- v0.12.0 — per-clip preprocessing (stabilization + horizon level) ---
    preprocessing: dict = field(default_factory=dict)


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

    # --- Source profiles ---
    source_profiles = dict(BUILTIN_SOURCE_PROFILES)
    for name, sp_raw in raw.get("source_profiles", {}).items():
        source_profiles[name] = SourceProfile(
            name=name,
            device_type=str(sp_raw.get("device_type", "generic")),
            default_orientation=str(sp_raw.get("default_orientation", "horizontal")),
            has_gimbal=bool(sp_raw.get("has_gimbal", False)),
            typical_artifacts=list(sp_raw.get("typical_artifacts", [])),
        )

    # --- Output profiles ---
    output_profiles = dict(BUILTIN_OUTPUT_PROFILES)
    for name, op_raw in raw.get("output_profiles", {}).items():
        output_profiles[name] = OutputProfile(
            name=name,
            aspect_ratio=tuple(op_raw.get("aspect_ratio", [16, 9])),
            resolution=tuple(op_raw.get("resolution", [1920, 1080])),
            fps=int(op_raw.get("fps", 30)),
            codec=str(op_raw.get("codec", "libx264")),
            bitrate=str(op_raw.get("bitrate", "20M")),
            audio_codec=str(op_raw.get("audio_codec", "aac")),
            audio_bitrate=str(op_raw.get("audio_bitrate", "192k")),
            preset=str(op_raw.get("preset", "medium")),
            platform_tags=list(op_raw.get("platform_tags", [])),
            aspect_ratio_mode=str(op_raw.get("aspect_ratio_mode", "blur_bars")),
        )

    default_profile_name = str(raw.get("default_output_profile", "youtube"))
    active_profile = output_profiles.get(
        default_profile_name,
        BUILTIN_OUTPUT_PROFILES["youtube"],
    )

    project_cfg = raw.get("project", {})
    edit_cfg = raw.get("edit", {})
    branding_cfg = raw.get("branding", {})

    return Config(
        input_dir=Path(paths.get("input", "./input")),
        output_dir=Path(paths.get("output", "./output")),
        plugins_dir=Path(paths.get("plugins", "./plugins")),
        max_duration=float(pipeline.get("max_duration", 120)),
        min_clip_score=float(pipeline.get("min_clip_score", 30)),
        sort_by=str(pipeline.get("sort_by", "score")),
        enabled_analyzers=list(pipeline.get("enabled_analyzers", [])),
        transition_style=str(transitions.get("style", "crossfade")),
        transition_duration=float(transitions.get("duration", 0.5)),
        music_track=Path(music) if music else None,
        sync_to_beat=bool(audio.get("sync_to_beat", False)),
        original_audio_volume=float(audio.get("original_audio_volume", 0.7)),
        music_volume=float(audio.get("music_volume", 0.3)),
        resolution=active_profile.resolution_str,
        fps=active_profile.fps,
        codec=active_profile.codec,
        bitrate=active_profile.bitrate,
        audio_codec=active_profile.audio_codec,
        audio_bitrate=active_profile.audio_bitrate,
        preset=active_profile.preset,
        output_profile=active_profile,
        source_profiles=source_profiles,
        aspect_ratio_mode=str(raw.get("aspect_ratio_mode", active_profile.aspect_ratio_mode)),
        target_length=float(edit_cfg.get("target_length", 0.0)),
        pacing=str(edit_cfg.get("pacing", "medium")),
        mood=str(edit_cfg.get("mood", "neutral")),
        property_type=str(edit_cfg.get("property_type", "generic")),
        hero_clips=list(edit_cfg.get("hero_clips", [])),
        must_include=list(edit_cfg.get("must_include", [])),
        beat_aligned=bool(edit_cfg.get("beat_aligned", True)),
        project_name=str(project_cfg.get("name", "skygrid_project")),
        outputs=list(raw.get("outputs", [])),
        client_name=str(branding_cfg.get("client_name", "")),
        target_lufs=float(audio.get("target_lufs", -14.0)),
        loudnorm_tp=float(audio.get("loudnorm_tp", -1.5)),
        loudnorm_lra=float(audio.get("loudnorm_lra", 11.0)),
        duck_music=bool(audio.get("duck_music", True)),
        duck_threshold=float(audio.get("duck_threshold", 0.05)),
        duck_ratio=float(audio.get("duck_ratio", 8.0)),
        duck_attack=float(audio.get("duck_attack", 20.0)),
        duck_release=float(audio.get("duck_release", 250.0)),
        branding=dict(branding_cfg),
        preprocessing=dict(raw.get("preprocessing", {})),
    )


def resolve_output_profile(
    config: Config,
    profile_name: str,
) -> Config:
    """Return a new Config with the output profile overridden.

    Also updates resolution, fps, codec, bitrate, etc. to match the new
    profile.

    Args:
        config: Current configuration.
        profile_name: Name of a built-in or user-defined output profile.

    Returns:
        A new Config reflecting the chosen profile.

    Raises:
        click.BadParameter: If the profile name is unknown.
    """
    all_profiles = {**BUILTIN_OUTPUT_PROFILES}
    # Merge any user-defined profiles from config.
    # (They're already baked into config at load time, but the profile
    #  might have been specified on the CLI after loading.)
    if profile_name not in all_profiles:
        raise ValueError(
            f"Unknown output profile '{profile_name}'. "
            f"Available: {', '.join(sorted(all_profiles))}"
        )

    profile = all_profiles[profile_name]
    return Config(
        input_dir=config.input_dir,
        output_dir=config.output_dir,
        plugins_dir=config.plugins_dir,
        max_duration=config.max_duration,
        min_clip_score=config.min_clip_score,
        sort_by=config.sort_by,
        enabled_analyzers=config.enabled_analyzers,
        transition_style=config.transition_style,
        transition_duration=config.transition_duration,
        music_track=config.music_track,
        sync_to_beat=config.sync_to_beat,
        original_audio_volume=config.original_audio_volume,
        music_volume=config.music_volume,
        resolution=profile.resolution_str,
        fps=profile.fps,
        codec=profile.codec,
        bitrate=profile.bitrate,
        audio_codec=profile.audio_codec,
        audio_bitrate=profile.audio_bitrate,
        preset=profile.preset,
        output_profile=profile,
        source_profiles=config.source_profiles,
        aspect_ratio_mode=profile.aspect_ratio_mode,
        target_length=config.target_length,
        pacing=config.pacing,
        mood=config.mood,
        property_type=config.property_type,
        hero_clips=config.hero_clips,
        must_include=config.must_include,
        beat_aligned=config.beat_aligned,
        project_name=config.project_name,
        outputs=config.outputs,
        client_name=config.client_name,
        target_lufs=config.target_lufs,
        loudnorm_tp=config.loudnorm_tp,
        loudnorm_lra=config.loudnorm_lra,
        duck_music=config.duck_music,
        duck_threshold=config.duck_threshold,
        duck_ratio=config.duck_ratio,
        duck_attack=config.duck_attack,
        duck_release=config.duck_release,
        branding=config.branding,
        preprocessing=config.preprocessing,
    )
