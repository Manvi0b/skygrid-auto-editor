"""Output profile — target platform format and encoding settings."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class OutputProfile:
    """Defines the target format for rendered output.

    Each profile encapsulates resolution, aspect ratio, codec, and bitrate
    settings for a specific platform (YouTube, Reels, TikTok, etc.).

    Attributes:
        name: Human-readable profile identifier (e.g. ``"youtube"``).
        aspect_ratio: Target aspect ratio as ``(width, height)`` integers
            (e.g. ``(16, 9)``).
        resolution: Target pixel dimensions as ``(width, height)``.
        fps: Target frame rate.
        codec: FFmpeg video codec name.
        bitrate: FFmpeg video bitrate string (e.g. ``"20M"``).
        audio_codec: FFmpeg audio codec name.
        audio_bitrate: FFmpeg audio bitrate string.
        preset: FFmpeg encoding preset.
        platform_tags: Labels for metadata embedding (e.g. ``["youtube", "web"]``).
        aspect_ratio_mode: How to handle clips that don't match the target
            aspect ratio — ``"crop"``, ``"blur_bars"``, or ``"letterbox"``.
    """

    name: str
    aspect_ratio: tuple[int, int] = (16, 9)
    resolution: tuple[int, int] = (1920, 1080)
    fps: int = 30
    codec: str = "libx264"
    bitrate: str = "20M"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    preset: str = "medium"
    platform_tags: list[str] = field(default_factory=list)
    aspect_ratio_mode: str = "blur_bars"

    @property
    def width(self) -> int:
        """Target width in pixels."""
        return self.resolution[0]

    @property
    def height(self) -> int:
        """Target height in pixels."""
        return self.resolution[1]

    @property
    def resolution_str(self) -> str:
        """Return resolution as ``"WxH"``."""
        return f"{self.resolution[0]}x{self.resolution[1]}"

    @property
    def is_portrait(self) -> bool:
        """True if the target is taller than it is wide."""
        return self.aspect_ratio[1] > self.aspect_ratio[0]

    @property
    def is_landscape(self) -> bool:
        """True if the target is wider than it is tall."""
        return self.aspect_ratio[0] > self.aspect_ratio[1]

    @property
    def is_square(self) -> bool:
        """True if the target aspect ratio is 1:1."""
        return self.aspect_ratio[0] == self.aspect_ratio[1]


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

BUILTIN_OUTPUT_PROFILES: dict[str, OutputProfile] = {
    "youtube": OutputProfile(
        name="youtube",
        aspect_ratio=(16, 9),
        resolution=(1920, 1080),
        fps=30,
        codec="libx264",
        bitrate="20M",
        audio_codec="aac",
        audio_bitrate="192k",
        preset="medium",
        platform_tags=["youtube", "web"],
    ),
    "youtube_4k": OutputProfile(
        name="youtube_4k",
        aspect_ratio=(16, 9),
        resolution=(3840, 2160),
        fps=30,
        codec="libx264",
        bitrate="45M",
        audio_codec="aac",
        audio_bitrate="320k",
        preset="slow",
        platform_tags=["youtube", "web", "4k"],
    ),
    "reels": OutputProfile(
        name="reels",
        aspect_ratio=(9, 16),
        resolution=(1080, 1920),
        fps=30,
        codec="libx264",
        bitrate="15M",
        audio_codec="aac",
        audio_bitrate="192k",
        preset="medium",
        platform_tags=["instagram", "reels", "mobile"],
    ),
    "tiktok": OutputProfile(
        name="tiktok",
        aspect_ratio=(9, 16),
        resolution=(1080, 1920),
        fps=30,
        codec="libx264",
        bitrate="15M",
        audio_codec="aac",
        audio_bitrate="192k",
        preset="medium",
        platform_tags=["tiktok", "mobile"],
    ),
    "instagram_square": OutputProfile(
        name="instagram_square",
        aspect_ratio=(1, 1),
        resolution=(1080, 1080),
        fps=30,
        codec="libx264",
        bitrate="12M",
        audio_codec="aac",
        audio_bitrate="192k",
        preset="medium",
        platform_tags=["instagram", "square"],
    ),
    "twitter": OutputProfile(
        name="twitter",
        aspect_ratio=(16, 9),
        resolution=(1280, 720),
        fps=30,
        codec="libx264",
        bitrate="10M",
        audio_codec="aac",
        audio_bitrate="128k",
        preset="medium",
        platform_tags=["twitter", "web"],
    ),
}
