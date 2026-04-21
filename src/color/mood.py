"""Mood presets expressed as FFmpeg filter strings.

The real pipeline wants proper 3D LUT files (.cube) applied via FFmpeg's
``lut3d`` filter — those are swappable in by dropping files into ``luts/``.
This module ships a built-in approximation using the ``eq``, ``curves``,
and ``colorbalance`` filters so that ``--mood`` produces a visible look
out of the box without external assets.
"""

from __future__ import annotations

from pathlib import Path

_LUT_DIR = Path(__file__).resolve().parent.parent.parent / "luts"


# Each mood maps to a comma-joined FFmpeg filter chain.  Keep chains
# colour-only; resolution / aspect happens upstream in the renderer.
_MOOD_FILTERS: dict[str, str] = {
    # Warm highlights, lifted blacks, teal shadows — real-estate default.
    "luxury": (
        "eq=contrast=1.05:saturation=1.05:gamma=1.02,"
        "colorbalance=rs=0.05:gs=0.00:bs=-0.05:"
        "rm=0.02:gm=0.00:bm=-0.02:"
        "rh=0.08:gh=0.02:bh=-0.06"
    ),
    # Saturated, punchy contrast.
    "energetic": (
        "eq=contrast=1.15:saturation=1.25:gamma=0.98"
    ),
    # Desaturated, crushed blacks, cool tint.
    "moody": (
        "eq=contrast=1.12:saturation=0.80:gamma=0.95,"
        "colorbalance=rs=-0.05:gs=0.00:bs=0.08:"
        "rm=-0.02:gm=0.00:bm=0.04"
    ),
    # High exposure, pastel tones.
    "bright": (
        "eq=contrast=0.98:brightness=0.05:saturation=0.95:gamma=1.05"
    ),
    # High contrast, deep shadows, orange/teal classic.
    "dramatic": (
        "eq=contrast=1.20:saturation=1.10:gamma=0.95,"
        "colorbalance=rs=0.08:gs=0.00:bs=-0.06:"
        "rh=-0.06:gh=0.00:bh=0.08"
    ),
    # No-op.
    "neutral": "",
}


def mood_filter(mood: str) -> str:
    """Return an FFmpeg filter chain for *mood* (empty string if none).

    If a ``luts/<mood>.cube`` file exists it takes precedence and is
    applied via ``lut3d``.  Otherwise the built-in filter-chain
    approximation is returned.

    Args:
        mood: Mood label.

    Returns:
        FFmpeg filter chain string (may be empty).
    """
    lut_path = _LUT_DIR / f"{mood}.cube"
    if lut_path.exists():
        return f"lut3d=file='{lut_path.as_posix()}'"
    return _MOOD_FILTERS.get(mood, "")


def available_moods() -> list[str]:
    """Return the list of built-in mood labels."""
    return sorted(_MOOD_FILTERS.keys())
