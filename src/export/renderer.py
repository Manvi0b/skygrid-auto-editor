"""FFmpeg-based final renderer for the assembled video."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from src.color.mood import mood_filter
from src.config import Config
from src.models.output_profile import OutputProfile

logger = logging.getLogger(__name__)


def render(
    assembled_path: Path,
    output_path: Path,
    config: Config,
    profile: OutputProfile | None = None,
) -> Path:
    """Re-encode *assembled_path* to the final output using the output profile.

    Applies the profile's resolution, codec, bitrate, and fps.  Optionally
    mixes in a background music track at the configured volume levels.

    Args:
        assembled_path: Path to the intermediate assembled video.
        output_path: Desired path for the final rendered file.
        config: Pipeline configuration.
        profile: Output profile override.  Falls back to ``config.output_profile``.

    Returns:
        Path to the rendered output file.

    Raises:
        RuntimeError: If ffmpeg encoding fails.
    """
    target = profile or config.output_profile
    tgt_w, tgt_h = target.width, target.height

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the ffmpeg command.
    inputs = ["-y", "-i", str(assembled_path)]
    filter_parts: list[str] = []

    # Video scaling / padding (safety pass — assembler already adapts,
    # but this ensures exact pixel dimensions in the final file).
    vf = (
        f"scale={tgt_w}:{tgt_h}:force_original_aspect_ratio=decrease,"
        f"pad={tgt_w}:{tgt_h}:(ow-iw)/2:(oh-ih)/2"
    )

    # Mood colour grade — appended so the final file has a consistent look.
    color_chain = mood_filter(getattr(config, "mood", "neutral"))
    if color_chain:
        vf = f"{vf},{color_chain}"
        logger.info("Applying mood grade: %s", config.mood)

    has_music = config.music_track is not None and config.music_track.exists()

    # ------------------------------------------------------------------
    # Audio graph (v0.10.0)
    # ------------------------------------------------------------------
    # Goals:
    #   * Platform-correct integrated loudness via `loudnorm` (default -14
    #     LUFS — YouTube / Spotify target).
    #   * Sidechain-compress the music bus with the nat-sound bus as the
    #     key, so music ducks when there's on-camera audio.
    #
    # Graph when music is present and ducking is on:
    #   [0:a]volume=orig          → [a0]           (nat-sound bus)
    #   [a0]asplit=2              → [a0m][a0k]     (mix + sidechain key)
    #   [1:a]volume=music         → [a1]           (music bus)
    #   [a1][a0k]sidechaincompress ... → [a1d]      (ducked music)
    #   [a0m][a1d]amix=2          → [amix]
    #   [amix]loudnorm=I=...      → [aout]
    # ------------------------------------------------------------------
    lufs_chain = (
        f"loudnorm=I={config.target_lufs}:"
        f"TP={config.loudnorm_tp}:"
        f"LRA={config.loudnorm_lra}"
    )

    if has_music:
        inputs.extend(["-i", str(config.music_track)])
        orig_vol = config.original_audio_volume
        music_vol = config.music_volume

        if config.duck_music:
            filter_parts.append(
                f"[0:a]volume={orig_vol},asplit=2[a0m][a0k];"
                f"[1:a]volume={music_vol}[a1];"
                f"[a1][a0k]sidechaincompress="
                f"threshold={config.duck_threshold}:"
                f"ratio={config.duck_ratio}:"
                f"attack={config.duck_attack}:"
                f"release={config.duck_release}[a1d];"
                f"[a0m][a1d]amix=inputs=2:duration=shortest:dropout_transition=0[amix];"
                f"[amix]{lufs_chain}[aout]"
            )
        else:
            filter_parts.append(
                f"[0:a]volume={orig_vol}[a0];"
                f"[1:a]volume={music_vol}[a1];"
                f"[a0][a1]amix=inputs=2:duration=shortest[amix];"
                f"[amix]{lufs_chain}[aout]"
            )
        audio_map = ["-map", "0:v", "-map", "[aout]"]
    else:
        # No music — still LUFS-normalise the nat-sound so exports are
        # consistent regardless of whether a track was supplied.
        filter_parts.append(f"[0:a]{lufs_chain}[aout]")
        audio_map = ["-map", "0:v", "-map", "[aout]"]

    cmd: list[str] = ["ffmpeg", *inputs]

    full_filter = f"[0:v]{vf}[vout];{';'.join(filter_parts)}"
    cmd.extend(["-filter_complex", full_filter])
    cmd.extend(["-map", "[vout]"])
    cmd.extend(audio_map)

    cmd.extend([
        "-r", str(target.fps),
        "-c:v", target.codec,
        "-b:v", target.bitrate,
        "-preset", target.preset,
        "-c:a", target.audio_codec,
        "-b:a", target.audio_bitrate,
        "-movflags", "+faststart",
        str(output_path),
    ])

    logger.info(
        "Rendering final output → %s  [%s  %dx%d  %s  %s]",
        output_path, target.name, tgt_w, tgt_h, target.codec, target.bitrate,
    )
    logger.debug("ffmpeg command: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"FFmpeg render failed: {exc.stderr}") from exc

    logger.info("Render complete: %s", output_path)
    return output_path
