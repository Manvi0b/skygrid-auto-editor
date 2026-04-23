"""Branding pass — prepends a title card, appends an outro card, and
optionally overlays a lower-third text banner on the assembled video.

Runs between :func:`src.assembler.editor.assemble` and
:func:`src.export.renderer.render`.  All three elements are optional
and driven off the ``branding:`` block in ``config.yaml``:

.. code-block:: yaml

    branding:
      client_name: "Acme Real Estate"
      title_card:
        enabled: true
        text: "123 Sunset Blvd"      # defaults to project_name
        subtitle: "Acme Real Estate"  # defaults to client_name
        duration: 2.0
        background: "black"
        text_color: "white"
        font_size: 84
      outro_card:
        enabled: true
        text: "acme.com"
        duration: 2.0
      lower_third:
        enabled: true
        text: "Listed by Acme"      # defaults to client_name
        start: 1.0
        duration: 3.0

Everything is done in a single ffmpeg ``filter_complex`` call so the
concat never touches the disk as an intermediate.  Cards are generated
inline via the ``color`` / ``anullsrc`` lavfi sources at the target
profile's resolution / fps so concat works without a re-sample.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path
from typing import Any

from src.config import Config
from src.models.output_profile import OutputProfile

logger = logging.getLogger(__name__)


def apply_branding(
    assembled_path: Path,
    config: Config,
    profile: OutputProfile | None = None,
) -> Path:
    """Produce a branded copy of *assembled_path* (or return it unchanged).

    Args:
        assembled_path: The intermediate video from the assembler.
        config: Pipeline configuration (reads ``config.branding``).
        profile: Output profile override.  Falls back to
            ``config.output_profile``.

    Returns:
        Path to the branded video (``<stem>_branded.mp4``) if any
        branding element is enabled, otherwise the original
        ``assembled_path`` unchanged.
    """
    b = config.branding or {}
    title = b.get("title_card") or {}
    outro = b.get("outro_card") or {}
    lt = b.get("lower_third") or {}

    if not (title.get("enabled") or outro.get("enabled") or lt.get("enabled")):
        return assembled_path

    target = profile or config.output_profile
    w, h = target.width, target.height
    fps = target.fps

    branded = assembled_path.with_name(assembled_path.stem + "_branded.mp4")

    title_text = _pick(title, "text", config.project_name)
    title_sub = _pick(title, "subtitle", config.client_name)
    title_dur = float(title.get("duration", 2.0))
    title_bg = str(title.get("background", "black"))
    title_fg = str(title.get("text_color", "white"))
    title_font = int(title.get("font_size") or max(48, h // 14))

    outro_text = _pick(outro, "text", "")
    outro_dur = float(outro.get("duration", 2.0))
    outro_bg = str(outro.get("background", "black"))
    outro_fg = str(outro.get("text_color", "white"))
    outro_font = int(outro.get("font_size") or max(48, h // 14))

    lt_text = _pick(lt, "text", config.client_name)
    lt_start = float(lt.get("start", 1.0))
    lt_dur = float(lt.get("duration", 3.0))
    lt_font = int(lt.get("font_size") or max(28, h // 24))
    lt_fg = str(lt.get("text_color", "white"))
    lt_bg = str(lt.get("background", "black@0.55"))  # ffmpeg supports alpha on box

    # ------------------------------------------------------------------
    # Build filter_complex.  Inputs are added in this order:
    #   [0] assembled.mp4 (video + audio)
    #   [1] title color video       (if enabled)
    #   [2] title silence audio     (if enabled)
    #   [3] outro color video       (if enabled)
    #   [4] outro silence audio     (if enabled)
    # ------------------------------------------------------------------
    inputs: list[str] = ["-i", str(assembled_path)]
    stream_idx = 1  # next free ffmpeg input/stream index ([0] is the body).
    parts: list[str] = []

    # Body video w/ optional lower-third overlay.
    # Scale/pad first so cards (generated at target w×h) can concat cleanly
    # even when the assembler passed through a clip at a different size.
    body_scale = (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )
    if lt.get("enabled") and lt_text:
        escaped = _escape_text(lt_text)
        drawtext = (
            f"drawtext=text='{escaped}':fontcolor={lt_fg}:fontsize={lt_font}:"
            f"box=1:boxcolor={lt_bg}:boxborderw=18:"
            f"x=(w*0.06):y=h-(h*0.14):"
            f"enable='between(t,{lt_start},{lt_start + lt_dur})'"
        )
        parts.append(f"[0:v]{body_scale},{drawtext}[vbody]")
    else:
        parts.append(f"[0:v]{body_scale}[vbody]")
    parts.append("[0:a]aresample=48000[abody]")

    body_v = "[vbody]"
    body_a = "[abody]"

    segments_v: list[str] = []
    segments_a: list[str] = []

    if title.get("enabled"):
        n = stream_idx
        inputs.extend(["-f", "lavfi", "-i",
                       f"color=c={title_bg}:s={w}x{h}:r={fps}:d={title_dur}"])
        stream_idx += 1
        a_idx = stream_idx
        inputs.extend(["-f", "lavfi", "-i",
                       f"anullsrc=channel_layout=stereo:sample_rate=48000"])
        stream_idx += 1
        sub_line = ""
        if title_sub:
            sub_line = (
                f",drawtext=text='{_escape_text(title_sub)}':"
                f"fontcolor={title_fg}:fontsize={max(24, title_font // 2)}:"
                f"x=(w-text_w)/2:y=(h+text_h)/2+40"
            )
        parts.append(
            f"[{n}:v]drawtext=text='{_escape_text(title_text)}':"
            f"fontcolor={title_fg}:fontsize={title_font}:"
            f"x=(w-text_w)/2:y=(h-text_h)/2{sub_line}"
            f",fade=t=in:st=0:d=0.4,fade=t=out:st={max(0.1, title_dur - 0.4)}:d=0.4[vt]"
        )
        parts.append(f"[{a_idx}:a]atrim=duration={title_dur}[at]")
        segments_v.append("[vt]")
        segments_a.append("[at]")

    segments_v.append(body_v)
    segments_a.append(body_a)

    if outro.get("enabled"):
        n = stream_idx
        inputs.extend(["-f", "lavfi", "-i",
                       f"color=c={outro_bg}:s={w}x{h}:r={fps}:d={outro_dur}"])
        stream_idx += 1
        a_idx = stream_idx
        inputs.extend(["-f", "lavfi", "-i",
                       f"anullsrc=channel_layout=stereo:sample_rate=48000"])
        stream_idx += 1
        parts.append(
            f"[{n}:v]drawtext=text='{_escape_text(outro_text)}':"
            f"fontcolor={outro_fg}:fontsize={outro_font}:"
            f"x=(w-text_w)/2:y=(h-text_h)/2"
            f",fade=t=in:st=0:d=0.4,fade=t=out:st={max(0.1, outro_dur - 0.4)}:d=0.4[vo]"
        )
        parts.append(f"[{a_idx}:a]atrim=duration={outro_dur}[ao]")
        segments_v.append("[vo]")
        segments_a.append("[ao]")

    # Concat.
    n_seg = len(segments_v)
    concat_in = "".join(v + a for v, a in zip(segments_v, segments_a))
    parts.append(f"{concat_in}concat=n={n_seg}:v=1:a=1[vout][aout]")

    filter_complex = ";".join(parts)

    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "medium",
        "-c:a", "aac", "-b:a", "192k",
        "-r", str(fps),
        str(branded),
    ]
    logger.info(
        "Branding: title=%s  outro=%s  lower_third=%s",
        bool(title.get("enabled")), bool(outro.get("enabled")),
        bool(lt.get("enabled")),
    )
    logger.debug("ffmpeg branding cmd: %s", " ".join(shlex.quote(c) for c in cmd))
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        logger.error("Branding ffmpeg failed — falling back to un-branded assembly.\n%s",
                     exc.stderr[-1200:] if exc.stderr else "")
        return assembled_path

    logger.info("Branded → %s", branded)
    return branded


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _pick(d: dict[str, Any], key: str, fallback: str) -> str:
    """Return ``d[key]`` if truthy, otherwise *fallback*."""
    v = d.get(key)
    if v is None or v == "":
        return fallback
    return str(v)


def _escape_text(s: str) -> str:
    """Escape text for ffmpeg drawtext (single-quote-delimited strings).

    drawtext treats ``\\``, ``:``, ``'`` and ``%`` specially — plus it
    parses the filter argument list on ``:``, so colons inside the text
    must be escaped too.
    """
    # Order matters: escape backslashes first.
    s = s.replace("\\", "\\\\")
    s = s.replace("'", "\\'")
    s = s.replace(":", "\\:")
    s = s.replace("%", "\\%")
    return s
