"""CLI entry point for skygrid-auto-editor."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from src.config import Config, load_config, resolve_output_profile
from src.models.output_profile import BUILTIN_OUTPUT_PROFILES
from src.pipeline import analyze_only, run, run_all

_PROFILE_NAMES = list(BUILTIN_OUTPUT_PROFILES.keys())


@click.group()
@click.option(
    "--config", "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to config.yaml (default: repo-root config.yaml).",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable debug logging.",
)
@click.pass_context
def cli(ctx: click.Context, config_path: Path | None, verbose: bool) -> None:
    """SkyGrid Auto Editor — automated drone-footage editing pipeline."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config_path)


# ------------------------------------------------------------------
# edit — full pipeline
# ------------------------------------------------------------------

@cli.command()
@click.option(
    "--input", "input_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Override input directory.",
)
@click.option(
    "--output", "output_file",
    type=click.Path(path_type=Path),
    default=None,
    help="Override output file path (e.g. ./output/final.mp4).",
)
@click.option(
    "--profile", "profile_name",
    type=click.Choice(_PROFILE_NAMES, case_sensitive=False),
    default=None,
    help="Output profile: youtube, reels, tiktok, instagram_square, twitter, youtube_4k.",
)
@click.option("--music", "music_track", type=click.Path(exists=True, path_type=Path), default=None,
              help="Path to a music track — enables beat-aligned sequencing.")
@click.option("--target-length", "target_length", type=float, default=None,
              help="Target edit length in seconds (overrides max_duration).")
@click.option("--pacing", type=click.Choice(["slow", "medium", "fast", "cinematic", "energetic"]),
              default=None, help="Cut rhythm: slow / medium / fast / cinematic / energetic.")
@click.option("--mood", type=click.Choice(["luxury", "energetic", "moody", "bright", "dramatic", "neutral"]),
              default=None, help="Mood preset (drives colour and pacing bias).")
@click.option("--no-beats", is_flag=True, default=False,
              help="Disable beat-aligned sequencing even if music is set.")
@click.pass_context
def edit(
    ctx: click.Context,
    input_dir: Path | None,
    output_file: Path | None,
    profile_name: str | None,
    music_track: Path | None,
    target_length: float | None,
    pacing: str | None,
    mood: str | None,
    no_beats: bool,
) -> None:
    """Run the full editing pipeline: analyze → rank → sequence → render."""
    config: Config = ctx.obj["config"]
    if input_dir:
        config = _override_input(config, input_dir)
    if profile_name:
        config = resolve_output_profile(config, profile_name)
    config = _apply_edit_overrides(
        config, music_track=music_track, target_length=target_length,
        pacing=pacing, mood=mood, no_beats=no_beats,
    )

    click.echo(
        f"Profile: {config.output_profile.name}  "
        f"({config.output_profile.resolution_str}  "
        f"{config.output_profile.aspect_ratio[0]}:{config.output_profile.aspect_ratio[1]}  "
        f"AR mode: {config.aspect_ratio_mode})"
    )

    result = run(config, output_path=output_file)
    click.echo(f"Done — output saved to {result}")


# ------------------------------------------------------------------
# analyze — score clips without assembling
# ------------------------------------------------------------------

@cli.command()
@click.option(
    "--input", "input_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Override input directory.",
)
@click.option(
    "--profile", "profile_name",
    type=click.Choice(_PROFILE_NAMES, case_sensitive=False),
    default=None,
    help="Output profile (for orientation sorting).",
)
@click.pass_context
def analyze(
    ctx: click.Context,
    input_dir: Path | None,
    profile_name: str | None,
) -> None:
    """Analyze all clips and print scores (no assembly or rendering)."""
    config: Config = ctx.obj["config"]
    if input_dir:
        config = _override_input(config, input_dir)
    if profile_name:
        config = resolve_output_profile(config, profile_name)

    clips = analyze_only(config)

    click.echo(f"\n{'='*72}")
    click.echo(f"  Analysis complete — {len(clips)} clip(s)")
    click.echo(f"{'='*72}\n")

    for i, clip in enumerate(clips, 1):
        _print_clip_detail(i, clip)

    click.echo(f"{'='*72}")


# ------------------------------------------------------------------
# list — ranked summary table
# ------------------------------------------------------------------

@cli.command("list")
@click.option(
    "--input", "input_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Override input directory.",
)
@click.option(
    "--top", "top_n",
    type=int,
    default=0,
    help="Show only the top N clips (0 = all).",
)
@click.option(
    "--profile", "profile_name",
    type=click.Choice(_PROFILE_NAMES, case_sensitive=False),
    default=None,
    help="Output profile (for orientation sorting).",
)
@click.pass_context
def list_clips(
    ctx: click.Context,
    input_dir: Path | None,
    top_n: int,
    profile_name: str | None,
) -> None:
    """Show a ranked table of clips with composite scores."""
    config: Config = ctx.obj["config"]
    if input_dir:
        config = _override_input(config, input_dir)
    if profile_name:
        config = resolve_output_profile(config, profile_name)

    clips = analyze_only(config)
    if top_n > 0:
        clips = clips[:top_n]

    click.echo(
        f"\n {'#':>3}  {'Score':>6}  {'Dur':>6}  {'Resolution':>11}  "
        f"{'FPS':>5}  {'Orient':>10}  {'Source':>14}  {'Tags':<20}  {'File'}"
    )
    click.echo(
        f" {'—'*3}  {'—'*6}  {'—'*6}  {'—'*11}  {'—'*5}  "
        f"{'—'*10}  {'—'*14}  {'—'*20}  {'—'*30}"
    )

    for i, clip in enumerate(clips, 1):
        tags_str = ", ".join(clip.tags) if clip.tags else "—"
        source = clip.source_profile or "unknown"
        click.echo(
            f" {i:>3}  {clip.composite_score:>6.1f}  "
            f"{clip.duration:>5.1f}s  {clip.resolution:>11}  "
            f"{clip.fps:>5.1f}  {clip.orientation:>10}  "
            f"{source:>14}  {tags_str:<20}  {clip.path.name}"
        )

    click.echo()


# ------------------------------------------------------------------
# profiles — list available output profiles
# ------------------------------------------------------------------

@cli.command("edit-multi")
@click.option("--input", "input_dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None, help="Override input directory.")
@click.option("--music", "music_track",
              type=click.Path(exists=True, path_type=Path), default=None,
              help="Music track for beat-aligned sequencing.")
@click.option("--pacing", type=click.Choice(["slow", "medium", "fast", "cinematic", "energetic"]),
              default=None)
@click.option("--mood", type=click.Choice(["luxury", "energetic", "moody", "bright", "dramatic", "neutral"]),
              default=None)
@click.option("--project-name", "project_name", type=str, default=None,
              help="Override project_name (used in filenames).")
@click.option("--client-name", "client_name", type=str, default=None,
              help="Override client_name (used in filenames).")
@click.pass_context
def edit_multi(
    ctx: click.Context,
    input_dir: Path | None,
    music_track: Path | None,
    pacing: str | None,
    mood: str | None,
    project_name: str | None,
    client_name: str | None,
) -> None:
    """Render every entry in config.outputs from a single analysis pass."""
    config: Config = ctx.obj["config"]
    if input_dir:
        config = _override_input(config, input_dir)
    config = _apply_edit_overrides(
        config, music_track=music_track, target_length=None,
        pacing=pacing, mood=mood, no_beats=False,
    )
    if project_name:
        config = _clone_with(config, project_name=project_name)
    if client_name:
        config = _clone_with(config, client_name=client_name)

    if not config.outputs:
        raise click.UsageError(
            "config.outputs is empty — add an 'outputs:' list to config.yaml "
            "(e.g. [{length: 60, aspect: 16:9}, {length: 30, aspect: 9:16}])"
        )

    click.echo(f"Rendering {len(config.outputs)} output(s) for project '{config.project_name}'")
    for i, spec in enumerate(config.outputs, 1):
        click.echo(f"  {i}. {spec}")

    results = run_all(config)
    click.echo("\nRendered files:")
    for p in results:
        click.echo(f"  {p}")


@cli.command("export")
@click.option("--edl-file", "edl_file", type=click.Path(exists=True, path_type=Path),
              default=None, help="Path to edl.json (default: <output_dir>/edl.json).")
@click.option("--output-dir", "out_dir", type=click.Path(path_type=Path),
              default=None, help="Where to write the FCPXML/EDL/OTIO files.")
@click.option("--basename", type=str, default="edit",
              help="Filename stem for the exports.")
@click.pass_context
def export_cmd(
    ctx: click.Context,
    edl_file: Path | None,
    out_dir: Path | None,
    basename: str,
) -> None:
    """Convert an existing edl.json to FCPXML + CMX EDL (+ OTIO if installed)."""
    from src.export.timeline_export import export_all
    from src.models.edl import EDL, EDLEntry
    import json as _json

    config: Config = ctx.obj["config"]
    edl_path = edl_file or (config.output_dir / "edl.json")
    if not edl_path.exists():
        raise click.UsageError(f"No edl.json found at {edl_path} — run `edit` first.")

    with open(edl_path) as fh:
        data = _json.load(fh)

    entries = [
        EDLEntry(
            source_path=Path(e["source_path"]),
            source_in=float(e["source_in"]),
            source_out=float(e["source_out"]),
            timeline_in=float(e["timeline_in"]),
            duration=float(e["duration"]),
            transition_in=e.get("transition_in", "cut"),
            transition_out=e.get("transition_out", "cut"),
            beat_aligned=bool(e.get("beat_aligned", False)),
            shot_type=e.get("shot_type"),
            movement=e.get("movement"),
            score=float(e.get("score", 0.0)),
            notes=e.get("notes", ""),
        )
        for e in data.get("entries", [])
    ]
    edl = EDL(
        entries=entries,
        target_duration=float(data.get("target_duration", 0.0)),
        bpm=float(data.get("bpm", 0.0)),
        mood=data.get("mood", "neutral"),
        pacing=data.get("pacing", "medium"),
    )

    target_dir = out_dir or config.output_dir
    target = config.output_profile
    written = export_all(
        edl, target_dir, basename=basename,
        fps=target.fps, width=target.width, height=target.height,
        project_name=config.project_name,
    )
    click.echo(f"Wrote {len(written)} timeline file(s):")
    for p in written:
        click.echo(f"  {p}")


@cli.command("profiles")
def list_profiles() -> None:
    """Show all available output profiles."""
    click.echo(f"\n {'Name':<20}  {'Ratio':>7}  {'Resolution':>11}  {'FPS':>4}  {'Codec':<8}  {'Bitrate':>8}")
    click.echo(f" {'—'*20}  {'—'*7}  {'—'*11}  {'—'*4}  {'—'*8}  {'—'*8}")

    for p in BUILTIN_OUTPUT_PROFILES.values():
        ar = f"{p.aspect_ratio[0]}:{p.aspect_ratio[1]}"
        click.echo(
            f" {p.name:<20}  {ar:>7}  {p.resolution_str:>11}  "
            f"{p.fps:>4}  {p.codec:<8}  {p.bitrate:>8}"
        )

    click.echo()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _apply_edit_overrides(
    config: Config,
    music_track: Path | None,
    target_length: float | None,
    pacing: str | None,
    mood: str | None,
    no_beats: bool,
) -> Config:
    """Return a new Config with CLI edit-time overrides merged in."""
    from dataclasses import replace
    # dataclass is frozen with slots — use manual reconstruction via _override_input pattern.
    cfg = config
    if music_track is not None:
        cfg = _clone_with(cfg, music_track=music_track)
    if target_length is not None:
        cfg = _clone_with(cfg, target_length=float(target_length))
    if pacing is not None:
        cfg = _clone_with(cfg, pacing=pacing)
    if mood is not None:
        cfg = _clone_with(cfg, mood=mood)
    if no_beats:
        cfg = _clone_with(cfg, beat_aligned=False)
    return cfg


def _clone_with(config: Config, **overrides) -> Config:
    """Return a new Config with the given fields overridden."""
    from dataclasses import replace
    return replace(config, **overrides)


def _override_input(config: Config, input_dir: Path) -> Config:
    """Return a new Config with *input_dir* replaced."""
    from dataclasses import replace
    return replace(config, input_dir=input_dir)


def _print_clip_detail(rank: int, clip: 'Clip') -> None:
    """Print a detailed analysis block for a single clip."""
    click.echo(f"  #{rank}  {clip.path.name}")
    click.echo(f"       Duration:    {clip.duration:.2f}s")
    click.echo(f"       Resolution:  {clip.resolution}  @ {clip.fps:.1f} fps")
    click.echo(f"       Orientation: {clip.orientation}")
    click.echo(f"       Source:      {clip.source_profile or 'unknown'}")
    click.echo(f"       Composite:   {clip.composite_score:.1f} / 100")

    if clip.scores:
        click.echo(f"       Scores:")
        for name, score in clip.scores.items():
            click.echo(f"         {name:.<24} {score:.1f}")

    if clip.tags:
        click.echo(f"       Tags:        {', '.join(clip.tags)}")

    for key in ("scene_count", "avg_scene_length_s", "avg_motion_px", "rms_db", "silence_ratio"):
        if key in clip.metadata:
            click.echo(f"       {key}: {clip.metadata[key]}")

    click.echo()


def main() -> None:
    """Package entry point."""
    cli()


if __name__ == "__main__":
    main()
