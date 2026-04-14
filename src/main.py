"""CLI entry point for skygrid-auto-editor."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from src.config import Config, load_config, resolve_output_profile
from src.models.output_profile import BUILTIN_OUTPUT_PROFILES
from src.pipeline import analyze_only, run

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
@click.pass_context
def edit(
    ctx: click.Context,
    input_dir: Path | None,
    output_file: Path | None,
    profile_name: str | None,
) -> None:
    """Run the full editing pipeline: analyze → rank → assemble → render."""
    config: Config = ctx.obj["config"]
    if input_dir:
        config = _override_input(config, input_dir)
    if profile_name:
        config = resolve_output_profile(config, profile_name)

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

def _override_input(config: Config, input_dir: Path) -> Config:
    """Return a new Config with *input_dir* replaced."""
    return Config(
        input_dir=input_dir,
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
        resolution=config.resolution,
        fps=config.fps,
        codec=config.codec,
        bitrate=config.bitrate,
        audio_codec=config.audio_codec,
        audio_bitrate=config.audio_bitrate,
        preset=config.preset,
        output_profile=config.output_profile,
        source_profiles=config.source_profiles,
        aspect_ratio_mode=config.aspect_ratio_mode,
    )


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
