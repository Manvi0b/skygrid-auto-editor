# SkyGrid Auto Editor

A modular Python video editing pipeline built for drone footage. Drop clips into a folder, run a single command, and get a polished edit — scored, trimmed, transitioned, formatted for your target platform, and rendered automatically.

## Versions

This project is tagged so you can browse its progression:

- **[v0.1.0](https://github.com/Manvi0b/skygrid-auto-editor/releases/tag/v0.1.0)** — Core pipeline. Plugin-based analyzer system, CLI (`edit` / `analyze` / `list`), FFmpeg rendering, beat-synced music mixing, dead-air trimming.
- **[v0.2.0](https://github.com/Manvi0b/skygrid-auto-editor/releases/tag/v0.2.0)** — Multi-format & multi-source support. Source profiles (DJI, GoPro, Osmo), output profiles (YouTube / Reels / TikTok / Instagram Square / 4K), aspect-ratio adaptation (crop / blur bars / letterbox), orientation auto-detection and sorting, `--profile` CLI flag.

```bash
git checkout v0.1.0    # see the original core pipeline
git checkout v0.2.0    # see multi-format support added
git checkout main      # latest
```

## What This Project Does

SkyGrid Auto Editor ingests raw video clips, runs them through a plugin-based analysis system to score quality, filters out the bad takes, trims dead air, applies transitions, adapts aspect ratios for your target platform, and renders the final video — all from the command line.

### Pipeline Flow

```
Input Clips → Ingest → Analyze → Rank/Filter → Assemble → Render → Output
```

1. **Ingest** — Scans an input folder for video files (.mp4, .mov, .avi, .mkv, .webm, .m4v, .mts), extracts metadata (duration, resolution, fps, orientation) via ffprobe, and auto-detects the source device from container tags (DJI, GoPro, etc.).
2. **Analyze** — Runs each clip through a set of analyzers that score quality on a 0–100 scale.
3. **Rank & Filter** — Sorts clips by composite score and drops anything below the configured minimum. Auto-prioritises clips matching the target aspect ratio.
4. **Assemble** — Adapts aspect ratio to the target output profile, trims leading/trailing silence, applies crossfade or fade-to-black transitions, and concatenates into a timeline respecting a max duration budget.
5. **Render** — Re-encodes the final video via FFmpeg using the output profile's codec, bitrate, and resolution. Optionally mixes in a background music track.

## Built-in Analyzers

| Analyzer | What It Measures | How It Scores |
|---|---|---|
| **Shake Detector** | Camera stability via Lucas-Kanade optical flow. Tracks feature points frame-to-frame and measures displacement magnitudes. | Low motion = high score. Penalises jerk (high variance). Tags clips `"shaky"` (<40) or `"stable"` (≥85). |
| **Scene Detector** | Visual variety via PySceneDetect content-aware scene boundaries. | Bell-curve scoring around ~8 scene changes/minute. Tags `"static"` or `"chaotic"`. Stores scene count, average scene length, and boundary timestamps. |
| **Audio Analyzer** | Audio presence, RMS/peak volume in dB, and silence segments. | Peaks at -20 dB RMS. Penalises excessive silence. Tags `"no_audio"`, `"silent"`, `"quiet"`, or `"loud"`. Detects individual silence segments ≥0.5s. |

## Source & Output Profiles (v0.2.0+)

### Source Profiles

Auto-detected from container metadata (DJI, GoPro, and Osmo cameras embed their model name in the MP4 container):

| Profile | Device Type | Orientation | Gimbal | Known Artifacts |
|---|---|---|---|---|
| `dji_mini3pro`, `dji_mini4pro`, `dji_air3`, `dji_mavic3` | drone | horizontal | ✓ | prop_shadow, jello |
| `osmo_pocket3` | gimbal | mixed | ✓ | — |
| `osmo_action5` | handheld | horizontal | ✗ | rolling_shutter |
| `gopro_hero12` | handheld | horizontal | ✗ | fisheye, rolling_shutter |
| `generic` | generic | horizontal | ✗ | — |

### Output Profiles

| Profile | Aspect | Resolution | Bitrate | Target Platforms |
|---|---|---|---|---|
| `youtube` | 16:9 | 1920×1080 | 20M | YouTube, web |
| `youtube_4k` | 16:9 | 3840×2160 | 45M | YouTube 4K |
| `reels` | 9:16 | 1080×1920 | 15M | Instagram Reels |
| `tiktok` | 9:16 | 1080×1920 | 15M | TikTok |
| `instagram_square` | 1:1 | 1080×1080 | 12M | Instagram square |
| `twitter` | 16:9 | 1280×720 | 10M | Twitter/X |

### Aspect Ratio Adaptation

When clip orientation doesn't match the target output, three modes are available:

- **`crop`** — Centre-crop to fill the target exactly.
- **`blur_bars`** — Scale clip to fit; fill sides/top with a heavily blurred version of itself (Instagram-style).
- **`letterbox`** — Scale clip to fit; pad with black bars.

## Plugin System

Any `.py` file dropped into the `plugins/` directory is auto-discovered at runtime:

```python
from src.analyzers.base import BaseAnalyzer
from src.models.clip import Clip

class MyCustomAnalyzer(BaseAnalyzer):
    @property
    def name(self) -> str:
        return "my_custom"

    def analyze(self, clip: Clip) -> Clip:
        score = do_something(clip.path)
        return clip.with_score(self.name, score)
```

Then enable it in `config.yaml`:

```yaml
pipeline:
  enabled_analyzers:
    - "shake_detector"
    - "scene_detector"
    - "audio_analyzer"
    - "my_custom"
```

## Project Structure

```
skygrid-auto-editor/
├── src/
│   ├── main.py              # CLI — edit, analyze, list, profiles
│   ├── pipeline.py          # Orchestration + plugin discovery
│   ├── config.py            # Config loader with output/source profile parsing
│   ├── models/
│   │   ├── clip.py          # Immutable Clip (incl. orientation, source_profile)
│   │   ├── source.py        # SourceProfile + built-in device profiles
│   │   └── output_profile.py # OutputProfile + built-in platform targets
│   ├── ingest/
│   │   └── loader.py        # ffprobe metadata, source detection, orientation sort
│   ├── analyzers/
│   │   ├── base.py          # Abstract BaseAnalyzer
│   │   ├── shake_detector.py
│   │   ├── scene_detector.py
│   │   └── audio_analyzer.py
│   ├── assembler/
│   │   └── editor.py        # Aspect adaptation, dead-air trim, transitions
│   ├── audio/
│   │   └── music_sync.py    # Beat detection, BPM, cut-to-beat snapping
│   ├── export/
│   │   └── renderer.py      # FFmpeg wrapper + music mixing
│   └── utils/
│       └── ffmpeg_helpers.py # ffprobe/ffmpeg + orientation + source detection
├── plugins/                  # Drop custom analyzers here
├── input/                    # Place source clips here
├── output/                   # Rendered output
├── config.yaml               # All pipeline settings
└── requirements.txt
```

## Installation

Requires Python 3.11+ and FFmpeg installed on your system.

```bash
git clone https://github.com/Manvi0b/skygrid-auto-editor.git
cd skygrid-auto-editor
pip install -r requirements.txt
```

## Usage

### Render for a specific platform

```bash
# YouTube landscape (default)
python -m src.main edit --input ./input --output ./output/youtube.mp4 --profile youtube

# Instagram Reels / TikTok (9:16 portrait with blur bars)
python -m src.main edit --input ./input --output ./output/reel.mp4 --profile reels

# Instagram square
python -m src.main edit --input ./input --output ./output/square.mp4 --profile instagram_square

# 4K YouTube
python -m src.main edit --input ./input --output ./output/4k.mp4 --profile youtube_4k
```

### Analyze clips without rendering

```bash
python -m src.main analyze --input ./input
```

### List ranked clips

```bash
python -m src.main list --input ./input
python -m src.main list --input ./input --top 5 --profile reels
```

Output now includes orientation and detected source:

```
   #   Score     Dur   Resolution    FPS      Orient          Source  Tags             File
 ———  ——————  ——————  ———————————  —————  ——————————  ——————————————  ———————————————  ———————————
   1    87.3   12.4s    3840x2160   30.0  horizontal    dji_mini3pro  stable           DJI_0042.MP4
```

### List available output profiles

```bash
python -m src.main profiles
```

### Global options

```bash
python -m src.main --help
python -m src.main -v edit ...                       # Debug logging
python -m src.main --config custom.yaml edit ...    # Custom config
```

## Configuration

All settings live in `config.yaml`. Key sections:

```yaml
pipeline:
  max_duration: 120          # Max output length in seconds
  min_clip_score: 30         # Minimum score (0–100) to include a clip
  sort_by: "score"

transitions:
  style: "crossfade"         # "cut", "crossfade", or "fade_black"
  duration: 0.5

audio:
  music_track: null           # Path to background music
  original_audio_volume: 0.7
  music_volume: 0.3

source_profiles:
  dji_mini3pro:
    device_type: drone
    default_orientation: horizontal
    has_gimbal: true
    typical_artifacts: [prop_shadow, jello]

output_profiles:
  reels:
    aspect_ratio: [9, 16]
    resolution: [1080, 1920]
    fps: 30
    codec: libx264
    bitrate: 15M
    aspect_ratio_mode: blur_bars

default_output_profile: youtube
aspect_ratio_mode: blur_bars    # "crop", "blur_bars", or "letterbox"
```

## Dependencies

- **moviepy** — Video clip loading, trimming, transitions, concatenation
- **opencv-python** — Optical flow, frame analysis, Gaussian blur for bar fills
- **scenedetect** — Content-aware scene boundary detection
- **librosa** — Audio analysis, beat detection, BPM estimation
- **click** — CLI framework
- **pyyaml** — Configuration loading
- **FFmpeg** (system) — Video probing, audio extraction, final rendering

## Design Decisions

- **Immutable Clip dataclass** — Frozen with `with_*()` methods that return new instances. Analyzers never mutate data.
- **0–100 scoring** — All analyzers score on the same scale for intuitive comparison and filtering.
- **Config-driven** — Everything from enabled analyzers to codec settings is controlled via YAML, no code changes needed.
- **Profile-driven output** — One flag (`--profile reels`) adjusts resolution, aspect ratio, codec, bitrate, and how clips get adapted to fit.
- **Graceful failure** — If an analyzer crashes on a clip, it's logged and skipped; the pipeline continues.
- **Two-pass output** — Assembly creates an intermediate file, then the renderer does a final encode pass with scaling, padding, and optional music mixing.
