from __future__ import annotations

import json
from pathlib import Path

from app.config import Settings
from app.pipeline.renderer import _video_filters
from app.pipeline.types import ClipPlan
from app.schemas import ReframeMode, TransitionSettings
from app.services.layout_facecam import (
    build_facecam_filter_graph,
    load_webcam_preset,
    parse_webcam_region,
    save_webcam_preset,
)
from app.services.media import MediaInfo


def test_parse_webcam_region() -> None:
    # None or empty
    assert parse_webcam_region(None) is None
    assert parse_webcam_region("") is None

    # Tuple / list
    assert parse_webcam_region((100, 200, 300, 400)) == (100, 200, 300, 400)
    assert parse_webcam_region([0, 0, 1920, 1080]) == (0, 0, 1920, 1080)

    # Comma or whitespace separated strings
    assert parse_webcam_region("100, 200, 300, 400") == (100, 200, 300, 400)
    assert parse_webcam_region("100 200 300 400") == (100, 200, 300, 400)
    assert parse_webcam_region("100:200:300:400") == (100, 200, 300, 400)

    # Dict
    assert parse_webcam_region({"x": 10, "y": 20, "w": 300, "h": 200}) == (10, 20, 300, 200)

    # Invalid cases
    assert parse_webcam_region("invalid") is None
    assert parse_webcam_region([10, 20, 30]) is None  # only 3 elements
    assert parse_webcam_region((10, 20, 0, 100)) is None  # w <= 0
    assert parse_webcam_region((10, 20, 100, -5)) is None  # h <= 0
    assert parse_webcam_region((-1, 20, 100, 100)) is None  # x < 0


def test_save_and_load_webcam_preset(tmp_path: Path) -> None:
    preset_file = tmp_path / "webcam_presets.json"

    # Save a preset for 'shroud' at 1920x1080
    save_webcam_preset(
        channel="Shroud",
        region=(1400, 50, 480, 270),
        source_size=(1920, 1080),
        presets_file=preset_file,
    )

    assert preset_file.exists()
    data = json.loads(preset_file.read_text(encoding="utf-8"))
    assert "shroud" in data

    # Load exact resolution
    loaded = load_webcam_preset("shroud", (1920, 1080), preset_file)
    assert loaded == (1400, 50, 480, 270)

    # Load scaled resolution (4K: 3840x2160)
    loaded_4k = load_webcam_preset("Shroud", (3840, 2160), preset_file)
    assert loaded_4k is not None
    assert loaded_4k == (2800, 100, 960, 540)

    # Missing channel returns None
    assert load_webcam_preset("unknown_channel", (1920, 1080), preset_file) is None


def test_build_facecam_filter_graph_structure(tmp_path: Path) -> None:
    media = MediaInfo(
        duration=60.0,
        width=1920,
        height=1080,
        fps=30.0,
        has_audio=True,
        has_video=True,
    )
    settings = Settings()

    filters = build_facecam_filter_graph(
        media=media,
        output_size=(1080, 1920),
        webcam_region=(1400, 50, 480, 270),
        webcam_height=600,
        gap=0,
        divider=False,
        remove_webcam_from_gameplay=False,
        smart_center_x=None,
        caption_path=None,
        transitions=TransitionSettings(),
        settings=settings,
        fonts_dir=tmp_path / "fonts",
        work_dir=tmp_path,
        duration=30.0,
    )

    filter_str = ",".join(filters)
    # Check that split, crops, scales, and vstack are present
    assert "[0:v]split=2[in_cam][in_game]" in filter_str
    assert "scale=1080:600" in filter_str
    assert "scale=1080:1320" in filter_str
    assert "vstack=inputs=2" in filter_str


def test_build_facecam_filter_graph_divider_and_gap(tmp_path: Path) -> None:
    media = MediaInfo(
        duration=60.0,
        width=1920,
        height=1080,
        fps=30.0,
        has_audio=True,
        has_video=True,
    )
    settings = Settings()

    filters = build_facecam_filter_graph(
        media=media,
        output_size=(1080, 1920),
        webcam_region=(1400, 50, 480, 270),
        webcam_height=550,
        gap=10,
        divider=True,
        remove_webcam_from_gameplay=False,
        smart_center_x=None,
        caption_path=None,
        transitions=TransitionSettings(),
        settings=settings,
        fonts_dir=tmp_path / "fonts",
        work_dir=tmp_path,
        duration=30.0,
    )
    filter_str = ",".join(filters)

    # Divider adds drawbox
    assert "drawbox=" in filter_str
    # Gap adds color gap or pad
    assert "panel_gap" in filter_str or "pad=" in filter_str


def test_build_facecam_filter_graph_with_captions(tmp_path: Path) -> None:
    media = MediaInfo(
        duration=60.0,
        width=1920,
        height=1080,
        fps=30.0,
        has_audio=True,
        has_video=True,
    )
    settings = Settings()
    ass_path = tmp_path / "test_subtitles.ass"
    ass_path.write_text("[Script Info]\n", encoding="utf-8")

    filters = build_facecam_filter_graph(
        media=media,
        output_size=(1080, 1920),
        webcam_region=(1400, 50, 480, 270),
        webcam_height=600,
        gap=0,
        divider=False,
        remove_webcam_from_gameplay=False,
        smart_center_x=None,
        caption_path=ass_path,
        transitions=TransitionSettings(),
        settings=settings,
        fonts_dir=tmp_path / "fonts",
        work_dir=tmp_path,
        duration=30.0,
    )
    filter_str = ",".join(filters)
    assert "ass=" in filter_str


def test_build_facecam_filter_graph_remove_webcam_from_gameplay(tmp_path: Path) -> None:
    media = MediaInfo(
        duration=60.0,
        width=1920,
        height=1080,
        fps=30.0,
        has_audio=True,
        has_video=True,
    )
    settings = Settings()

    # Webcam on top right: x=1400 in 1920 width
    filters_shifted = build_facecam_filter_graph(
        media=media,
        output_size=(1080, 1920),
        webcam_region=(1400, 50, 480, 270),
        webcam_height=600,
        gap=0,
        divider=False,
        remove_webcam_from_gameplay=True,
        smart_center_x=None,
        caption_path=None,
        transitions=TransitionSettings(),
        settings=settings,
        fonts_dir=tmp_path / "fonts",
        work_dir=tmp_path,
        duration=30.0,
    )
    filter_str = ",".join(filters_shifted)
    # Gameplay panel should shift left away from webcam
    assert "[in_game]crop=" in filter_str


def test_video_filters_delegates_to_facecam(tmp_path: Path) -> None:
    media = MediaInfo(
        duration=60.0,
        width=1920,
        height=1080,
        fps=30.0,
        has_audio=True,
        has_video=True,
    )
    plan = ClipPlan(start=0.0, end=30.0, score=1.0, selection_mode="hype")

    # When webcam_region is given, _video_filters delegates to build_facecam_filter_graph
    filters = _video_filters(
        plan=plan,
        index=0,
        media=media,
        output_size=(1080, 1920),
        face_track=None,
        caption_path=None,
        transitions=TransitionSettings(),
        settings=Settings(),
        fonts_dir=tmp_path / "fonts",
        work_dir=tmp_path,
        reframe_mode=ReframeMode.CENTER,
        webcam_region=(1400, 50, 480, 270),
        webcam_height=600,
    )
    assert any("split=2" in f for f in filters)
    assert any("vstack=inputs=2" in f for f in filters)


def test_renderer_fallback_when_no_webcam(tmp_path: Path) -> None:
    # If layout_mode is FACECAM_TOP but auto-detect returns None and no manual region,
    # it must fall back to standard 9:16 reframe without crashing
    media = MediaInfo(
        duration=60.0,
        width=1920,
        height=1080,
        fps=30.0,
        has_audio=True,
        has_video=True,
    )
    plan = ClipPlan(start=0.0, end=30.0, score=1.0, selection_mode="hype")

    # When webcam_region is None (fallback), _video_filters builds standard reframe
    filters = _video_filters(
        plan=plan,
        index=0,
        media=media,
        output_size=(1080, 1920),
        face_track=None,
        caption_path=None,
        transitions=TransitionSettings(),
        settings=Settings(),
        fonts_dir=tmp_path / "fonts",
        work_dir=tmp_path,
        reframe_mode=ReframeMode.CENTER,
        webcam_region=None,
        webcam_height=600,
    )

    ",".join(filters)
    assert any("crop=" in f for f in filters)
    assert not any("split=2" in f for f in filters)
    assert not any("vstack=" in f for f in filters)
