"""Tests for Auto Hype Clip scoring, peak detection, and live ingestion."""

from __future__ import annotations

import numpy as np

from app.config import Settings
from app.pipeline.types import CutBoundary, Transcript, Word
from app.services.audio import EnergyProfile
from app.services.hype import (
    ChatMessage,
    WindowScore,
    _parse_time_text,
    _score_audio_loudness,
    _score_chat_activity,
    _score_speech_excitement,
    find_top_peaks,
    snap_boundaries_to_peak,
)
from app.services.ingest_live import RollingBufferRecorder, check_required_tools


def test_check_required_tools_on_current_system() -> None:
    settings = Settings()
    ready, error_msg = check_required_tools(settings)
    assert ready is True
    assert error_msg is None


def test_parse_time_text() -> None:
    assert _parse_time_text("01:23:45") == 5025.0
    assert _parse_time_text("02:30") == 150.0
    assert _parse_time_text("45") == 45.0
    assert _parse_time_text("invalid") is None


def test_score_audio_loudness_detects_spikes() -> None:
    # 20 seconds of audio with a huge spike at 10-15s
    rms_values = [0.05] * 10 + [0.85] * 5 + [0.05] * 5
    energy = EnergyProfile(
        window_seconds=1.0,
        rms=rms_values,
        duration=20.0,
    )

    window_times = [0.0, 5.0, 10.0, 15.0]
    scores = _score_audio_loudness(energy, window_times, window_size=5.0)
    assert len(scores) == len(window_times)
    # Peak should occur at 10.0s window
    peak_idx = int(np.argmax(scores))
    assert window_times[peak_idx] == 10.0
    assert scores[peak_idx] > scores[0]


def test_score_chat_activity_detects_hype_burst() -> None:
    messages = [
        ChatMessage(time_offset=1.0, text="hello", is_hype=False, is_caps=False),
        ChatMessage(time_offset=2.0, text="hey", is_hype=False, is_caps=False),
        # Burst of hype at 10-14s
        ChatMessage(time_offset=10.5, text="LET'S GOOO 🔥🔥🔥", is_hype=True, is_caps=True),
        ChatMessage(time_offset=11.0, text="POG POGGERS", is_hype=True, is_caps=True),
        ChatMessage(time_offset=11.5, text="🤯 WHAT A SHOT", is_hype=True, is_caps=True),
        ChatMessage(time_offset=12.0, text="W CLUTCH", is_hype=True, is_caps=True),
        ChatMessage(time_offset=13.0, text="GG", is_hype=True, is_caps=False),
    ]

    window_times = [0.0, 5.0, 10.0, 15.0]
    scores = _score_chat_activity(messages, window_times, window_size=5.0)
    peak_idx = int(np.argmax(scores))
    assert window_times[peak_idx] == 10.0
    assert scores[peak_idx] > 0.8


def test_score_speech_excitement_detects_fast_speech_and_exclamations() -> None:
    words = [
        Word(start=0.5, end=1.0, text="normal"),
        Word(start=1.2, end=1.8, text="speech"),
        # Intense shouty speech at 10-14s
        Word(start=10.1, end=10.3, text="OH!"),
        Word(start=10.4, end=10.6, text="MY!"),
        Word(start=10.7, end=10.9, text="GOD!"),
        Word(start=11.0, end=11.2, text="LOOK!"),
        Word(start=11.3, end=11.5, text="AT!"),
        Word(start=11.6, end=11.8, text="THAT!"),
        Word(start=12.0, end=12.3, text="INSANE!"),
    ]
    transcript = Transcript(words=words, duration=20.0)

    window_times = [0.0, 5.0, 10.0, 15.0]
    scores = _score_speech_excitement(transcript, window_times, window_size=5.0)
    peak_idx = int(np.argmax(scores))
    assert window_times[peak_idx] == 10.0


def test_snap_boundaries_centers_on_peak_and_respects_limits() -> None:
    boundaries = [
        CutBoundary(time=0.0, kind="sentence", strength=1.0),
        CutBoundary(time=18.5, kind="sentence", strength=0.9),
        CutBoundary(time=32.0, kind="pause", strength=0.8),
        CutBoundary(time=68.2, kind="sentence", strength=0.95),
        CutBoundary(time=95.0, kind="sentence", strength=1.0),
    ]

    # Peak at 45.0s in 120s video
    start, end = snap_boundaries_to_peak(
        peak_time=45.0,
        duration=120.0,
        boundaries=boundaries,
        min_clip=30.0,
        max_clip=60.0,
    )

    assert 30.0 <= (end - start) <= 60.0
    # Peak at 45 should be inside the clip interval
    assert start < 45.0 < end
    # Snapped to existing boundaries or safe offset
    assert start in {32.0, 18.5} or start <= 35.0


def test_find_top_peaks_respects_separation() -> None:
    window_scores = [
        WindowScore(time=0.0, audio=0.1, chat=0.1, speech=0.1, total=0.1),
        WindowScore(time=20.0, audio=0.9, chat=0.9, speech=0.9, total=0.9),  # Peak 1
        WindowScore(time=25.0, audio=0.85, chat=0.85, speech=0.85, total=0.85),  # Close to Peak 1
        WindowScore(time=80.0, audio=0.8, chat=0.8, speech=0.8, total=0.8),  # Peak 2 (> 45s away)
    ]

    top_peaks = find_top_peaks(window_scores, top_n=2, min_separation_seconds=45.0)
    assert len(top_peaks) == 2
    assert top_peaks[0] == 1  # 20.0s
    assert top_peaks[1] == 3  # 80.0s


def test_rolling_buffer_cleanup(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path)
    recorder = RollingBufferRecorder(
        url="https://youtube.com/dummy",
        buffer_dir=tmp_path,
        settings=settings,
        buffer_minutes=1,  # 60s
        segment_seconds=10,  # 6 segments max
    )

    # Create 10 dummy segments
    for i in range(10):
        seg = recorder.segments_dir / f"seg_{i:05d}.mp4"
        seg.write_bytes(b"0" * 60_000)

    assert len(recorder.get_existing_segments()) == 10
    recorder.cleanup_old_segments()
    # Should enforce max_segments = 6
    remaining = recorder.get_existing_segments()
    assert len(remaining) == 6
    assert remaining[0].name == "seg_00004.mp4"
