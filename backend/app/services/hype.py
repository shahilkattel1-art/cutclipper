"""Hype detection and moment scoring combining audio, chat, and transcript."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import numpy as np

from app.config import Settings
from app.pipeline.selection import build_boundaries
from app.pipeline.types import CutBoundary, Transcript
from app.schemas import HypeMoment, HypeScoreBreakdown
from app.services.audio import EnergyProfile, analyze_energy, extract_audio
from app.services.transcriber import transcribe_audio

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]

_HYPE_WORDS_RE = re.compile(
    r"\b(let'?s\s*go|pog|poggers|pogu|w|gg|omg|holy|sheesh|wtf|no\s*way|hype|clutch|lmao|lol|insane|crazy|goat)\b",
    re.IGNORECASE,
)
_HYPE_EMOJIS_RE = re.compile(r"[🔥😂🤯😱👏💀🚀⚡🐐💯🎉🥳]")


@dataclass(slots=True)
class ChatMessage:
    time_offset: float
    text: str
    is_hype: bool
    is_caps: bool


@dataclass(slots=True)
class WindowScore:
    time: float
    audio: float
    chat: float
    speech: float
    total: float


def fetch_chat_messages(
    url: str,
    duration: float,
    *,
    cancel_event: Event | None = None,
) -> tuple[list[ChatMessage], bool, list[str]]:
    """Fetch live or VOD chat messages using chat-downloader.

    Returns (messages, chat_available, warnings).
    """
    messages: list[ChatMessage] = []
    warnings: list[str] = []

    try:
        from chat_downloader import ChatDownloader  # type: ignore[import-untyped]
    except ImportError:
        warnings.append("chat-downloader is not installed. Chat scoring will be skipped.")
        return [], False, warnings

    try:
        downloader = ChatDownloader()
        chat_iter = downloader.get_chat(url)
        for item in chat_iter:
            if cancel_event and cancel_event.is_set():
                break
            # time_in_seconds or time_offset
            time_offset = item.get("time_in_seconds")
            if time_offset is None:
                offset_text = item.get("time_text")
                if offset_text:
                    time_offset = _parse_time_text(offset_text)
            if time_offset is None:
                continue

            time_sec = float(time_offset)
            if time_sec > duration + 30.0:
                break
            if time_sec < 0:
                continue

            text = item.get("message") or ""
            if not text:
                continue

            has_hype_word = bool(_HYPE_WORDS_RE.search(text))
            has_hype_emoji = bool(_HYPE_EMOJIS_RE.search(text))
            is_caps = (
                len(text) >= 4 and sum(1 for c in text if c.isupper()) / max(1, len(text)) > 0.65
            )

            messages.append(
                ChatMessage(
                    time_offset=round(time_sec, 2),
                    text=text,
                    is_hype=has_hype_word or has_hype_emoji,
                    is_caps=is_caps,
                )
            )
            # Limit total chat messages to avoid excessive memory
            if len(messages) >= 50_000:
                break
    except Exception as exc:
        logger.warning("could not fetch live chat: %s", exc)
        warnings.append(f"Live chat could not be retrieved: {exc}. Using audio and speech scoring.")
        return [], False, warnings

    return messages, True, warnings


def _parse_time_text(text: str) -> float | None:
    parts = text.strip().lstrip("-").split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 1:
            return float(parts[0])
    except (ValueError, IndexError):
        return None
    return None


def _score_audio_loudness(
    energy: EnergyProfile,
    window_times: list[float],
    window_size: float,
) -> np.ndarray:
    """Compute audio loudness spikes relative to local rolling average."""
    if not energy.rms:
        return np.zeros(len(window_times), dtype=np.float32)

    raw_scores = []
    for t in window_times:
        rms = energy.mean_energy(t, t + window_size)
        raw_scores.append(rms)

    arr = np.array(raw_scores, dtype=np.float32)
    if len(arr) < 2 or np.max(arr) == 0:
        return np.zeros(len(window_times), dtype=np.float32)

    # Compute rolling average over ~30s (or up to len(arr))
    rolling_radius = 3
    k_size = min(len(arr), rolling_radius * 2 + 1)
    if k_size % 2 == 0:
        k_size = max(1, k_size - 1)
    kernel = np.ones(k_size, dtype=np.float32) / float(k_size)
    rolling_mean = np.convolve(arr, kernel, mode="same")
    spikes = np.maximum(0.0, arr - rolling_mean)

    # Normalize to [0, 1]
    max_spike = np.max(spikes)
    if max_spike > 1e-5:
        return spikes / max_spike
    return arr / max(1e-5, np.max(arr))


def _score_chat_activity(
    messages: list[ChatMessage],
    window_times: list[float],
    window_size: float,
) -> np.ndarray:
    """Compute chat velocity, hype emojis/words, and caps spam density."""
    if not messages:
        return np.zeros(len(window_times), dtype=np.float32)

    offsets = np.array([m.time_offset for m in messages], dtype=np.float32)
    hype_weights = np.array(
        [2.5 if m.is_hype else (1.5 if m.is_caps else 1.0) for m in messages], dtype=np.float32
    )

    scores = []
    for t in window_times:
        mask = (offsets >= t) & (offsets < t + window_size)
        if not np.any(mask):
            scores.append(0.0)
        else:
            weighted_count = np.sum(hype_weights[mask])
            scores.append(float(weighted_count))

    arr = np.array(scores, dtype=np.float32)
    max_val = np.max(arr)
    if max_val > 1e-5:
        return arr / max_val
    return np.zeros(len(window_times), dtype=np.float32)


def _score_speech_excitement(
    transcript: Transcript,
    window_times: list[float],
    window_size: float,
) -> np.ndarray:
    """Compute speech rate and vocal excitement from transcript tokens."""
    if not transcript.words:
        return np.zeros(len(window_times), dtype=np.float32)

    word_starts = np.array([w.start for w in transcript.words], dtype=np.float32)
    scores = []

    for t in window_times:
        mask = (word_starts >= t) & (word_starts < t + window_size)
        sub_words = [transcript.words[i] for i, m in enumerate(mask) if m]
        if not sub_words:
            scores.append(0.0)
            continue

        # Words per second
        rate = len(sub_words) / window_size
        excitement_bonus = 0.0
        for w in sub_words:
            text = w.text
            if "!" in text or "?" in text:
                excitement_bonus += 0.4
            if text.isupper() and len(text) > 2:
                excitement_bonus += 0.3

        scores.append(rate + excitement_bonus)

    arr = np.array(scores, dtype=np.float32)
    max_val = np.max(arr)
    if max_val > 1e-5:
        return arr / max_val
    return np.zeros(len(window_times), dtype=np.float32)


def _smooth_signals(arr: np.ndarray, window_radius: int = 1) -> np.ndarray:
    """Apply light triangular smoothing."""
    if len(arr) < 3:
        return arr
    kernel = np.array([0.2, 0.6, 0.2], dtype=np.float32)
    return np.convolve(arr, kernel, mode="same")


def find_top_peaks(
    window_scores: list[WindowScore],
    top_n: int = 1,
    min_separation_seconds: float = 45.0,
) -> list[int]:
    """Return indices of top local peaks with separation constraints."""
    if not window_scores:
        return []

    totals = [ws.total for ws in window_scores]
    sorted_indices = sorted(range(len(totals)), key=lambda i: totals[i], reverse=True)

    chosen: list[int] = []
    for idx in sorted_indices:
        t = window_scores[idx].time
        if any(abs(t - window_scores[prev].time) < min_separation_seconds for prev in chosen):
            continue
        chosen.append(idx)
        if len(chosen) >= top_n:
            break

    return chosen


def snap_boundaries_to_peak(
    peak_time: float,
    duration: float,
    boundaries: list[CutBoundary],
    *,
    min_clip: float = 30.0,
    max_clip: float = 60.0,
) -> tuple[float, float]:
    """Center on the peak with ~10-15s build-up and ~10-20s after, snapped cleanly."""
    # Target 45s ideal clip length centered around peak (-15s buildup, +25s aftermath)
    target_start = max(0.0, peak_time - 15.0)
    target_end = min(duration, peak_time + 25.0)

    # If target is shorter than min_clip, expand
    if target_end - target_start < min_clip:
        deficit = min_clip - (target_end - target_start)
        target_start = max(0.0, target_start - deficit / 2)
        target_end = min(duration, target_start + min_clip)

    # Snap start to nearest prior boundary
    valid_starts = [
        b.time for b in boundaries if b.time <= peak_time - 5.0 and b.time >= target_start - 10.0
    ]
    snapped_start = (
        min(valid_starts, key=lambda t: abs(t - target_start)) if valid_starts else target_start
    )

    # Snap end to nearest subsequent boundary
    valid_ends = [
        b.time for b in boundaries if b.time >= peak_time + 5.0 and b.time <= target_end + 10.0
    ]
    snapped_end = min(valid_ends, key=lambda t: abs(t - target_end)) if valid_ends else target_end

    # Clamp duration to [min_clip, max_clip]
    clip_dur = snapped_end - snapped_start
    if clip_dur < min_clip:
        snapped_end = min(duration, snapped_start + min_clip)
    elif clip_dur > max_clip:
        snapped_end = snapped_start + max_clip

    return round(snapped_start, 3), round(snapped_end, 3)


async def analyze_hype_moments(
    source_path: Path,
    url: str,
    duration: float,
    settings: Settings,
    *,
    top_n: int = 1,
    min_clip_seconds: float = 30.0,
    max_clip_seconds: float = 60.0,
    cancel_event: Event | None = None,
    on_stage: Callable[[str, float], None] | None = None,
) -> tuple[list[HypeMoment], Transcript, EnergyProfile, list[str]]:
    """Complete multi-modal hype analysis pipeline."""
    warnings: list[str] = []
    work_dir = source_path.parent / "hype_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    if on_stage:
        on_stage("Scoring audio loudness & extracting voice", 0.15)

    # 1. Audio extraction & energy profile
    audio_path = work_dir / "audio.wav"
    await extract_audio(source_path, audio_path, cancel_event=cancel_event)
    energy = await analyze_energy(audio_path)

    # 2. Transcribe for speech excitement & sentence boundaries
    if on_stage:
        on_stage("Transcribing speech for excitement scoring", 0.35)
    transcript = await transcribe_audio(
        audio_path,
        duration,
        settings,
        cancel_event=cancel_event,
        language="auto",
    )

    # 3. Live chat activity
    if on_stage:
        on_stage("Fetching live chat activity & hype emojis", 0.55)
    messages, chat_available, chat_warnings = await asyncio.to_thread(
        fetch_chat_messages,
        url,
        duration,
        cancel_event=cancel_event,
    )
    warnings.extend(chat_warnings)

    # Adjust weights if chat was disabled
    w_audio = settings.hype_weight_audio
    w_chat = settings.hype_weight_chat if chat_available else 0.0
    w_speech = settings.hype_weight_speech
    total_w = w_audio + w_chat + w_speech
    w_audio /= total_w
    w_chat /= total_w
    w_speech /= total_w

    # 4. Generate 5-second sliding windows
    window_size = float(settings.hype_window_seconds)
    step = float(settings.hype_step_seconds)
    window_times = [float(t) for t in np.arange(0.0, max(0.1, duration - window_size), step)]
    if not window_times:
        window_times = [0.0]

    audio_scores = _score_audio_loudness(energy, window_times, window_size)
    chat_scores = _score_chat_activity(messages, window_times, window_size)
    speech_scores = _score_speech_excitement(transcript, window_times, window_size)

    total_scores = w_audio * audio_scores + w_chat * chat_scores + w_speech * speech_scores
    smoothed = _smooth_signals(total_scores)

    window_objs = [
        WindowScore(
            time=t,
            audio=float(audio_scores[i]),
            chat=float(chat_scores[i]),
            speech=float(speech_scores[i]),
            total=float(smoothed[i]),
        )
        for i, t in enumerate(window_times)
    ]

    # 5. Snap to clean sentence/silence boundaries
    if on_stage:
        on_stage("Finding peak moments & snapping boundaries", 0.75)

    boundaries = build_boundaries(
        duration=duration,
        transcript=transcript,
        scenes=[],
        energy=energy,
        min_clip=min_clip_seconds,
    )

    peak_indices = find_top_peaks(window_objs, top_n=top_n)
    if not peak_indices:
        peak_indices = [0]

    moments: list[HypeMoment] = []
    for rank, p_idx in enumerate(peak_indices, start=1):
        ws = window_objs[p_idx]
        start, end = snap_boundaries_to_peak(
            ws.time + window_size / 2,
            duration,
            boundaries,
            min_clip=min_clip_seconds,
            max_clip=max_clip_seconds,
        )
        moments.append(
            HypeMoment(
                rank=rank,
                peak_time=round(ws.time + window_size / 2, 2),
                start=start,
                end=end,
                duration=round(end - start, 2),
                score=round(ws.total * 100.0, 1),
                breakdown=HypeScoreBreakdown(
                    audio=round(ws.audio * 100.0, 1),
                    chat=round(ws.chat * 100.0, 1),
                    speech=round(ws.speech * 100.0, 1),
                    total=round(ws.total * 100.0, 1),
                ),
            )
        )

    return moments, transcript, energy, warnings
