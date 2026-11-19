"""Live feed ingestion using yt-dlp and FFmpeg rolling segment buffer."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event

from app.config import Settings
from app.core.errors import JobCancelledError, MediaFetchError, ProcessingError
from app.services.downloader import _base_command, _classify_error
from app.services.media import get_media_tools
from app.services.process import run_process

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def check_required_tools(settings: Settings) -> tuple[bool, str | None]:
    """Verify that FFmpeg and yt-dlp are installed and callable.

    Returns (is_ready, error_instructions).
    """
    try:
        tools = get_media_tools(settings)
        if not tools.ffmpeg or not Path(tools.ffmpeg).exists():
            return (
                False,
                "FFmpeg is missing. Install it with: brew install ffmpeg (macOS) or "
                "sudo apt-get install -y ffmpeg (Ubuntu/Debian).",
            )
    except Exception as exc:
        return (
            False,
            f"FFmpeg check failed: {exc}. Install it with: brew install ffmpeg (macOS) or "
            "sudo apt-get install -y ffmpeg (Ubuntu/Debian).",
        )

    # Check yt-dlp via python module
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        return (
            False,
            "yt-dlp is missing. Install it with: pip install yt-dlp.",
        )

    return True, None


@dataclass(slots=True)
class StreamMetadata:
    url: str
    title: str
    is_live: bool
    duration: float | None = None
    direct_stream_url: str | None = None
    channel: str | None = None
    view_count: int | None = None
    chat_available: bool = True
    warnings: list[str] = field(default_factory=list)


async def probe_stream_or_vod(
    url: str,
    settings: Settings,
    *,
    cancel_event: Event | None = None,
) -> StreamMetadata:
    """Fetch metadata and stream URLs using yt-dlp.

    Handles live streams, offline streams, private streams, and VODs.
    """
    ready, error_msg = check_required_tools(settings)
    if not ready and error_msg:
        raise ProcessingError(error_msg, code="missing_tools")

    cmd = [
        *_base_command(settings),
        "--dump-json",
        "--skip-download",
        url,
    ]

    result = await run_process(cmd, cancel_event=cancel_event)
    if result.returncode != 0:
        err = _classify_error(result.stderr, action="probe stream")
        # Check for specific live stream offline notices
        lower_err = result.stderr.lower()
        if "live stream will begin" in lower_err or "offline" in lower_err:
            raise MediaFetchError(
                "This live stream is currently offline or has not started yet.",
                code="stream_offline",
                retryable=True,
            )
        raise err

    stdout = result.stdout.strip()
    if not stdout:
        raise MediaFetchError("Could not retrieve stream information.", code="stream_probe_failed")

    try:
        data = json.loads(stdout.splitlines()[0])
    except (json.JSONDecodeError, IndexError) as exc:
        raise MediaFetchError(
            "Invalid stream metadata returned by yt-dlp.", code="invalid_metadata"
        ) from exc

    is_live = bool(data.get("is_live") or data.get("live_status") == "is_live")
    is_upcoming = bool(data.get("live_status") == "is_upcoming")

    if is_upcoming:
        raise MediaFetchError(
            "This stream is scheduled for the future and is not live yet.",
            code="stream_upcoming",
            retryable=True,
        )

    title = data.get("title") or "live_stream"
    duration = float(data.get("duration") or 0.0) if not is_live else None
    direct_url = data.get("url")

    # If stream format is HLS/m3u8, direct_url will be the manifest
    if not direct_url:
        formats = data.get("formats") or []
        for f in reversed(formats):
            if f.get("url") and ("m3u8" in f.get("url", "") or f.get("vcodec") != "none"):
                direct_url = f.get("url")
                break

    return StreamMetadata(
        url=url,
        title=title,
        is_live=is_live,
        duration=duration,
        direct_stream_url=direct_url,
        channel=data.get("uploader") or data.get("channel"),
        view_count=data.get("view_count"),
        chat_available=not bool(data.get("is_chat_disabled", False)),
    )


class RollingBufferRecorder:
    """Records a live stream into 10-second segments, keeping a rolling buffer."""

    def __init__(
        self,
        url: str,
        buffer_dir: Path,
        settings: Settings,
        *,
        buffer_minutes: int | None = None,
        segment_seconds: int | None = None,
        cancel_event: Event | None = None,
    ) -> None:
        self.url = url
        self.buffer_dir = buffer_dir
        self.settings = settings
        self.buffer_minutes = buffer_minutes or settings.hype_buffer_minutes
        self.segment_seconds = segment_seconds or settings.hype_segment_seconds
        self.cancel_event = cancel_event or Event()
        self.max_segments = max(6, int((self.buffer_minutes * 60) // self.segment_seconds))
        self.segments_dir = self.buffer_dir / "segments"
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        self._process: asyncio.subprocess.Process | None = None

    def get_existing_segments(self) -> list[Path]:
        """Return sorted list of completed segment files."""
        if not self.segments_dir.exists():
            return []
        segments = sorted(
            [
                p
                for p in self.segments_dir.glob("seg_*.mp4")
                if p.is_file() and p.stat().st_size > 50_000
            ],
            key=lambda p: p.name,
        )
        return segments

    def cleanup_old_segments(self) -> None:
        """Enforce circular rolling buffer limit."""
        segments = self.get_existing_segments()
        if len(segments) > self.max_segments:
            overflow = len(segments) - self.max_segments
            for seg in segments[:overflow]:
                with contextlib.suppress(OSError):
                    seg.unlink(missing_ok=True)

    async def record_for_duration(
        self,
        duration_seconds: float,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> Path:
        """Run rolling recording for a set duration, then merge segments into one file."""
        tools = get_media_tools(self.settings)
        stream_meta = await probe_stream_or_vod(
            self.url, self.settings, cancel_event=self.cancel_event
        )

        # For VODs or streams that are already complete, download or slice directly
        if not stream_meta.is_live:
            return await self._download_vod(on_progress=on_progress)

        segment_pattern = str(self.segments_dir / "seg_%05d.mp4")

        format_selector = (
            "bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a]/"
            "bv*[height<=1080][vcodec^=avc1]+ba/"
            "bv*[height<=1080]+ba/"
            "b[height<=1080]/b"
        )
        yt_cmd = [
            *_base_command(self.settings),
            "-f",
            format_selector,
            "--no-warnings",
            "--quiet",
            "-o",
            "-",
            self.url,
        ]
        ffmpeg_cmd = [
            tools.ffmpeg,
            "-y",
            "-i",
            "pipe:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-f",
            "segment",
            "-segment_time",
            str(self.segment_seconds),
            "-segment_format",
            "mp4",
            "-reset_timestamps",
            "1",
            segment_pattern,
        ]

        logger.info(
            "starting live stream recording",
            extra={"url": self.url, "target_seconds": duration_seconds},
        )

        r, w = os.pipe()
        try:
            proc_ytdlp = await asyncio.create_subprocess_exec(
                *yt_cmd,
                stdout=w,
                stderr=asyncio.subprocess.PIPE,
            )
        finally:
            os.close(w)

        try:
            proc_ffmpeg = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdin=r,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        finally:
            os.close(r)

        self._process = proc_ffmpeg

        start_time = time.monotonic()
        try:
            while time.monotonic() - start_time < duration_seconds:
                if self.cancel_event.is_set():
                    proc_ytdlp.terminate()
                    proc_ffmpeg.terminate()
                    raise JobCancelledError()

                elapsed = time.monotonic() - start_time
                fraction = min(0.98, elapsed / max(1.0, duration_seconds))
                if on_progress:
                    on_progress(
                        fraction,
                        f"Recording live feed ({int(elapsed)}s / {int(duration_seconds)}s)",
                    )

                self.cleanup_old_segments()
                await asyncio.sleep(2.0)

                # Check if yt-dlp exited early
                if proc_ytdlp.returncode is not None:
                    break
        finally:
            if proc_ytdlp.returncode is None:
                proc_ytdlp.terminate()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(proc_ytdlp.wait(), timeout=3.0)
            if proc_ffmpeg.returncode is None:
                proc_ffmpeg.terminate()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(proc_ffmpeg.wait(), timeout=3.0)

        segments = self.get_existing_segments()
        if not segments:
            stderr_bytes = await proc_ytdlp.stderr.read() if proc_ytdlp.stderr else b""
            stderr_str = stderr_bytes.decode(errors="replace")
            if stderr_str:
                logger.warning("yt-dlp live recording failed", extra={"stderr": stderr_str})
                raise _classify_error(stderr_str, action="record live stream")
            raise ProcessingError(
                "No live stream segments were recorded. The stream may be offline or unreachable.",
                code="no_segments_recorded",
            )

        return await self.merge_rolling_buffer()

    async def merge_rolling_buffer(self) -> Path:
        """Merge all current buffer segments into a single cohesive source video."""
        segments = self.get_existing_segments()
        if not segments:
            raise ProcessingError(
                "No live stream segments were recorded. The stream may be offline or unreachable.",
                code="no_segments_recorded",
            )

        concat_list = self.buffer_dir / "concat_list.txt"
        with open(concat_list, "w", encoding="utf-8") as f:
            for seg in segments:
                safe_path = str(seg.resolve()).replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")

        merged_path = self.buffer_dir / "buffer_merged.mp4"
        tools = get_media_tools(self.settings)

        cmd = [
            tools.ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(merged_path),
        ]
        result = await run_process(cmd, cancel_event=self.cancel_event)
        if result.returncode != 0 or not merged_path.exists():
            raise ProcessingError(
                f"Failed to concatenate live stream buffer: {result.stderr[-400:]}",
                code="concat_failed",
            )

        return merged_path

    async def _download_vod(
        self,
        on_progress: ProgressCallback | None = None,
    ) -> Path:
        """For VOD testing, download video file using yt-dlp."""
        vod_path = self.buffer_dir / "vod_source.mp4"
        if vod_path.exists() and vod_path.stat().st_size > 1_000_000:
            return vod_path

        format_selector = (
            "bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a]/"
            "bv*[height<=1080][vcodec^=avc1]+ba/"
            "bv*[height<=1080]+ba/"
            "b[height<=1080]/b"
        )
        cmd = [
            *_base_command(self.settings),
            "-f",
            format_selector,
            "--merge-output-format",
            "mp4",
            "-o",
            str(vod_path),
            self.url,
        ]

        if on_progress:
            on_progress(0.1, "Downloading video stream for hype analysis")

        result = await run_process(cmd, cancel_event=self.cancel_event)
        if result.returncode != 0 or not vod_path.exists():
            raise _classify_error(result.stderr, action="download VOD")

        return vod_path
