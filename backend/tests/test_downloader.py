from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.core.errors import URLValidationError
from app.services.downloader import (
    _auth_required_message,
    _base_command,
    _classify_error,
    _cookies_configured,
)


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(data_dir=tmp_path, **overrides)


def test_cookies_file_is_passed_to_ytdlp(tmp_path: Path) -> None:
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    command = _base_command(_settings(tmp_path, ytdlp_cookies_file=str(cookies)))
    assert "--cookies" in command
    assert str(cookies) in command


def test_missing_cookies_file_fails_fast(tmp_path: Path) -> None:
    settings = _settings(tmp_path, ytdlp_cookies_file=str(tmp_path / "missing-cookies.txt"))
    with pytest.raises(URLValidationError):
        _base_command(settings)


def test_drop_in_data_cookies_file_is_used(tmp_path: Path) -> None:
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    settings = _settings(tmp_path)
    assert _cookies_configured(settings)
    command = _base_command(settings)
    assert "--cookies" in command
    assert str(cookies) in command


def test_browser_cookies_are_passed_to_ytdlp(tmp_path: Path) -> None:
    settings = _settings(tmp_path, ytdlp_cookies_from_browser="chrome")
    command = _base_command(settings)
    assert "--cookies-from-browser" in command
    assert "chrome" in command
    assert _cookies_configured(settings)


def test_browser_profile_and_keyring_specs_are_accepted(tmp_path: Path) -> None:
    for spec in ("firefox:default-release", "chrome+keyring"):
        command = _base_command(_settings(tmp_path, ytdlp_cookies_from_browser=spec))
        assert spec in command


def test_unknown_browser_is_rejected(tmp_path: Path) -> None:
    settings = _settings(tmp_path, ytdlp_cookies_from_browser="netscape")
    with pytest.raises(URLValidationError):
        _base_command(settings)


def test_auth_error_explains_how_to_supply_cookies() -> None:
    error = _classify_error(
        "ERROR: Sign in to confirm your age. This video may be inappropriate for some users.",
        action="fetch the video details",
        cookies_configured=False,
    )
    assert error.code == "auth_required"
    assert "CUTCLIPPER_YTDLP_COOKIES_FROM_BROWSER" in error.message
    assert "cookies.txt" in error.message
    assert "upload" in error.message.lower()


def test_auth_error_with_cookies_asks_the_user_to_refresh_them() -> None:
    message = _auth_required_message(cookies_configured=True)
    assert "Refresh" in message
    assert "cookies" in message.lower()


def test_bot_check_is_classified_as_authentication() -> None:
    error = _classify_error(
        "ERROR: Sign in to confirm you're not a bot. Use --cookies for the authentication.",
        action="download the video",
    )
    assert error.code == "auth_required"
    assert error.retryable is False


def test_missing_youtube_formats_explains_the_solver_dependency() -> None:
    error = _classify_error(
        "ERROR: [youtube] abc: No video formats found! Signature solving failed.",
        action="download the video",
    )
    assert error.code == "formats_unavailable"
    assert "yt-dlp-ejs" in error.message
    assert "Node.js" in error.message


def test_metadata_supports_live_streams(tmp_path: Path) -> None:
    from app.services.downloader import _metadata_from_info

    settings = _settings(tmp_path)
    info = {
        "id": "live123",
        "title": "Exciting Live Stream",
        "is_live": True,
        "live_status": "is_live",
        "uploader": "StreamerOne",
        "width": 1920,
        "height": 1080,
        "fps": 60,
        "acodec": "mp4a.40.2",
    }
    meta = _metadata_from_info(
        info, normalized_url="https://youtube.com/watch?v=live123", settings=settings
    )
    assert meta.is_live is True
    assert meta.title == "Exciting Live Stream"
    assert meta.uploader == "StreamerOne"
    assert meta.duration == 0.0


def test_upcoming_live_stream_raises_helpful_error(tmp_path: Path) -> None:
    from app.core.errors import MediaFetchError
    from app.services.downloader import _metadata_from_info

    settings = _settings(tmp_path)
    info = {
        "id": "upcoming123",
        "title": "Scheduled Premier",
        "live_status": "is_upcoming",
    }
    with pytest.raises(MediaFetchError) as exc_info:
        _metadata_from_info(
            info, normalized_url="https://youtube.com/watch?v=upcoming123", settings=settings
        )
    assert exc_info.value.code == "stream_upcoming"
    assert "scheduled for the future" in exc_info.value.message
