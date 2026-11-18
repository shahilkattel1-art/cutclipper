from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.pipeline.types import Sentence, Transcript, Word
from app.schemas import VideoMetadata
from app.services.uploads import StoredUpload


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        yield c


def test_transcribe_captions_endpoint_succeeds_and_caches(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upload_id = "a" * 32
    upload_dir = tmp_path / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    video_path = upload_dir / "video.mp4"
    video_path.write_bytes(b"dummy")

    fake_record = StoredUpload(
        id=upload_id,
        path=video_path,
        thumbnail_path=None,
        metadata=VideoMetadata(
            title="Test Video",
            duration=10.0,
            has_audio=True,
            source="upload",
            source_url="upload://test",
        ),
    )

    fake_transcript = Transcript(
        words=[
            Word(start=0.1, end=0.8, text="Hello"),
            Word(start=0.9, end=1.5, text="world"),
        ],
        sentences=[Sentence(start=0.1, end=1.5, text="Hello world")],
        language="en",
        language_probability=0.99,
        duration=10.0,
        warnings=[],
    )

    # Mock manager upload loading and audio extraction/transcription
    manager = app.state.jobs
    monkeypatch.setattr(manager.uploads, "load", lambda uid: fake_record)
    monkeypatch.setattr(
        "app.api.routes.extract_audio",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "app.api.routes.transcribe_audio",
        AsyncMock(return_value=fake_transcript),
    )

    # 1. Fresh transcription (calls transcribe_audio and builds segments)
    res = client.post(
        "/api/captions/transcribe",
        json={"upload_id": upload_id, "language": "auto", "task": "transcribe"},
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["upload_id"] == upload_id
    assert len(data["segments"]) > 0
    assert data["segments"][0]["text"] == "Hello world"
    assert data["cached"] is False

    # 2. Cached transcription (calls _refresh_cached_transcript)
    res_cached = client.post(
        "/api/captions/transcribe",
        json={"upload_id": upload_id, "language": "auto", "task": "transcribe"},
    )
    assert res_cached.status_code == 200, res_cached.text
    data_cached = res_cached.json()
    assert data_cached["cached"] is True
    assert data_cached["segments"][0]["text"] == "Hello world"


def test_transcribe_captions_translate_task(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upload_id = "b" * 32
    upload_dir = tmp_path / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    video_path = upload_dir / "video.mp4"
    video_path.write_bytes(b"dummy")

    fake_record = StoredUpload(
        id=upload_id,
        path=video_path,
        thumbnail_path=None,
        metadata=VideoMetadata(
            title="Nepali Speech",
            duration=5.0,
            has_audio=True,
            source="upload",
            source_url="upload://test",
        ),
    )

    fake_transcript = Transcript(
        words=[
            Word(start=0.2, end=1.0, text="Welcome"),
            Word(start=1.1, end=2.0, text="everyone"),
        ],
        sentences=[Sentence(start=0.2, end=2.0, text="Welcome everyone")],
        language="en",
        language_probability=0.95,
        duration=5.0,
        warnings=[],
    )

    manager = app.state.jobs
    monkeypatch.setattr(manager.uploads, "load", lambda uid: fake_record)
    monkeypatch.setattr(
        "app.api.routes.extract_audio",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "app.api.routes.transcribe_audio",
        AsyncMock(return_value=fake_transcript),
    )

    res = client.post(
        "/api/captions/transcribe",
        json={"upload_id": upload_id, "language": "ne", "task": "translate"},
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["translated"] is True
    assert data["segments"][0]["text"] == "Welcome everyone"
