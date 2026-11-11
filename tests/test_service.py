import io
import threading
import time
import wave

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from pydantic import ValidationError

from kani_tts_mcp.backend import split_text
from kani_tts_mcp.config import Settings
from kani_tts_mcp.jobs import JobStore
from kani_tts_mcp.schema import SpeechRequest
from kani_tts_mcp.service import create_app
from kani_tts_mcp.voices import import_voice, voice_path


class StubBackend:
    def __init__(self, home):
        self.home, self.active, self.peak = home, 0, 0
        self.lock = threading.Lock()

    def readiness(self):
        return {"ready": True}

    def unload(self):
        pass

    def generate(self, req, job_id, progress, cancelled):
        with self.lock:
            self.active += 1
            self.peak = max(self.active, self.peak)
        try:
            for i in range(12):
                progress("test", i)
                time.sleep(0.005)
            path = self.home / "outputs" / (job_id + ".wav")
            with wave.open(str(path), "wb") as wav:
                wav.setparams((1, 2, 22050, 0, "NONE", "not compressed"))
                wav.writeframes(bytes(1920))
            path.with_suffix(".json").write_text("{}")
            return {"audio_path": str(path), "duration_seconds": 0.04}
        finally:
            self.active -= 1


def wait(store, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = store.get(job_id)
        if result["status"] in {"succeeded", "failed", "cancelled"}:
            return result
        time.sleep(0.005)
    raise AssertionError("Job did not finish")


def test_serial_queue_idempotency_and_restart(tmp_path):
    backend = StubBackend(tmp_path)
    store = JobStore(tmp_path, backend)
    req = SpeechRequest(text="Hello world.", client_request_id="retry", seed=-1)
    first = store.submit(req)
    assert 0 <= first["request"]["seed"] < 2**32
    assert store.submit(req)["job_id"] == first["job_id"]
    with pytest.raises(ValueError, match="different request"):
        store.submit(SpeechRequest(text="Changed", client_request_id="retry"))
    jobs = [first] + [store.submit(SpeechRequest(text="Next")) for _ in range(3)]
    store.start()
    try:
        for job in jobs:
            assert wait(store, job["job_id"])["status"] == "succeeded"
        assert backend.peak == 1
    finally:
        store.close()
    unfinished = store.submit(SpeechRequest(text="Restart me"))
    store.update(unfinished["job_id"], status="running")
    recovered = JobStore(tmp_path, backend)
    assert recovered.get(unfinished["job_id"])["status"] == "failed"
    assert recovered.submit(req)["job_id"] == first["job_id"]


def test_cancel_queued_and_running(tmp_path):
    store = JobStore(tmp_path, StubBackend(tmp_path))
    queued = store.submit(SpeechRequest(text="Queued"))
    assert store.cancel(queued["job_id"])["status"] == "cancelled"
    running = store.submit(SpeechRequest(text="Running"))
    store.start()
    try:
        deadline = time.monotonic() + 5
        while store.get(running["job_id"])["status"] == "queued":
            assert time.monotonic() < deadline
            time.sleep(0.001)
        store.cancel(running["job_id"])
        assert wait(store, running["job_id"])["status"] == "cancelled"
    finally:
        store.close()


def reference(seconds=4, rate=24000):
    data = io.BytesIO()
    sf.write(
        data,
        0.1 * np.sin(np.arange(int(rate * seconds)) * (2 * np.pi * 200 / rate)),
        rate,
        format="WAV",
    )
    return data.getvalue()


def test_http_auth_voice_upload_and_artifacts(tmp_path):
    settings = Settings(home=tmp_path, token="test-token")
    with TestClient(create_app(settings, StubBackend(tmp_path))) as http:
        assert http.get("/health").status_code == 401
        http.headers["Authorization"] = "Bearer test-token"
        assert (
            http.get("/health", headers={"Origin": "https://other.example"}).status_code
            == 403
        )
        assert (
            http.post("/v1/jobs", json={"text": "Hello", "arbitrary": True}).status_code
            == 422
        )
        assert (
            http.post(
                "/v1/jobs", json={"text": "Hello", "voice_id": "a" * 32}
            ).status_code
            == 409
        )
        voice = http.post("/v1/voices", content=reference(), params={"name": "Test"})
        assert voice.status_code == 201
        voice_id = voice.json()["voice_id"]
        assert http.get("/v1/voices").json()["voices"][0]["voice_id"] == voice_id
        samples, rate = sf.read(voice_path(tmp_path, voice_id))
        assert rate == 16000 and abs(len(samples) / rate - 4) < 0.01
        job = http.post("/v1/jobs", json={"text": "Hello world", "voice_id": voice_id})
        assert job.status_code == 202
        job_id = job.json()["job_id"]
        assert wait(http.app.state.store, job_id)["status"] == "succeeded"
        assert http.get(f"/v1/jobs/{job_id}/audio").content.startswith(b"RIFF")
        assert http.get(f"/v1/jobs/{job_id}/metadata").json() == {}
        assert http.get(f"/v1/jobs/{job_id}/config.json").status_code == 404
        assert http.get("/v1/jobs/unknown").status_code == 404
        assert http.post("/v1/voices", content=b"not audio").status_code == 422


@pytest.mark.parametrize(
    "data",
    [
        {"text": "   "},
        {"text": "x" * 12001},
        {"temperature": float("nan")},
        {"seed": -2},
        {"seed": 2**32},
        {"voice_id": "../file"},
        {"top_p": 0},
        {"max_new_tokens": 3001},
        {"text": "<start_of_human> Hello"},
    ],
)
def test_request_bounds(data):
    with pytest.raises(ValidationError):
        SpeechRequest(**({"text": "Hello"} | data))


def test_voice_bounds(tmp_path):
    for data in (reference(1), reference(31)):
        with pytest.raises(ValueError, match="3-30"):
            import_voice(tmp_path, data)
    with pytest.raises(ValueError):
        voice_path(tmp_path, "../config")


def test_long_text_preserves_words_and_bounds():
    text = "This is a sentence. " * 100
    parts = split_text(text)
    assert len(parts) > 1 and all(0 < len(p) <= 300 for p in parts)
    assert " ".join(parts).split() == text.split()
    assert all(len(p) <= 300 for p in split_text("x" * 1000))
