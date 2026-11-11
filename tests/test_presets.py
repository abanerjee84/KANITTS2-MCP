import json
import sys

import pytest

from kani_tts_mcp.cli import main
from kani_tts_mcp.config import Settings
from kani_tts_mcp.presets import load_preset, setup_presets
from kani_tts_mcp.schema import SpeechRequest


def test_cli_supplies_named_speaker_not_just_accent(monkeypatch, tmp_path, capsys):
    requests = []

    def submit(client, request):
        requests.append(request)
        return {"job_id": "a" * 32, "status": "queued", "request": request}

    monkeypatch.setenv("KANI_TTS_HOME", str(tmp_path))
    monkeypatch.setattr("kani_tts_mcp.client.Client.submit", submit)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kani-tts",
            "generate",
            "--text",
            "Well, that went better than expected. Let's call it a win.",
            "--voice",
            "jermaine",
            "--seed",
            "204",
        ],
    )
    assert main() == 0
    assert requests[0]["voice"] == "jermaine"
    assert requests[0]["voice_id"] is None
    assert requests[0]["language_tag"] == "en_oakl"
    assert json.loads(capsys.readouterr().out)["request"]["voice"] == "jermaine"


@pytest.mark.parametrize(
    "voice,accent",
    [
        ("frank", "en_bost"),
        ("chelsea", "en_nyork"),
        ("rory", "en_glasg"),
        ("jermaine", "en_oakl"),
        ("baddy", "en_scou"),
    ],
)
def test_preset_resolves_accent_and_keeps_explicit_override(voice, accent):
    assert SpeechRequest(text="Hello", voice=voice).language_tag == accent
    assert (
        SpeechRequest(text="Hello", voice=voice, language_tag="en_us").language_tag
        == "en_us"
    )


def test_preset_selection_is_unambiguous():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="not both"):
        SpeechRequest(text="Hello", voice="jermaine", voice_id="a" * 32)
    with pytest.raises(ValidationError):
        SpeechRequest(text="Hello", voice="unknown")
    unconditioned = SpeechRequest(text="Hello", language_tag="en_oakl")
    assert unconditioned.voice is None and unconditioned.voice_id is None


def test_preset_setup_checks_pinned_hash_before_using_download(monkeypatch, tmp_path):
    import httpx

    monkeypatch.setattr(
        httpx.Client,
        "get",
        lambda client, url: httpx.Response(
            200, content=b"wrong weights", request=httpx.Request("GET", url)
        ),
    )
    with pytest.raises(RuntimeError, match="SHA-256"):
        setup_presets(Settings(home=tmp_path))
    assert not list((tmp_path / "models/presets").glob("*.pt"))


def test_missing_and_tampered_presets_never_fall_back(tmp_path):
    torch = pytest.importorskip("torch")
    settings = Settings(home=tmp_path)
    with pytest.raises(RuntimeError, match="setup-presets"):
        load_preset(settings, "jermaine", dtype=torch.bfloat16, device="cpu")
    path = tmp_path / "models/presets/jermaine.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="failed verification"):
        load_preset(settings, "jermaine", dtype=torch.bfloat16, device="cpu")


def test_retries_of_jobs_created_before_preset_support(tmp_path):
    from kani_tts_mcp.jobs import JobStore

    store = JobStore(tmp_path, None)
    req = SpeechRequest(text="Hello", client_request_id="legacy", seed=204)
    first = store.submit(req)
    legacy = first["request"].copy()
    legacy.pop("voice")
    with store.connect() as db:
        db.execute(
            "UPDATE jobs SET request=? WHERE id=?",
            (json.dumps(legacy), first["job_id"]),
        )
    assert store.submit(req)["job_id"] == first["job_id"]
