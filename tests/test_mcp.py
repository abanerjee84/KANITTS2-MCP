"""Exercise real subprocess stdio and HTTP MCP sessions against an isolated service."""

import asyncio
import json
import os
import socket
import struct
import subprocess
import sys
import time
import wave
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from kani_tts_mcp.client import Client
from kani_tts_mcp.config import Settings


@pytest.fixture
def service(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    settings = Settings(
        home=tmp_path, port=port, token="test-only-secret", keep_loaded=False
    )
    settings.save()
    env = os.environ.copy()
    env["KANI_TTS_HOME"] = str(tmp_path)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    with wave.open(str(tmp_path / "reference.wav"), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(struct.pack("<h", 1000) * 64000)
    log = (tmp_path / "server.log").open("wb")
    process = subprocess.Popen(
        [sys.executable, "-m", "kani_tts_mcp", "serve"],
        env=env,
        stdout=log,
        stderr=log,
        stdin=subprocess.DEVNULL,
    )
    client = Client(settings, autostart=False)
    try:
        deadline = time.monotonic() + 15
        while not client.reachable():
            if process.poll() is not None or time.monotonic() > deadline:
                raise AssertionError((tmp_path / "server.log").read_text("utf-8"))
            time.sleep(0.1)
        yield settings, env
    finally:
        process.terminate()
        process.wait(timeout=15)
        log.close()


async def exercise(session, reference_path):
    await session.initialize()
    names = {t.name for t in (await session.list_tools()).tools}
    assert names == {
        "generate_speech",
        "get_speech_job",
        "cancel_speech_job",
        "speech_service_status",
        "upload_voice",
        "list_voices",
    }
    result = await session.call_tool("speech_service_status", {})
    assert not result.isError
    text = "\n".join(c.text for c in result.content if c.type == "text")
    assert '"sample_rate": 22050' in text
    uploaded = await session.call_tool(
        "upload_voice",
        {"audio_path": str(reference_path), "name": "MCP regression reference"},
    )
    assert not uploaded.isError
    voices = await session.call_tool("list_voices", {})
    assert uploaded.structuredContent["voice_id"] in [
        v["voice_id"] for v in voices.structuredContent["voices"]
    ]
    # These tests intentionally have no model weights. A real submission must fail honestly.
    submitted = await session.call_tool(
        "generate_speech",
        {"text": "Hello from Kani TTS.", "seed": -1},
    )
    assert not submitted.isError
    job_id = submitted.structuredContent["job_id"]
    assert 0 <= submitted.structuredContent["request"]["seed"] < 2**32
    for _ in range(40):
        response = await session.call_tool("get_speech_job", {"job_id": job_id})
        if response.structuredContent["status"] == "failed":
            assert "setup-models" in response.structuredContent["error"]
            break
        await asyncio.sleep(0.1)
    else:
        raise AssertionError("Missing models did not produce an actionable error")


async def test_stdio_mcp(service):
    settings, env = service
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "kani_tts_mcp", "mcp"], env=env
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(
            read, write, read_timeout_seconds=timedelta(seconds=15)
        ) as session:
            await exercise(session, settings.home / "reference.wav")


async def test_http_mcp(service):
    settings, env = service
    async with httpx.AsyncClient(
        headers={"Authorization": "Bearer " + settings.token}, trust_env=False
    ) as http:
        async with streamable_http_client(settings.url + "/mcp", http_client=http) as (
            read,
            write,
            _,
        ):
            async with ClientSession(
                read, write, read_timeout_seconds=timedelta(seconds=15)
            ) as session:
                await exercise(session, settings.home / "reference.wav")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows driver environment")
def test_filtered_mcp_environment_keeps_nvml_available():
    from mcp.client.stdio import get_default_environment

    from kani_tts_mcp.config import gpu_inventory

    baseline = gpu_inventory()
    if not baseline:
        pytest.skip("No local NVIDIA inventory")
    code = "import json; from kani_tts_mcp.config import gpu_inventory; print(json.dumps(gpu_inventory()))"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=get_default_environment(),
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    assert {g["uuid"] for g in json.loads(result.stdout)} == {
        g["uuid"] for g in baseline
    }


@pytest.mark.skipif(sys.platform != "win32", reason="Windows MCP process-job lifetime")
async def test_autostart_survives_adapter_disconnect(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    settings = Settings(home=tmp_path, port=port, token="test-lifetime-token")
    settings.save()
    client = Client(settings, autostart=False)
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "kani_tts_mcp", "mcp"],
        env={"KANI_TTS_HOME": str(tmp_path)},
    )
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("speech_service_status", {})
                assert not result.isError
        assert client.reachable(), "Shared worker died with its stdio adapter"
    finally:
        if client.reachable():
            client._request("POST", "/admin/shutdown")
        deadline = time.monotonic() + 10
        while client.reachable() and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
