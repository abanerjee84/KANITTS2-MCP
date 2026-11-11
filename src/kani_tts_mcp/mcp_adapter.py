import asyncio
import base64
import json

from mcp.server.fastmcp import FastMCP
from mcp.types import AudioContent, CallToolResult, TextContent, ToolAnnotations

from .client import Client
from .schema import PresetName, SpeechRequest


def structured(value):
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(value))],
        structuredContent=value,
    )


def create_mcp(settings, autostart=True):
    client = Client(settings, autostart=autostart)
    mcp = FastMCP(
        "local-kani",
        host="127.0.0.1",
        port=settings.port,
        streamable_http_path="/mcp",
        stateless_http=True,
        instructions="Local KaniTTS 2 400M English speech on RTX 3060. Returns job_id immediately; "
        "poll get_speech_job. For voice cloning upload 3-30 seconds of clean single-speaker audio "
        "with upload_voice, then pass voice_id. References are local paths. WAV output at 22050 Hz. "
        "Select a named demo speaker with voice (frank, chelsea, rory, jermaine, baddy, callum, andrew, kore, puck). "
        "A language_tag sets accent only; it does not select a speaker. "
        "Long text is split into short segments. Model weights use LFM1.0 terms.",
    )
    write = ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, openWorldHint=False
    )
    read = ToolAnnotations(readOnlyHint=True, openWorldHint=False)

    @mcp.tool(annotations=write)
    async def generate_speech(
        text: str,
        voice_id: str | None = None,
        language_tag: str | None = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        max_new_tokens: int = 2500,
        seed: int | None = None,
        client_request_id: str | None = None,
        voice: PresetName | None = None,
    ) -> CallToolResult:
        """Queue English speech. Set voice for a named preset or voice_id for a cloned voice.

        Omitted language_tag uses the preset's accent, or en_us without a preset.
        Omit both voice and voice_id for a model-selected speaker. Seed -1/omitted is random.
        """
        req = SpeechRequest(
            text=text,
            voice_id=voice_id,
            voice=voice,
            language_tag=language_tag,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            seed=seed,
            client_request_id=client_request_id,
        )
        return structured(await asyncio.to_thread(client.submit, req.model_dump()))

    @mcp.tool(annotations=write)
    async def upload_voice(
        audio_path: str, name: str = "Reference voice"
    ) -> CallToolResult:
        """Import an absolute local WAV/FLAC/OGG recording; return reusable voice_id for cloning.

        Use 3-30 seconds (10-20 recommended) of clean speech from one speaker.
        Embedding extraction happens in the GPU queue on the first speech request.
        """
        return structured(
            await asyncio.to_thread(client.upload_voice, audio_path, name)
        )

    @mcp.tool(annotations=read)
    async def list_voices() -> CallToolResult:
        """List named preset voices with availability, plus imported reference voices and IDs."""
        return structured(await asyncio.to_thread(client.voices))

    @mcp.tool(annotations=read)
    async def get_speech_job(
        job_id: str, include_audio: bool = False
    ) -> CallToolResult:
        """Get status, errors, WAV path and timing. Optionally attach audio up to 8 MiB after success."""
        job = await asyncio.to_thread(client.job, job_id)
        result = structured(job)
        if include_audio and job["status"] == "succeeded":
            if job["result"]["duration_seconds"] > 180:
                raise ValueError(
                    "Use the local audio path or HTTP download for audio over 180 seconds"
                )
            data = await asyncio.to_thread(client.artifact, job_id)
            if len(data) > 8 * 1024**2:
                raise ValueError(
                    "Use the local audio path or HTTP download for audio over 8 MiB"
                )
            result.content.append(
                AudioContent(
                    type="audio",
                    data=base64.b64encode(data).decode(),
                    mimeType="audio/wav",
                )
            )
        return result

    @mcp.tool(annotations=write)
    async def cancel_speech_job(job_id: str) -> CallToolResult:
        """Cancel a queued job or stop running generation at the next model forward pass."""
        return structured(await asyncio.to_thread(client.cancel, job_id))

    @mcp.tool(annotations=read)
    async def speech_service_status() -> CallToolResult:
        """Report readiness, model, voice cloning availability and configured GPU."""
        return structured(await asyncio.to_thread(client.health))

    return mcp
