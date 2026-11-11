# KANITTS2-MCP

Local **KaniTTS 2 English speech generation and voice cloning** through **MCP, a global CLI, or HTTP**. One background worker serves every client through a persistent SQLite queue.

Uses the [400M English checkpoint](https://huggingface.co/nineninesix/kani-tts-2-en), a WavLM speaker embedder, and NVIDIA NanoCodec. Outputs are **22,050 Hz mono PCM-16 WAV**, with request metadata and memory measurements. The supported target is **Windows x64 with an RTX 3060 12 GB**. The worker selects that card by UUID and refuses to switch to another GPU.

Voice cloning is built in. Import 3-30 seconds of one speaker, then reuse the returned voice ID. No transcript or voice fine-tuning is needed. The [upstream guide](https://github.com/nineninesix-ai/kani-tts-2) recommends 10-20 seconds of clear speech without music or background noise.

**Locally validated:** 32 tests pass, including both MCP transports and named voice selection. A 25.75-second cloned narration across two segments took 41.91 seconds with models resident and peaked at 2,719 MiB dedicated VRAM. A short warm speech request produced 7.12 seconds of audio in 11.14 seconds. See [measurements and limits](docs/gpu-validation.md); voice similarity was not scored.

## Install

Requires Git, [uv](https://docs.astral.sh/uv/getting-started/installation/), a CUDA 12.8-compatible NVIDIA driver, and enough disk space for Python, CUDA PyTorch, caches, outputs, and about 2.3 GiB of model files. A CUDA Toolkit or C++ compiler is not required by this integration.

In PowerShell:

```powershell
Set-Location E:\TOOLS
git clone https://github.com/abanerjee84/KANITTS2-MCP.git
Set-Location KANITTS2-MCP
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -DownloadModels -RegisterCodex
```

The setup script creates a Python 3.12 environment, installs CUDA PyTorch and locked packages, downloads pinned checkpoints with SHA-256 verification, installs `kani-tts` on your user PATH, and optionally registers MCP. Restart Codex to discover the tools. Open a new terminal to receive the PATH update.

Use `-RegisterClaudeCode` or `-RegisterGemini` for those clients. Registration preserves existing settings, makes a backup before adding an entry, and refuses to replace a different `local-kani` entry.

Without a PATH refresh, run commands as:

```powershell
.\bin\kani-tts.cmd doctor
# Or:
.\.venv\Scripts\python.exe -m kani_tts_mcp doctor
```

## Generate speech

```powershell
kani-tts generate --text "Hello. This speech is generated locally." --voice andrew --seed 42 --out hello.wav
kani-tts generate --text-file script.txt --wait --out narration.wav
kani-tts generate --request examples/speech.json --wait
```

`--out` waits automatically and saves a copy. Without `--wait` or `--out`, submission returns a job ID immediately. Use `kani-tts status JOB_ID`, `kani-tts wait JOB_ID`, or `kani-tts cancel JOB_ID` afterward. `kani-tts download JOB_ID --out speech.wav` retrieves a completed WAV. Add `--artifact metadata` to download the JSON sidecar.

Omit `--seed`, or use `--seed -1`, for a random seed. The resolved seed is stored with the job. A fixed seed improves repeatability on the same software and hardware; it is not a promise of identical results across versions.

Six English accent tags are available:

| Tag | Accent |
| --- | --- |
| `en_us` | US English, default |
| `en_nyork` | New York |
| `en_oakl` | Oakland |
| `en_glasg` | Glasgow |
| `en_bost` | Boston |
| `en_scou` | Liverpool |

```powershell
kani-tts generate --text "Good morning. How are you today?" --language-tag en_glasg --wait
```

**Accent tags do not select speakers.** For example, `--language-tag en_oakl --seed 204` alone does not reproduce Jermaine. Use `--voice jermaine` to supply the actual speaker embedding. Without a preset or reference voice, the model chooses a speaker.

## Named preset voices

`--voice` selects the official demo's speaker embedding and defaults to that speaker's accent. The preset names are `frank`, `chelsea`, `rory`, `jermaine`, `baddy`, `callum`, `andrew`, `kore`, and `puck`. Use `kani-tts voices` to list presets with availability alongside imported references. You can override the accent with `--language-tag` when desired.

Fresh `setup-models` installations include the presets. For an existing checkout after updating:

```powershell
git pull
kani-tts stop
kani-tts setup-presets
```

These five commands select the exact speaker embeddings, text, accent defaults, and seeds used for the example clips:

```powershell
kani-tts generate --text "Good morning. Your coffee is ready, and the day is yours." --voice frank --seed 201 --out frank.wav
kani-tts generate --text "The city is wide awake, so let's make something happen." --voice chelsea --seed 202 --out chelsea.wav
kani-tts generate --text "Take your time. A good story is always worth the wait." --voice rory --seed 203 --out rory.wav
kani-tts generate --text "Well, that went better than expected. Let's call it a win." --voice jermaine --seed 204 --out jermaine.wav
kani-tts generate --text "Come on, we're nearly there. Just one more step." --voice baddy --seed 205 --out baddy.wav
```

Preset downloads are pinned to the [official demo revision](https://huggingface.co/spaces/nineninesix/kanitts-2-en/tree/3b4930245592b63263d490503212ce44d8a00277/speakers), verified with SHA-256, and loaded locally with `weights_only=True`. Missing or corrupted presets produce an error with `setup-presets` instructions; they never silently fall back to an unconditioned voice. Choose one of `--voice`, `--voice-id`, or `--reference-audio` for each request.

## Clone a voice

```powershell
# One command: import a recording and generate speech with it.
kani-tts generate --text "Here is a new sentence in the reference voice." --reference-audio C:\audio\reference.wav --wait --out cloned.wav

# Or import once, then reuse the returned voice_id.
kani-tts upload-voice C:\audio\reference.wav --name "My voice"
kani-tts voices
kani-tts generate --text "Another sentence in the same voice." --voice-id VOICE_ID --wait --out another.wav
```

References accept WAV, FLAC, OGG, and other formats supported by SoundFile. Use 3-30 seconds, at most 32 MiB, with a single speaker. Inputs are copied, mixed to mono, and resampled to 16 kHz. Embeddings are computed on the first queued speech request and cached as safetensors. The reference remains in `data/voices`; no audio is uploaded to a hosted inference service.

Cloning quality depends on the recording and the model. An accent tag can influence a cloned voice too. Use recordings you have permission to use.

## MCP

The stdio adapter connects to the same worker as the CLI and HTTP API. It starts the worker automatically. On Windows the worker is launched hidden through WMI and survives an MCP client's disconnect.

```powershell
kani-tts register codex
kani-tts register claude-code
kani-tts register gemini
kani-tts mcp-config
```

For other hosts, adapt [mcp-config.example.json](mcp-config.example.json) to the checkout path. Start the adapter with `.venv/Scripts/python.exe -m kani_tts_mcp mcp`. Diagnostics go to stderr; stdio is reserved for MCP.

| Tool | Purpose |
| --- | --- |
| `generate_speech` | Queue English text, preset `voice` or reference `voice_id`, accent, sampling parameters and seed |
| `upload_voice` | Import an absolute local reference-audio path and return a voice ID |
| `list_voices` | List named presets with availability and imported references |
| `get_speech_job` | Read status, stage, errors, output paths, timing and memory |
| `cancel_speech_job` | Cancel queued work or interrupt generation at a model boundary |
| `speech_service_status` | Read readiness, GPU and model availability |

`get_speech_job(include_audio=true)` attaches completed WAV audio, limited to 180 seconds and 8 MiB. For larger output, use the local path or HTTP download. The client must support MCP audio content to play the attachment.

Example request:

```json
{
  "text": "Welcome. This sentence was generated on the local machine.",
  "voice_id": null,
  "voice": "jermaine",
  "language_tag": null,
  "temperature": 1.0,
  "top_p": 0.95,
  "repetition_penalty": 1.1,
  "max_new_tokens": 2500,
  "seed": -1,
  "client_request_id": "optional-unique-retry-key"
}
```

Retrying an identical request with the same `client_request_id` returns the existing job. Reusing that ID for different parameters returns an error.

For MCP and HTTP requests, `voice` is the preset name and `voice_id` is an imported recording ID. They are mutually exclusive. Omitted or null `language_tag` uses the preset's accent, or `en_us` when no preset is selected. Restart the MCP host after updating to discover the new `voice` parameter.

## HTTP

The worker binds only to `127.0.0.1:8768`. Every endpoint, including `/mcp` and `/health`, requires the bearer token stored in local `data/config.json`. Browser requests from other origins are rejected. Do not publish that config file or token.

```powershell
kani-tts start
$kaniConfig = Get-Content data/config.json -Raw | ConvertFrom-Json
$kaniHeaders = @{Authorization = "Bearer $($kaniConfig.token)"}
$kaniJob = Invoke-RestMethod http://127.0.0.1:8768/v1/jobs -Method Post -Headers $kaniHeaders -ContentType application/json -Body '{"text":"Hello from HTTP.","seed":42}'
Invoke-RestMethod "http://127.0.0.1:8768/v1/jobs/$($kaniJob.job_id)" -Headers $kaniHeaders
```

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Service and inference readiness |
| POST | `/v1/jobs` | Submit a request; returns HTTP 202 |
| GET | `/v1/jobs/{id}` | Poll job |
| POST | `/v1/jobs/{id}/cancel` | Cancel job |
| GET | `/v1/jobs/{id}/audio` | Download successful WAV |
| GET | `/v1/jobs/{id}/metadata` | Download request/result JSON |
| POST | `/v1/voices?name=MyVoice` | Upload raw audio bytes, not multipart or a local path |
| GET | `/v1/voices` | List preset voices and imported references |
| POST | `/admin/shutdown` | Stop worker |
| MCP | `/mcp` | Streamable HTTP MCP; use the same bearer header |

For references from an isolated HTTP client, upload raw audio to `/v1/voices`, then submit the returned ID. MCP's `upload_voice` path refers to the computer running this project.

## Worker and GPU policy

`kani-tts start`, `stop`, and `doctor` manage or inspect the shared worker. `kani-tts serve` runs it in the foreground for diagnostics. Each data home has one worker lock and a queue bounded at 32 active jobs. Jobs persist across restarts; interrupted running jobs are marked failed, and queued jobs resume.

The default is to release the models after every job so that the RTX 3060 can run FLUX3-MCP or YUE2-MCP. For repeated narration, keep the models resident:

```powershell
kani-tts stop
kani-tts configure --residency resident
kani-tts start

# Release weights after each job again:
kani-tts stop
kani-tts configure --residency unload
```

These projects have separate workers and queues. This project does not stop or coordinate the other workers automatically. Stop another resident model worker if it leaves insufficient VRAM.

The CUDA allocator budget defaults to 8 GiB on the selected RTX 3060. TTS uses BF16; codec and speaker inference use FP32. Model inference stays on that GPU. CPU work handles loading, tokenization, reference resampling, file I/O, and metrics. CUDA selection occurs before importing the inference runtime. No CPU model-offload fallback is configured.

Long input is split into segments of at most 300 characters, preferring sentence boundaries, with 150 ms between segments. Voice embeddings are reused between segments. This avoids asking the model for one continuous long utterance, but prosody can change at joins. Inputs are limited to 12,000 characters. Generated duration is model-dependent; there is no exact-duration parameter.

Cancellation is checked between stages and on each TTS forward pass. Loading, embedding extraction, and a running codec operation finish before cancellation is observed. Failed or cancelled requests do not expose successful artifacts.

## Local files

Set `KANI_TTS_HOME` to use another data directory. Defaults:

```text
data/config.json       GPU UUID, bearer token, port, allocator budget, residency
data/models/           Pinned weights, tokenizers, download verification manifest
data/models/presets/   Verified named speaker embeddings from the official demo
data/voices/           Reference WAVs, voice metadata, cached embeddings
data/outputs/          Job WAVs and JSON metadata
data/jobs.sqlite3      Persistent queue
data/service.log       Worker diagnostics
```

All are ignored by Git. Removing a reference after it has been used can prevent later requests using its ID from running. Make configuration changes while the worker is stopped. If multiple RTX 3060 cards are present, set the intended `gpu_uuid` in `data/config.json`.

## Implementation and validation

Kani's pinned model and speaker modules are included in `_upstream/`. The integration loads the TTS checkpoint strictly, including its speaker projection and learnable RoPE parameters. It reads NVIDIA's original codec archive with `weights_only=True`, translates legacy weight-normalization names, and uses `nanocodec-torch` to load the decoder. A grouped transpose convolution replaces the package's per-group Python loop; tests compare the outputs numerically.

This avoids installing NeMo's training stack, its conflicting Transformers constraint, TorchCodec, Triton, vLLM, and FFmpeg for reference decoding. Upstream source and checkpoint pins are recorded in [research notes](docs/upstream-research.md). Installed package versions are in [requirements.lock.txt](requirements.lock.txt).

```powershell
powershell -ExecutionPolicy Bypass -File scripts/test.ps1
```

Tests cover actual stdio and HTTP MCP sessions, hidden-worker lifetime after disconnect, request bounds, queue persistence, idempotency, cancellation, authentication, voice upload/resampling, artifact retrieval, and codec convolution equivalence. Protocol tests intentionally use isolated homes without model files so missing weights must produce actionable errors.

To generate all five named examples through the public CLI:

```powershell
.\.venv\Scripts\python.exe scripts/validate_presets_live.py
# Compare against the original local demo WAVs, if you have them:
.\.venv\Scripts\python.exe scripts/validate_presets_live.py --compare-dir data/demos
```

The preset test writes a sanitized report to `docs/preset-validation.json` and audio to `data/preset-validation/`. It uses [five-voices.json](examples/five-voices.json); it does not depend on the temporary script used to create the original demos.

For an opt-in GPU test that generates speech, imports a synthetic reference, generates a clone, checks MCP audio attachments, and synthesizes a longer passage through HTTP MCP:

```powershell
.\.venv\Scripts\python.exe scripts/validate_mcp_live.py
```

This writes local WAVs and a sanitized `docs/live-mcp-validation.json` report. See [GPU validation notes](docs/gpu-validation.md) for the measured local results and their limits.

If `doctor` reports missing files, rerun `kani-tts setup-models`. Downloads resume by verified ranges. If the worker will not start, inspect `data/service.log`; another process may own port 8768. If a generation reaches its token ceiling without an end marker, shorten the text or raise `--max-new-tokens` up to 3000. Audio quality, pronunciation, identity preservation, and prosody still depend on the model.

## Licenses

The Python package's Apache-2.0 declaration does not cover every asset: English weights use **LFM1.0**, speaker weights and the derived Orange speaker implementation use **CC-BY-SA-3.0**, and codec weights use the **NVIDIA Open Model License**. See [third-party notices](THIRD_PARTY_NOTICES.md) and each checkpoint's terms. Weights and recordings are downloaded separately and are not committed to this repository.
