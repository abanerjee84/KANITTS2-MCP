# Local GPU validation

Validated on 2026-11-11 with Windows x64, RTX 3060 12 GB, driver 610.88, Python 3.12.9, PyTorch 2.8.0+cu128, and Transformers 4.56.0. TTS runs in BF16; speaker and codec models run in FP32. The RTX 5060 Ti was excluded from the worker's CUDA visibility.

| Operation | Audio duration | Backend elapsed | Peak dedicated VRAM |
| --- | ---: | ---: | ---: |
| CLI speech, including initial model loading | 11.68 s | 45.985 s | 1,489 MiB |
| Stdio MCP speech, models already loaded | 7.12 s | 11.141 s | 1,281 MiB |
| Stdio MCP cloning, including a new speaker embedding | 6.00 s | 30.594 s | 2,391 MiB |
| HTTP MCP cloned narration, two segments, embedding cached | 25.75 s | 41.907 s | 2,719 MiB |

Elapsed time includes backend loading when applicable, synthesis, decoding, and saving. It excludes client startup, worker startup, and queue time. The first CLI run and MCP runs used different texts; these rows are examples, not a controlled speed comparison. The model drive was a hard disk. Driver counters were sampled approximately every 500 ms; shorter peaks can be missed. Windows reported 78 MiB of shared GPU memory, which is not evidence of model offloading.

The reproducible MCP sequence is in `scripts/validate_mcp_live.py`; its sanitized results and requests are in [live-mcp-validation.json](live-mcp-validation.json). The script:

1. Discovers all six MCP tools through a real stdio subprocess.
2. Queues speech, polls it to success, retrieves an MCP WAV attachment, and checks sample rate, finite samples, duration, and nonzero RMS.
3. Imports that generated recording as a synthetic reference through MCP, lists the voice ID, and synthesizes different text with it.
4. Opens a Streamable HTTP MCP session against the same worker, then synthesizes a longer cloned passage across two segments.

Cloning used a synthetic reference to test the embedding and conditioning pipeline without needing a personal recording. Speaker identity similarity, pronunciation, accent quality, and listener preference were not scored. Real recordings still need listening checks for the intended use.

The original 22 automated tests pass. They cover both MCP transports, worker survival after adapter disconnect, authentication, missing-weight errors, reference upload and resampling, request limits, idempotent retries, serialized execution, cancellation, restart recovery, artifact retrieval, long-text splitting, and convolution equivalence. Lint also passes. Named preset support adds ten tests for a current total of 32, plus preset selection checks through both real MCP transports.

The live tests caught and corrected three integration issues: NVIDIA's legacy `weight_g`/`weight_v` keys needed translation for the portable codec loader; FP32 speaker embeddings needed casting to the BF16 speech model; and an unnecessary NumPy import inside the adapter's upload thread could hang on Windows. The last issue has a regression test through a real stdio MCP upload.

The default configuration unloads models after each completed job. Resident mode was used for the MCP measurements above. The worker was stopped and the default unload policy restored after validation.

## Named demo voices and CLI parity

The original example clips used speaker embeddings from the official demo. An accent-only CLI request such as `--language-tag en_oakl --seed 204` supplied no speaker embedding and therefore could not reproduce Jermaine. The public CLI now accepts `--voice jermaine`, and the MCP/HTTP request accepts `"voice": "jermaine"`. Omitted accent tags resolve to the selected preset's original accent.

All five original clips were regenerated using the public CLI with the same text, preset and seed. Every WAV was byte-identical to its original demo on the validated machine:

| Preset | Seed | Duration | Byte-identical |
| --- | ---: | ---: | --- |
| Frank | 201 | 4.72 s | Yes |
| Chelsea | 202 | 4.40 s | Yes |
| Rory | 203 | 4.00 s | Yes |
| Jermaine | 204 | 2.64 s | Yes |
| Baddy | 205 | 3.60 s | Yes |

See [request and SHA-256 comparisons](preset-validation.json). Run `scripts/validate_presets_live.py --compare-dir data/demos` when the original local demo files are available. The comparison concerns reproducibility on this software/hardware combination; it does not promise identical bytes on other systems.

Preset files are pinned to official demo revision `3b4930245592b63263d490503212ce44d8a00277`, SHA-256 checked before loading, and read with `weights_only=True`. Preset selection and reference cloning are mutually exclusive. Missing or modified preset files cause an actionable error rather than silently changing the voice. The package wheel was also checked to include the preset manifest.
