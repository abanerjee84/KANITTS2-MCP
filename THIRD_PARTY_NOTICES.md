# Third-party components

This project is a local integration, independent of the model authors.

- `src/kani_tts_mcp/_upstream/model.py` and `speaker_embedder.py` are copied without changes from [nineninesix-ai/kani-tts-2](https://github.com/nineninesix-ai/kani-tts-2/tree/14a9caef4a6b90df95edd6afc482bf1161a43e84), commit `14a9caef4a6b90df95edd6afc482bf1161a43e84`. The package declares Apache-2.0. The speaker embedder explicitly credits Orange SA's CC-BY-SA-3.0 implementation; that attribution and license apply to the derived speaker module.
- [KaniTTS 2 English weights](https://huggingface.co/nineninesix/kani-tts-2-en) use the Liquid AI LFM1.0 license. These are separate from the Python package's Apache-2.0 declaration.
- [Speaker embedding weights](https://huggingface.co/nineninesix/speaker-emb-tbr) use CC-BY-SA-3.0 and derive from [Orange/Speaker-wavLM-tbr](https://huggingface.co/Orange/Speaker-wavLM-tbr).
- Named preset speaker embeddings are downloaded separately from the [official KaniTTS 2 demo](https://huggingface.co/spaces/nineninesix/kanitts-2-en/tree/3b4930245592b63263d490503212ce44d8a00277/speakers). The source revision and per-file SHA-256 values are pinned in `src/kani_tts_mcp/preset-manifest.json`. These files are not included in this repository; upstream terms apply.
- [NVIDIA NanoCodec weights](https://huggingface.co/nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps) use the NVIDIA Open Model License. They are downloaded separately and loaded with `weights_only=True`.
- [nanocodec-torch 1.0.0](https://pypi.org/project/nanocodec-torch/1.0.0/) is the portable PyTorch codec by NineNineSix. Its wheel includes Apache-2.0 license and NVIDIA attribution. Its original NVIDIA weights are used because the advertised converted checkpoint is unavailable.
- Queue, client, configuration, registration, Windows background launch, memory sampling, and download modules follow the local FLUX3-MCP and YUE2-MCP projects.

Weights, generated speech, reference recordings, caches, and credentials are not included in Git. Each model's terms still apply when downloaded. Full license texts for the vendored sources are in `licenses/`.
