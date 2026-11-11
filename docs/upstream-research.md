# KaniTTS 2 upstream integration notes

Checked 2026-11-11 against official repositories, model cards, and package metadata.

## Model and checkpoints

Use `nineninesix/kani-tts-2-en`, the 400M English KaniTTS 2 model, rather than the older `kani-tts-400m-en`. Output is 22,050 Hz mono. The model card reports about 3 GB VRAM on RTX 5080; that is an upstream benchmark, not an RTX 3060 measurement. Inputs exceeding approximately 40 seconds of speech can degrade. Split longer scripts into short paragraphs. [Model card](https://huggingface.co/nineninesix/kani-tts-2-en)

Pinned public, ungated Hugging Face revisions:

| Component | Repository | Revision | Inference files |
| --- | --- | --- | --- |
| English TTS | `nineninesix/kani-tts-2-en` | `733435d9229fb6d5d2cdf23a44399a155ca2ab78` | `model.safetensors`, tokenizer/config JSON files |
| Speaker embedder | `nineninesix/speaker-emb-tbr` | `419c1dba057b2f92aba4259a101d785cf8ef213d` | `model.safetensors`, `config.json` |
| NanoCodec | `nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps` | `5c8e22ed763c14d81337fbe6ca74062f3d10f7e5` | `nemo-nano-codec-22khz-0.6kbps-12.5fps.nemo` |

Revisions obtained from the official [TTS API](https://huggingface.co/api/models/nineninesix/kani-tts-2-en), [speaker API](https://huggingface.co/api/models/nineninesix/speaker-emb-tbr), and [codec API](https://huggingface.co/api/models/nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps). Exclude training checkpoints such as `optimizer.pt` from downloads.

## API and voice cloning

The upstream Python package is `kani-tts-2`, with import namespace `kani_tts`. Source inspected at commit `14a9caef4a6b90df95edd6afc482bf1161a43e84`.

```python
from kani_tts import KaniTTS, SpeakerEmbedder

tts = KaniTTS(
    "nineninesix/kani-tts-2-en",
    device_map="auto",
    max_new_tokens=3000,
    show_info=False,
)
embedder = SpeakerEmbedder(device="cpu")
speaker = embedder.embed_audio(mono_float32_audio, sample_rate=sample_rate)
audio, returned_text = tts(
    "Hello, world!",
    speaker_emb=speaker,
    temperature=1.0,
    top_p=0.95,
    repetition_penalty=1.1,
)
tts.save_audio(audio, "speech.wav")
```

`speaker_emb` is optional; omitting it generates without reference conditioning. Speaker embeddings have shape `[1, 128]`. The embedder accepts a local model directory through `model_name`. `embed_audio` resamples to 16 kHz and truncates to 30 seconds. Use a clean recording containing one speaker; upstream recommends 10–20 seconds. Saving reusable embeddings avoids reloading WavLM for every generation. [API implementation](https://github.com/nineninesix-ai/kani-tts-2/blob/14a9caef4a6b90df95edd6afc482bf1161a43e84/kani_tts/api.py), [speaker implementation](https://github.com/nineninesix-ai/kani-tts-2/blob/14a9caef4a6b90df95edd6afc482bf1161a43e84/kani_tts/speaker_embedder.py)

Implementation recommendation: decode reference files with SoundFile, average channels, and pass the array to `embed_audio`. This bypasses `torchaudio.load`, which otherwise introduces TorchCodec/FFmpeg file-decoding requirements. CPU embedding extraction is possible when GPU memory is limited. The completed local integration uses GPU embedding extraction; the measured combined workload fits the RTX 3060. These are local integration choices, not upstream performance claims. See [local validation](gpu-validation.md).

## Dependencies and Windows

Upstream package metadata requires Python >=3.10, PyTorch >=2.8, matching torchaudio >=2.8, NeMo Toolkit `[tts]` ==2.4.0, NumPy, SciPy, librosa, OmegaConf, and TorchCodec >=0.10. Its README separately requests Transformers ==4.56.0. [Package metadata](https://github.com/nineninesix-ai/kani-tts-2/blob/14a9caef4a6b90df95edd6afc482bf1161a43e84/pyproject.toml), [README](https://github.com/nineninesix-ai/kani-tts-2)

The NeMo 2.4.0 TTS extra requests Transformers 4.51–4.52 and includes training, text-normalization, and ASR dependencies. This conflicts with the Kani README pin. The official server acknowledges the conflict and installs NeMo first, then upgrades Transformers. Its installation uses Linux tools and Triton; it does not establish native Windows support. [NeMo package metadata](https://pypi.org/pypi/nemo-toolkit/2.4.0/json), [official server](https://github.com/nineninesix-ai/kani-tts-2-openai-server)

NeMo's TTS model package eagerly imports unrelated models. Its codec imports also traverse the ASR module package. Missing dependencies encountered along those paths may belong to unrelated models; a minimal Windows integration needs explicit import tests and actual audio generation. [TTS imports](https://github.com/NVIDIA/NeMo/blob/v2.4.0/nemo/collections/tts/models/__init__.py), [codec implementation](https://github.com/NVIDIA/NeMo/blob/v2.4.0/nemo/collections/tts/models/audio_codec.py), [codec modules](https://github.com/NVIDIA/NeMo/blob/v2.4.0/nemo/collections/tts/modules/audio_codec_modules.py)

### Portable codec alternative

NineNineSix also publishes `nanocodec-torch==1.0.0` on PyPI. The wheel implements NanoCodec with only PyTorch, NumPy, safetensors, and Hugging Face Hub. It includes original-NeMo-key weight mapping in `nanocodec_torch.utils.weight_loader_v2`. Its advertised GitHub source currently returns 404, and its advertised converted checkpoint is not publicly accessible. Use the available NVIDIA checkpoint rather than relying on that converted repository. [Publisher's package](https://pypi.org/project/nanocodec-torch/), [release metadata](https://pypi.org/pypi/nanocodec-torch/1.0.0/json)

The published wheel SHA-256 is `2b75897f375aefa191bafe1db0e6fc24dccf3851150846a4fb09b29db891968c`. Inspecting its code establishes this integration path: instantiate `AudioCodecModel()`, obtain the original state dictionary from the NVIDIA `.nemo` archive, convert CPU tensors to NumPy arrays, call `load_decoder_weights(model, weights)` and `load_quantizer_weights(model, weights)`, then use `model.decode(tokens, tokens_len)`. The expected token layout is `[B, 4, T]`. Checkpoint mapping and audio quality must be verified locally before treating this as a working backend. The default portable decoder runs grouped transpose convolution using a Python loop; it can be slow even when the language model fits easily in memory.

## Licenses

The components have separate licenses: English weights use Liquid AI LFM 1.0; the Kani package declares Apache-2.0; speaker weights use CC-BY-SA-3.0; NanoCodec weights use NVIDIA Open Model License. Retain the original notices when redistributing code. Do not describe all model assets as Apache-2.0. [TTS license metadata](https://huggingface.co/nineninesix/kani-tts-2-en), [Kani package metadata](https://github.com/nineninesix-ai/kani-tts-2/blob/main/pyproject.toml), [speaker model](https://huggingface.co/nineninesix/speaker-emb-tbr), [codec model](https://huggingface.co/nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps)
