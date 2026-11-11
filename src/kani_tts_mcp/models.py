"""Pinned downloads, excluding upstream training checkpoints."""

import json
import tarfile
from types import MethodType

PINS = {
    "tts": ("nineninesix/kani-tts-2-en", "733435d9229fb6d5d2cdf23a44399a155ca2ab78"),
    "speaker": (
        "nineninesix/speaker-emb-tbr",
        "419c1dba057b2f92aba4259a101d785cf8ef213d",
    ),
    "codec": (
        "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps",
        "5c8e22ed763c14d81337fbe6ca74062f3d10f7e5",
    ),
}
CODEC_FILE = "nemo-nano-codec-22khz-0.6kbps-12.5fps.nemo"


def setup_models(settings):
    import requests
    import truststore
    from huggingface_hub import HfApi, configure_http_backend, hf_hub_download

    from .download import file_sha256, verified_download

    # HF 0.x uses requests; use the Windows trust store without disabling TLS.
    truststore.inject_into_ssl()
    configure_http_backend(lambda: requests.Session())
    api = HfApi()
    manifest = {}
    for component, (repo, rev) in PINS.items():
        files = api.list_repo_files(repo, revision=rev)
        selected = (
            [
                n
                for n in files
                if "/" not in n
                and (
                    n.endswith((".json", ".safetensors"))
                    or n.startswith(("LICENSE", "README"))
                )
            ]
            if component != "codec"
            else [CODEC_FILE, "README.md"]
        )
        hashes = {}
        for name in selected:
            path = settings.models / component / name
            if name.endswith((".safetensors", ".nemo")):
                verified_download(repo, name, rev, path)
            else:
                hf_hub_download(repo, name, revision=rev, local_dir=path.parent)
            hashes[name] = file_sha256(path)
        manifest[component] = {"repo": repo, "revision": rev, "sha256": hashes}
    settings.models.mkdir(parents=True, exist_ok=True)
    from .presets import setup_presets

    manifest["presets"] = setup_presets(settings)
    (settings.models / "manifest.json").write_text(
        json.dumps(manifest, indent=2), "utf-8"
    )
    return manifest


def load_codec(path):
    """Load NVIDIA weights with the portable PyTorch codec, without NeMo."""
    import torch
    from nanocodec_torch.models.audio_codec import AudioCodecModel
    from nanocodec_torch.utils.weight_loader_v2 import (
        load_decoder_weights,
        load_quantizer_weights,
    )

    model = AudioCodecModel()
    with tarfile.open(path, "r:*") as archive:
        member = next(
            m for m in archive.getmembers() if m.name.endswith("model_weights.ckpt")
        )
        with archive.extractfile(member) as stream:
            raw = torch.load(stream, map_location="cpu", weights_only=True)
    raw = raw.get("state_dict", raw)
    weights = {
        k.replace(".weight_g", ".parametrizations.weight.original0").replace(
            ".weight_v", ".parametrizations.weight.original1"
        ): v.float().numpy()
        for k, v in raw.items()
        if isinstance(v, torch.Tensor)
    }
    load_decoder_weights(model, weights)
    load_quantizer_weights(model, weights)
    for layer in model.audio_decoder.up_sample_conv_layers:
        layer.forward = MethodType(grouped_transpose, layer)
    # Speech synthesis never encodes audio. Avoid keeping unused random encoder tensors.
    del model.audio_encoder
    return model.eval().to("cuda:0")


def grouped_transpose(layer, x):
    """Equivalent grouped CUDA kernel replacing the portable codec's Python loop."""
    from torch.nn.functional import conv_transpose1d

    output = conv_transpose1d(
        x,
        layer.weight.transpose(1, 2).contiguous(),
        layer.bias,
        stride=layer.stride,
        groups=layer.groups,
    )
    return output[:, :, : -layer.trim] if layer.trim else output
