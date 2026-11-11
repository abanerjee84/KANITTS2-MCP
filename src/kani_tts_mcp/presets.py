"""Named speaker embeddings from the pinned official KaniTTS 2 demo."""

import hashlib
import json
import ssl
from pathlib import Path

REPO = "nineninesix/kanitts-2-en"
REVISION = "3b4930245592b63263d490503212ce44d8a00277"
PRESETS = json.loads(
    Path(__file__).with_name("preset-manifest.json").read_text("utf-8")
)


def preset_path(settings, name):
    if name not in PRESETS:
        raise ValueError(f"Unknown preset voice: {name}")
    return settings.models / "presets" / (name + ".pt")


def list_presets(settings):
    return [
        {
            "voice": name,
            "name": item["name"],
            "accent": item["accent"],
            "language_tag": item["language_tag"],
            "available": preset_path(settings, name).is_file(),
        }
        for name, item in PRESETS.items()
    ]


def setup_presets(settings):
    import httpx
    import truststore
    from filelock import FileLock

    directory = settings.models / "presets"
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / "download.lock"), timeout=120):
        with httpx.Client(
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
            follow_redirects=True,
            timeout=30,
        ) as client:
            for name, item in PRESETS.items():
                path = preset_path(settings, name)
                if (
                    path.is_file()
                    and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
                ):
                    continue
                url = f"https://huggingface.co/spaces/{REPO}/resolve/{REVISION}/speakers/{item['file']}"
                response = client.get(url)
                response.raise_for_status()
                if hashlib.sha256(response.content).hexdigest() != item["sha256"]:
                    raise RuntimeError(f"Preset SHA-256 verification failed: {name}")
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(response.content)
                temporary.replace(path)
    return {"repo": REPO, "revision": REVISION, "presets": list_presets(settings)}


def load_preset(settings, name, *, dtype, device="cuda:0"):
    import torch

    path = preset_path(settings, name)
    if not path.is_file():
        raise RuntimeError(
            f"Preset {name!r} is not installed. Run kani-tts setup-presets."
        )
    if hashlib.sha256(path.read_bytes()).hexdigest() != PRESETS[name]["sha256"]:
        raise RuntimeError(
            f"Preset {name!r} failed verification. Run kani-tts setup-presets."
        )
    embedding = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(embedding, torch.Tensor) or embedding.shape not in {
        (128,),
        (1, 128),
    }:
        raise ValueError(f"Preset {name!r} has an invalid embedding shape")
    if not torch.isfinite(embedding).all():
        raise ValueError(f"Preset {name!r} has invalid embedding values")
    return embedding.reshape(1, 128).to(device=device, dtype=dtype)
