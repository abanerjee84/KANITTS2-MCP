"""Bounded local reference recordings; no pickle or executable user inputs."""

import io
import json
import math
import uuid

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .schema import MAX_REFERENCE_BYTES

MAX_BYTES = MAX_REFERENCE_BYTES


def voice_path(home, voice_id):
    if len(voice_id) != 32 or any(c not in "0123456789abcdef" for c in voice_id):
        raise ValueError("Invalid voice ID")
    path = home / "voices" / (voice_id + ".wav")
    if not path.is_file():
        raise ValueError("Unknown voice ID; upload a reference recording first")
    return path


def import_voice(home, data, name="Reference voice"):
    if len(data) > MAX_BYTES:
        raise ValueError("Reference exceeds 32 MiB")
    with sf.SoundFile(io.BytesIO(data)) as source:
        duration = source.frames / source.samplerate
        if not 3 <= duration <= 30:
            raise ValueError("Reference must contain 3-30 seconds of one speaker")
        if source.channels > 8 or source.samplerate > 192000:
            raise ValueError("Unsupported channel count or sample rate")
        samples = source.read(dtype="float32", always_2d=True).mean(axis=1)
        rate = source.samplerate
    if not np.isfinite(samples).all() or float(np.max(np.abs(samples))) < 1e-5:
        raise ValueError("Reference is silent or contains invalid samples")
    divisor = math.gcd(rate, 16000)
    samples = resample_poly(samples, 16000 // divisor, rate // divisor).astype(
        "float32"
    )
    identity = uuid.uuid4().hex
    directory = home / "voices"
    directory.mkdir(parents=True, exist_ok=True)
    sf.write(directory / (identity + ".wav"), samples, 16000, subtype="PCM_16")
    result = {
        "voice_id": identity,
        "name": name[:120],
        "duration_seconds": round(duration, 3),
        "sample_rate": 16000,
    }
    (directory / (identity + ".json")).write_text(json.dumps(result, indent=2), "utf-8")
    return result


def list_voices(home):
    return [
        json.loads(p.read_text("utf-8"))
        for p in sorted((home / "voices").glob("*.json"))
    ]
