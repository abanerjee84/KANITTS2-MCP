import gc
import importlib.util
import json
import re
import time

from .config import select_gpu
from .jobs import Cancelled
from .models import CODEC_FILE, PINS, load_codec
from .presets import list_presets, load_preset
from .voices import voice_path


def split_text(text, limit=300):
    """Keep sentence boundaries where possible; bound every generation."""
    words = re.findall(r"\S+", text)
    chunks, current = [], ""
    for word in words:
        while len(word) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(word[:limit])
            word = word[limit:]
        if len(current) + len(word) + 1 > limit:
            chunks.append(current)
            current = ""
        current = (current + " " + word).strip()
        if len(current) >= limit // 2 and re.search(r"[.!?][\"']?$", current):
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


class KaniBackend:
    def __init__(self, settings):
        self.settings = settings
        self.model = self.codec = self.tokenizer = self.embedder = None
        self.memory = None

    def readiness(self):
        required = [
            "tts/model.safetensors",
            "tts/config.json",
            "tts/tokenizer.json",
            "codec/" + CODEC_FILE,
        ]
        missing = [n for n in required if not (self.settings.models / n).is_file()]
        packages = [
            n
            for n in ("torch", "transformers", "nanocodec_torch", "accelerate")
            if importlib.util.find_spec(n) is None
        ]
        return {
            "ready": not missing and not packages and bool(self.settings.gpu_uuid),
            "loaded": self.model is not None,
            "model": PINS["tts"][0],
            "missing_files": missing,
            "missing_packages": packages,
            "gpu_uuid": self.settings.gpu_uuid,
            "sample_rate": 22050,
            "voice_cloning_ready": (
                self.settings.models / "speaker/model.safetensors"
            ).is_file(),
            "keep_loaded": self.settings.keep_loaded,
            "memory_budget_gib": self.settings.memory_gib,
            "last_memory": self.memory,
            "presets": list_presets(self.settings),
        }

    def _load(self, progress):
        ready = self.readiness()
        if not ready["ready"]:
            raise RuntimeError(
                f"Inference unavailable: {ready}. Run kani-tts setup-models and install inference dependencies."
            )
        select_gpu(self.settings)
        import torch
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoTokenizer, GenerationConfig

        from ._upstream.model import KaniTTS2ForCausalLM

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly the configured CUDA GPU must be visible")
        if "RTX 3060" not in torch.cuda.get_device_name(0):
            raise RuntimeError("Configured RTX 3060 is unavailable")
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(
            min(self.settings.memory_gib * 1024**3 / total, 0.9)
        )
        progress("loading_tts", 5)
        root = str(self.settings.models / "tts")
        config = AutoConfig.from_pretrained(root, local_files_only=True)
        self.model = KaniTTS2ForCausalLM(
            config,
            audio_tokens_start=64410,
            **{
                key: getattr(config, key)
                for key in (
                    "tokens_per_frame",
                    "audio_step",
                    "use_learnable_rope",
                    "alpha_min",
                    "alpha_max",
                    "speaker_emb_dim",
                )
            },
        )
        weights = load_file(str(self.settings.models / "tts/model.safetensors"))
        if "lm_head.weight" not in weights and config.tie_word_embeddings:
            weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
        self.model.load_state_dict(weights, strict=True)
        del weights
        self.model.generation_config = GenerationConfig.from_model_config(config)
        self.model = self.model.eval().to(device="cuda:0", dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        progress("loading_codec", 10)
        self.codec = load_codec(self.settings.models / "codec" / CODEC_FILE)

    def unload(self):
        self.model = self.codec = self.tokenizer = self.embedder = None
        gc.collect()
        import sys

        if "torch" in sys.modules:
            import torch

            torch.cuda.empty_cache()

    def generate(self, req, job_id, progress, cancelled):
        from .metrics import GPUMemorySampler

        sampler = GPUMemorySampler().start()
        started = time.monotonic()
        try:
            if self.model is None:
                self._load(progress)
            import numpy as np
            import soundfile as sf
            import torch
            from safetensors.torch import load_file, save_file

            torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(req.seed)
            torch.cuda.manual_seed_all(req.seed)
            embedding = None
            if req.voice:
                progress("loading_preset_voice", 15)
                embedding = load_preset(
                    self.settings, req.voice, dtype=self.model.dtype
                )
            elif req.voice_id:
                reference = voice_path(self.settings.home, req.voice_id)
                cached = reference.with_suffix(".safetensors")
                progress("embedding_voice", 15)
                if cached.is_file():
                    embedding = load_file(str(cached))["embedding"].to("cuda:0")
                else:
                    from ._upstream.speaker_embedder import SpeakerEmbedder

                    if self.embedder is None:
                        self.embedder = SpeakerEmbedder(
                            str(self.settings.models / "speaker"), device="cuda:0"
                        )
                    samples, rate = sf.read(reference, dtype="float32")
                    embedding = self.embedder.embed_audio(samples, sample_rate=rate)
                    save_file(
                        {"embedding": embedding.detach().cpu().contiguous()},
                        str(cached),
                    )
                if embedding.shape != (1, 128) or not torch.isfinite(embedding).all():
                    raise ValueError("Invalid speaker embedding")
                embedding = embedding.to(dtype=self.model.dtype, device="cuda:0")
            chunks = split_text(req.text)
            waves = []

            def check_cancel(module, args):
                if cancelled.is_set():
                    raise Cancelled()

            hook = self.model.register_forward_pre_hook(check_cancel)
            try:
                for index, chunk in enumerate(chunks):
                    progress(
                        f"speaking_{index + 1}_of_{len(chunks)}",
                        20 + int(65 * index / len(chunks)),
                    )
                    tokens = self.tokenizer(
                        f"{req.language_tag}: {chunk}", return_tensors="pt"
                    ).input_ids
                    ids = torch.cat(
                        [torch.tensor([[64403]]), tokens, torch.tensor([[2, 64404]])],
                        dim=1,
                    ).to("cuda:0")
                    with torch.inference_mode():
                        generated = self.model.generate(
                            input_ids=ids,
                            attention_mask=torch.ones_like(ids),
                            max_new_tokens=req.max_new_tokens,
                            do_sample=True,
                            temperature=req.temperature,
                            top_p=req.top_p,
                            repetition_penalty=req.repetition_penalty,
                            eos_token_id=64402,
                            speaker_emb=embedding,
                        )[0, ids.shape[1] :]
                        starts = (generated == 64401).nonzero().flatten()
                        ends = (generated == 64402).nonzero().flatten()
                        if len(starts) != 1 or len(ends) != 1:
                            raise ValueError(
                                "Model did not finish speech. Shorten text or increase max_new_tokens."
                            )
                        codes = generated[starts[0] + 1 : ends[0]]
                        if not codes.numel() or codes.numel() % 4:
                            raise ValueError("Model returned incomplete audio frames")
                        codes = (
                            codes.reshape(-1, 4)
                            - torch.arange(4, device="cuda:0") * 4032
                            - 64410
                        )
                        if codes.min() < 0 or codes.max() >= 4032:
                            raise ValueError("Model returned invalid audio codes")
                        progress(
                            f"decoding_{index + 1}_of_{len(chunks)}",
                            25 + int(65 * index / len(chunks)),
                        )
                        wave, _ = self.codec.decode(
                            codes.T.unsqueeze(0),
                            torch.tensor([len(codes)], device="cuda:0"),
                        )
                        wave = wave.detach().float().cpu().numpy().reshape(-1)
                    if not np.isfinite(wave).all() or not wave.size:
                        raise ValueError("Codec returned invalid audio")
                    waves.append(wave)
                    if index + 1 < len(chunks):
                        waves.append(np.zeros(int(22050 * 0.15), dtype="float32"))
            finally:
                hook.remove()
            progress("saving", 95)
            audio = np.concatenate(waves)
            path = self.settings.home / "outputs" / (job_id + ".wav")
            sf.write(path, audio, 22050, subtype="PCM_16")
            elapsed = time.monotonic() - started
            self.memory = {
                "peak_allocated_mib": round(
                    torch.cuda.max_memory_allocated() / 1024**2, 1
                ),
                "peak_reserved_mib": round(
                    torch.cuda.max_memory_reserved() / 1024**2, 1
                ),
                "gpu_uuid": self.settings.gpu_uuid,
                "gpu_name": torch.cuda.get_device_name(0),
                "windows_driver": sampler.stop(),
            }
            result = {
                "audio_path": str(path),
                "audio_url": f"/v1/jobs/{job_id}/audio",
                "sample_rate": 22050,
                "duration_seconds": round(len(audio) / 22050, 3),
                "elapsed_seconds": round(elapsed, 3),
                "chunks": len(chunks),
                "seed": req.seed,
                "voice_id": req.voice_id,
                "voice": req.voice,
                "memory": self.memory,
            }
            path.with_suffix(".json").write_text(
                json.dumps({"request": req.model_dump(), "result": result}, indent=2),
                "utf-8",
            )
            return result
        except BaseException:
            self.unload()
            raise
        finally:
            sampler.stop()
            if not self.settings.keep_loaded:
                self.unload()
