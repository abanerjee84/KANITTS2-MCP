"""Generate the five named voices using the public CLI, optionally comparing old demos."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from kani_tts_mcp.config import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare-dir", type=Path, help="Directory with the original five demo WAVs"
    )
    args = parser.parse_args()
    examples = json.loads((ROOT / "examples/five-voices.json").read_text("utf-8"))
    output = ROOT / "data/preset-validation"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "method": "Public CLI with named voice, original text and seed",
        "voices": [],
    }
    for example in examples:
        path = output / (example["voice"] + ".wav")
        command = [
            sys.executable,
            "-m",
            "kani_tts_mcp",
            "generate",
            "--text",
            example["text"],
            "--voice",
            example["voice"],
            "--seed",
            str(example["seed"]),
            "--out",
            str(path),
        ]
        result = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=600
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        job = json.loads(result.stdout)
        assert (
            job["status"] == "succeeded" and job["request"]["voice"] == example["voice"]
        )
        item = {
            "voice": example["voice"],
            "request": job["request"],
            "duration_seconds": job["result"]["duration_seconds"],
            "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if args.compare_dir:
            baseline = args.compare_dir / example["demo_file"]
            item["demo_sha256"] = hashlib.sha256(baseline.read_bytes()).hexdigest()
            item["identical_to_demo"] = item["demo_sha256"] == item["audio_sha256"]
            assert item["identical_to_demo"], f"Demo mismatch: {example['voice']}"
        report["voices"].append(item)
        print(json.dumps(item), flush=True)
    target = ROOT / "docs/preset-validation.json"
    target.write_text(json.dumps(report, indent=2), "utf-8")
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
