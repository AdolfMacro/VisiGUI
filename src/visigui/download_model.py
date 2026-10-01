from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

from .vision.detector import DEFAULT_MODEL_PATH

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


def download_model(output_path: Path = DEFAULT_MODEL_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=".hand_landmarker.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            digest = hashlib.sha256()
            with urlopen(MODEL_URL, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != MODEL_SHA256:
            raise ValueError("Downloaded MediaPipe model failed SHA-256 verification")
        if temporary_path.stat().st_size < 1024 * 1024:
            raise ValueError("Downloaded MediaPipe model is unexpectedly small")
        os.replace(temporary_path, output_path)
        return output_path
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the official MediaPipe hand-landmarker model.")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    print(download_model(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
