"""Extract blind review frames from a video manifest."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for item in manifest["items"]:
        target = out / f"{item['review_id']}.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-ss",
                str(item["timestamp"]),
                "-i",
                args.video,
                "-frames:v",
                "1",
                str(target),
            ],
            check=True,
        )
    print(json.dumps({"n_frames": len(manifest["items"]), "output_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()
