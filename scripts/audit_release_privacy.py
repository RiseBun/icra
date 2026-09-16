"""Fail fast if a review checkout contains common identifying artifacts."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

PATTERNS = {
    "local_user_or_host": re.compile(
        r"(?:[A-Za-z]:\\Users\\[^\\/\r\n]+|/home/[^/\s]+|slurmfs|user_data)"
    ),
    "email": re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I),
    "private_material": re.compile(r"(?:BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY|ghp_[A-Za-z0-9_]+)"),
}
SKIP_SUFFIXES = {
    ".pdf", ".png", ".jpg", ".jpeg", ".mp4", ".pyc",
    ".log", ".aux", ".out", ".toc", ".synctex",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    findings: list[str] = []
    for path in root.rglob("*"):
        if (
            not path.is_file()
            or ".git" in path.parts
            or path.suffix.lower() in SKIP_SUFFIXES
            or path.name == "audit_release_privacy.py"
        ):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{name}: {path.relative_to(root)}")
    if findings:
        print("Potential identifying material found:")
        print("\n".join(sorted(set(findings))))
        return 1
    print("No common identifying strings found in text artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
