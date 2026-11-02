#!/usr/bin/env python3
"""make_public_release.py - build a sanitized public copy of the angel-brain pipeline.

The private repo mixes code with personal vault text (raw vault dumps, contact info,
local paths). This copies an ALLOWLIST of files into a fresh directory and fails if
any copied file trips the privacy scan. Datasets are excluded unless --include-data.

Usage:
  python make_public_release.py --out ../../angel-brain-public
  python make_public_release.py --out ../../angel-brain-public --include-data
  python make_public_release.py --scan-only          # scan the allowlist in place
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

CODE_FILES = [
    "README.md", "ERROR-ANALYSIS.md", "LICENSE", "requirements-train.txt", ".github/workflows/ci.yml",
    "build_dataset.py", "assemble_dataset.py", "train_qlora.py",
    "eval_compare.py", "eval_system.py", "score_run.py", "council_judge.py",
    "export_gguf.py", "merge_16bit.py", "convert_gguf.sh", "sweep.ps1",
    "Modelfile.template", "make_public_release.py", "eval_questions.jsonl",
    "eval_questions_v2.jsonl", "build_eval_v2.py", "report.py", "calibrate_judge.py",
    "grounding.py", "build_grounded_dataset.py", "quantize_gguf.sh", "prompts/voice_v2_abstain.txt",
    "embed_corpus.py", "retrieval_metrics.py", "retrieval_arms.py", "label_retrieval.py",
    "score_retrieval.py",
    "calibrate_floor.py",
    "tests/test_score_run.py", "tests/test_release.py", "tests/test_report.py", "tests/test_grounding.py",
    "tests/test_retrieval_metrics.py", "tests/test_retrieval_arms.py",
]
# Private-repo name -> published name.
RENAMES = {"public.gitignore": ".gitignore"}
# Hand-authored voice demos. vault_persona.jsonl (raw vault dump) is never published.
DATA_FILES = [
    "gold_persona.jsonl", "gold_faith.jsonl", "gold_coding.jsonl",
    "gold_vault.jsonl", "gold_workflow.jsonl", "gold_more.jsonl",
]

PRIVACY_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "windows user path": re.compile(r"[A-Za-z]:\\+Users\\+[^\\\s\"']+", re.IGNORECASE),
    "posix home path": re.compile(r"/home/[a-z_][a-z0-9_-]*/"),
    "phone number": re.compile(r"\b\d{3}[-. ]\d{3}[-. ]\d{4}\b"),
    "api key": re.compile(r"\b(sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,})\b"),
}
# The scanner's own pattern source would match itself.
SCAN_EXEMPT = {"make_public_release.py", "tests/test_release.py"}


def scan_text(text: str) -> list[str]:
    """Return a label per privacy pattern found in text."""
    return [label for label, rx in PRIVACY_PATTERNS.items() if rx.search(text)]


def scan_file(rel: str, root: Path = HERE) -> list[str]:
    if rel in SCAN_EXEMPT:
        return []
    text = (root / rel).read_text(encoding="utf-8-sig", errors="replace")
    return [f"{rel}: {label}" for label in scan_text(text)]


def release_files(include_data: bool) -> list[str]:
    return CODE_FILES + list(RENAMES) + (DATA_FILES if include_data else [])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="Destination directory (must not exist or be empty).")
    ap.add_argument("--include-data", action="store_true", help="Also publish gold voice datasets.")
    ap.add_argument("--scan-only", action="store_true", help="Scan the allowlist; copy nothing.")
    args = ap.parse_args()

    files = release_files(args.include_data)
    missing = [f for f in files if not (HERE / f).is_file()]
    if missing:
        sys.exit(f"missing allowlisted files: {missing}")

    hits = [hit for f in files for hit in scan_file(f)]
    if hits:
        sys.exit("privacy scan FAILED:\n  " + "\n  ".join(hits))
    print(f"privacy scan passed ({len(files)} files)")
    if args.scan_only:
        return
    if not args.out:
        sys.exit("--out is required unless --scan-only")

    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty; refusing to overwrite")
    for rel in files:
        dest = out / RENAMES.get(rel, rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE / rel, dest)
    print(f"wrote {len(files)} files to {out}")


if __name__ == "__main__":
    main()
