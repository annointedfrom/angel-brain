#!/usr/bin/env python3
"""merge_16bit.py - merge a LoRA adapter to a 16-bit HF dir (no GGUF). WSL venv.

Decoupled from Unsloth's flaky save_pretrained_gguf auto-builder: this only does the
merge, leaving a clean HF folder that convert_gguf.sh feeds to llama.cpp manually.

    .venv/bin/python merge_16bit.py --adapter angelbrain_v2 --out angelbrain_v2_gguf
"""
from __future__ import annotations

import argparse
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge LoRA adapter -> 16-bit HF dir.")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq", type=int, default=2048)
    args = ap.parse_args()

    from unsloth import FastLanguageModel

    print(f"Loading {args.adapter} ...", flush=True)
    model, tok = FastLanguageModel.from_pretrained(
        args.adapter, max_seq_length=args.max_seq, load_in_4bit=True, dtype=None
    )
    print(f"Merging -> {args.out} (16-bit) ...", flush=True)
    model.save_pretrained_merged(args.out, tok, save_method="merged_16bit")
    print("MERGED_OK", flush=True)


if __name__ == "__main__":
    main()
