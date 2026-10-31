#!/usr/bin/env python3
"""export_gguf.py - merge a trained LoRA adapter and export a GGUF for Ollama. WSL.

Standalone (decoupled from training) so it de-risks the GGUF toolchain early on an
EXISTING adapter, and later exports the Phase-4 winner. Unsloth's save_pretrained_gguf
shells out to llama.cpp (clone + cmake build on first run) -- if that fails, the fix is
a one-time build-deps install handed to the user (cmake/gcc); training itself is unaffected.

    python export_gguf.py --adapter persona_adapter_v2 --out persona_v2_gguf --quant q4_k_m

Then author a Modelfile (FROM ./<gguf>, SYSTEM = persona prompt) and `ollama create`.
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
    ap = argparse.ArgumentParser(description="Merge LoRA + export GGUF (Unsloth).")
    ap.add_argument("--adapter", required=True, help="Trained adapter dir (e.g. persona_adapter_v2).")
    ap.add_argument("--out", help="Output GGUF dir (default <adapter>_gguf).")
    ap.add_argument("--quant", default="q4_k_m", help="GGUF quant method (default q4_k_m).")
    ap.add_argument("--max-seq", type=int, default=2048)
    args = ap.parse_args()
    out = args.out or f"{args.adapter}_gguf"

    try:
        from unsloth import FastLanguageModel
    except ImportError as e:
        raise SystemExit(f"Unsloth not installed ({e}). Run in the WSL training venv.")

    print(f"Loading adapter {args.adapter} ...")
    model, tok = FastLanguageModel.from_pretrained(
        args.adapter, max_seq_length=args.max_seq, load_in_4bit=True, dtype=None
    )
    print(f"Exporting GGUF ({args.quant}) -> {out}  (first run builds llama.cpp; be patient) ...")
    try:
        model.save_pretrained_gguf(out, tok, quantization_method=args.quant)
    except Exception as e:  # noqa: BLE001 - surface the toolchain gap clearly
        raise SystemExit(
            f"\nGGUF export failed: {e}\n"
            "Likely the llama.cpp build toolchain is missing. In WSL the user can run:\n"
            "  sudo apt-get install -y build-essential cmake\n"
            "then re-run this script. (Training is unaffected by this.)"
        )
    print(f"\nGGUF written under {out}/")
    print("Next: author a Modelfile (FROM ./<file>.gguf, SYSTEM=<persona>) then "
          "`ollama create angel-brain -f Modelfile`.")


if __name__ == "__main__":
    main()
