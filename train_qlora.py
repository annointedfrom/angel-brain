#!/usr/bin/env python3
"""train_qlora.py - QLoRA fine-tune a persona/domain adapter on the RTX 3070 (8 GB).

SCAFFOLD: requires the training stack (NOT installed as of 2026-06-09). Set up first:
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install unsloth trl peft datasets bitsandbytes accelerate
(Recommended under WSL2 + Ubuntu; Unsloth's first-class target. See SPEC.md.)

Then:
    python build_dataset.py "<vault paths>" --mode qa -o data.jsonl
    python train_qlora.py --data data.jsonl --base unsloth/Qwen3-8B --out adapter

4-bit QLoRA fits 8 GB. Output is a small LoRA adapter; optionally merged to GGUF for Ollama.
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description="QLoRA fine-tune on a local GPU (Unsloth).")
    ap.add_argument("--data", default="data.jsonl", help="Training JSONL (instruction/input/output).")
    ap.add_argument("--base", default="unsloth/Qwen3-4B",
                    help="Base model (Unsloth repo). 4B fits 8GB; 8B needs more free VRAM.")
    ap.add_argument("--out", default="adapter", help="Output adapter dir.")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4, help="Learning rate. Lower (1e-4) = gentler, less catastrophic forgetting.")
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--gguf", action="store_true", help="Also merge+export GGUF for Ollama.")
    args = ap.parse_args()

    # Imports are deferred so --help works without the (heavy) training stack installed.
    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from datasets import load_dataset
        from trl import SFTTrainer, SFTConfig  # trl 0.24+: SFTConfig replaces TrainingArguments
    except ImportError as e:
        raise SystemExit(
            f"Training stack not installed ({e}).\n"
            "Set up first (see this file's docstring / SPEC.md):\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
            "  pip install unsloth trl peft datasets bitsandbytes accelerate"
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base, max_seq_length=args.max_seq, load_in_4bit=True, dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=args.lora_r, lora_alpha=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )

    from grounding import chat_messages

    def to_text(ex: dict) -> dict:
        # Grounded examples (with "memories"/"system") render in the exact serve-time format.
        msgs = chat_messages(ex)
        # enable_thinking=False keeps the rendered assistant turn free of Qwen3's
        # <think> scaffold, matching how the model is queried at inference (eval_compare.py).
        try:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(msgs, tokenize=False)
        return {"text": text}

    ds = load_dataset("json", data_files=args.data, split="train").map(to_text)

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        processing_class=tokenizer,  # trl 0.24 renamed tokenizer -> processing_class
        args=SFTConfig(
            per_device_train_batch_size=2, gradient_accumulation_steps=4,
            warmup_steps=5, num_train_epochs=args.epochs, learning_rate=args.lr,
            bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported(),
            logging_steps=1, optim="adamw_8bit", output_dir="outputs", seed=42,
            dataset_text_field="text", packing=False,  # max_seq_length is set at model load
        ),
    )
    trainer.train()

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Adapter saved -> {args.out}")
    if args.gguf:
        model.save_pretrained_gguf(args.out + "_gguf", tokenizer, quantization_method="q4_k_m")
        print(f"GGUF saved -> {args.out}_gguf  (then: ollama create my-model -f Modelfile)")


if __name__ == "__main__":
    main()
