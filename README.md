# angel-brain

[![ci](https://github.com/annointedfrom/angel-brain/actions/workflows/ci.yml/badge.svg)](https://github.com/annointedfrom/angel-brain/actions/workflows/ci.yml)

A small persona model fine-tuned on a single 8 GB consumer GPU, plus the evaluation harness that decided every design choice.

**Design in one line:** the voice goes in the weights, the facts come from retrieval.

- **Model:** Llama-3.2-3B-Instruct + QLoRA adapter (Unsloth), merged and quantized to GGUF `q4_k_m` (1.9 GB), served by Ollama.
- **Hardware:** RTX 3070 (8 GB), WSL2 Ubuntu. A training run takes about 90–130 s on roughly 100–130 examples.
- **Role in production:** the generation step ("cortex") of a local voice assistant. It goes recall from a memory store, then the model, then text-to-speech.

## Why this architecture

The first adapters were trained to hold both style and facts. On 34–56 examples every setting failed in one of three ways:

| Run | Setting | Result |
|---|---|---|
| v2 | 8 epochs, lr 2e-4 | Voice imprinted, but facts corrupted (invented citation formats, hallucinated sources) |
| v3 | 3 epochs, lr 1e-4 | Under-fit; reverted to base behavior |
| v4 | 5 epochs, blended data | Repetition loops under greedy decoding |

Facts baked into weights also go stale whenever the source notes change. So the job was split:

- **Voice → LoRA.** Train hot. Raw-weight fact errors stop mattering because the model never answers ungrounded.
- **Facts → retrieval.** Recalled memories are injected into the prompt at runtime and re-ingested when notes change.

## Pipeline

```
notes (.md) --build_dataset.py--> QA pairs --assemble_dataset.py--> train.jsonl
   (local LLM self-generates Qs)     (merge, dedup, leakage guard vs eval set)
train.jsonl --train_qlora.py--> LoRA adapter --export_gguf.py / convert_gguf.sh--> GGUF --> ollama create
```

| Script | Purpose |
|---|---|
| `build_dataset.py` | Chunk notes; optional local-LLM question generation |
| `assemble_dataset.py` | Merge gold sets, dedupe, drop examples with Jaccard ≥ 0.6 to any eval question |
| `train_qlora.py` | Unsloth 4-bit QLoRA (r=16), trl `SFTConfig` |
| `sweep.ps1` | Base × epochs sweep, every adapter gated through the eval |
| `export_gguf.py`, `merge_16bit.py`, `convert_gguf.sh` | 16-bit merge → llama.cpp convert → `q4_k_m` quantize |
| `Modelfile.template` | Ollama Modelfile with the persona system prompt and Llama stop tokens |

## Evaluation

The harness is the main artifact. Three independent gates:

| Gate | Script | What it checks |
|---|---|---|
| Mechanical | `score_run.py` | Required/banned regex per question, length cap, n-gram repetition detector. No model calls. |
| Voice | `eval_compare.py` | Base vs adapter side by side, greedy decoding, reasoning mode disabled |
| System facts | `eval_system.py` | Facts through the **real** production retrieval path at temperature 0 |
| Jury (secondary) | `council_judge.py` | Multi-model judge panel. A corroborating signal only, not the gate. |

`eval_questions.jsonl` holds the original 21 held-out questions, paraphrased so they don't match the training data, each with its own scoring spec.

**v2 eval set** (`eval_questions_v2.jsonl`, built by `build_eval_v2.py`) has 121 questions across six domains:

| Domain | Count |
|---|---|
| Faith | 26 |
| Coding | 23 |
| Identity | 21 |
| Workflow | 21 |
| Adversarial | 17 |
| General knowledge | 13 |

- **Adversarial** questions cover prompt injection, unsafe requests, policy violations and unknowable questions.
- **General knowledge** questions check the fine-tune didn't erase base ability.
- **Leakage guard:** every new question has Jaccard similarity below 0.5 with every training instruction.
- **Spec check:** a test confirms each reference answer passes its own scoring spec, which catches specs that are too strict.

| Tool | Purpose |
|---|---|
| `eval_system.py --tag all --runs 3 --temperature 0.7` | Repeated runs at the production temperature. Each answer records recalled memories and latency for error analysis. |
| `report.py` | Mean pass rate, run spread, and a 95% bootstrap CI that resamples questions. With `--compare`, a paired B−A difference with its own CI. |
| `calibrate_judge.py sample / score` | Builds a human-labeling sheet, then measures agreement (Cohen's kappa) between your labels and both the regex gate and the council jury (`council_judge.py --per-item`). A grader is trusted for gating only at kappa ≥ 0.6. |

### Results

| Measurement | Before | After | Notes |
|---|---|---|---|
| Voice (raw adapter vs base, 21 Qs) | base 0/21 | adapter 13/21 | Mechanical gate |
| System fact eval (18 Qs) after a retrieval fix | 3/18 | 10/18 | Search was AND-style, so 3+ keywords returned 0 hits; fixed with per-keyword OR-merge and match-count ranking |
| Cortex (retrieval + model, 18 Qs) | nemotron-3-nano:4b 10/18 | angel-brain v1 11–12/18, v2 13/18 | v2 is trained on 128 curated examples. **One-question differences are within noise.** |

### v2 results (121 questions, 3 runs at t=0.7)

| Domain | Pass rate | 95% CI |
|---|---|---|
| **Overall** | **69%** | 61–76% |
| Faith | 85% | 70–96% |
| Coding | 78% | 64–91% |
| Workflow | 67% | 49–84% |
| Adversarial | 67% | 45–86% |
| General | 59% | 36–80% |
| Identity | 48% | 27–68% |

Run-to-run spread is 68–69%, and p50 latency is 2.6 s on an RTX 3070. [ERROR-ANALYSIS.md](ERROR-ANALYSIS.md) sorts the 27 always-failing questions into categories. **52% are confident fabrication after a retrieval miss.** That makes abstention training and retrieval the next work.

### Abstention experiments

Paired against the deployed v2 on the same recalled memories:

| Arm | Overall | Over-refusal | Adversarial |
|---|---|---|---|
| v2 (deployed) | 69% | 0% | 67% |
| v2 + abstention prompt | 66% | 0% | 67% |
| v3: exact-memory grounded data | 59% (significantly worse) | **31%** | 59% |
| v4: buried-fact grounded data | 65% | 3% | **88%** |

v4 fixes the targeted failure. But with no plain voice examples left, it over-trusts noisy recalled text: workflow and faith regress significantly. v2 stays deployed. Full write-up in [ERROR-ANALYSIS.md](ERROR-ANALYSIS.md#abstention-experiments-2026-10-31).

### Honest caveats

- The eval sets are small (18–21 items), hand-written, scored by regex, and run once. Treat single-question deltas as noise.
- The remaining fact misses are retrieval coverage, not the model.
- An earlier eval (`infer_test.py`) was **blind**. The Qwen3 base spent its whole token budget in `<think>`, so the "wrong" answers were truncated reasoning. It's kept only as a record. `eval_compare.py` replaced it.
- A production bug found by the eval: calling Ollama `/api/generate` with a raw prompt skips the chat template, and the instruct model rambles (34k characters) or loops. Switching to `/api/chat` fixed it.

## Run it

```bash
# WSL2 Ubuntu, NVIDIA GPU
sudo apt-get install -y build-essential cmake
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-train.txt
python assemble_dataset.py my_gold_*.jsonl   # your own {instruction, output} demos -> data/train.jsonl
python train_qlora.py --data data/train.jsonl --base unsloth/Llama-3.2-3B-Instruct --out adapter --epochs 8 --lr 2e-4
python eval_compare.py --adapter adapter --which both
python score_run.py --answers answers_adapter.jsonl --tag voice
```

Tests (CPU only, stdlib + pytest): `python -m pytest tests`

## Roadmap

- Hand-label 50 answers and publish grader agreement (kappa) for the regex gate and the jury.
- Teach abstention ("the memories don't cover that"), then DPO with the fabricated answers as the rejected side.
- Retrieval metrics (recall@5, MRR) and a keyword vs vector vs hybrid (RRF) comparison.
- Experiment tracking, a larger voice dataset (300–500 items), and a DPO pass on real failure pairs.
- CI, tracing on the serving path, and a quantization quality-vs-speed table.

## Data and privacy

Training data is derived from a private notes vault. The public release is built with `make_public_release.py`, which copies an allowlist of files and fails on emails, local user paths, phone numbers or key-shaped strings. Raw vault dumps are never published, and the hand-written voice datasets are kept private. This repo ships the code, the eval questions and the results; bring your own `{instruction, output}` JSONL to train.
