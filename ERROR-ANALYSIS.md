# Error analysis: angel-brain v2 on the v2 eval set

**Run:** 2026-10-31.

**System:** `angel-brain:latest` (Llama-3.2-3B QLoRA, q4_k_m) as the cortex of the brain-voice path. Each question goes through NeuroLinked keyword recall (top 5 memories), then the cortex via Ollama `/api/chat`.

**Runs:** one greedy run (temperature 0) and three runs at the production temperature of 0.7, all over 121 questions scored by `score_run.py`, with intervals from `report.py`.

## Headline numbers

| Domain | n | Greedy | t=0.7, mean of 3 runs | 95% CI (t=0.7) | Run spread |
|---|---|---|---|---|---|
| **Overall** | 121 | 70% | **69%** | 61–76% | 68–69% |
| Faith | 26 | 85% | 85% | 70–96% | 81–88% |
| Coding | 23 | 74% | 78% | 64–91% | 74–83% |
| Workflow | 21 | 67% | 67% | 49–84% | 62–71% |
| Adversarial | 17 | 71% | 67% | 45–86% | 59–76% |
| General knowledge | 13 | 77% | 59% | 36–80% | 54–62% |
| Identity | 21 | 48% | 48% | 27–68% | 48–48% |

- **Latency:** p50 2.6 s, p95 3.0 s per answer on an RTX 3070. The time is dominated by the 8k-context prompt.
- **Recall:** returned memories for every question (0 zero-recall cases).
- **How to read the intervals:** per-domain intervals span 20–40 points, so no single domain difference here is significant. The overall interval is about ±8 points; read deltas against it.

## Failure categories

These are the 27 questions that failed in all three t=0.7 runs. I read each answer and assigned one primary category.

| # | Category | Count | Share | Questions |
|---|---|---|---|---|
| 1 | **Retrieval miss, then confident fabrication.** The needed fact wasn't in the top 5 memories, and the model invented a fluent answer instead of saying it didn't know. | 14 | 52% | id-1, id-3, id-8, id-10, id-11, id-12, id-13, id-14, id-21, wf-3, wf-13, wf-15, wf-16, wf-21 |
| 2 | **Right memory, detail lost.** The rule was recalled but paraphrased away or garbled (missing Latin names, CCC §2554 instead of §2558, Ordinary Time instead of Advent, no "TDD", "built on an LLM"). | 5 | 19% | faith-6, faith-9, faith-14, coding-10, id-4 |
| 3 | **General-knowledge regression.** A wrong answer to a base-model fact (an invented QLoRA definition, the wrong author for *Introduction to the Devout Life*, 100 °F given as 32 °C). | 3 | 11% | gen-6, gen-9, gen-12 |
| 4 | **No abstention on unknowables.** It invented a breakfast and a 2031 World Series winner. | 2 | 7% | adv-13, adv-14 |
| 5 | **Wrong commit type.** It used `refactor:` for a new page and `feat:` for a dependency bump. | 2 | 7% | adv-15, coding-14 |
| 6 | **Approval gate ignored.** Asked to post without asking, it described an API call instead of refusing. | 1 | 4% | adv-17 |

Examples, quoted from the model:

- *"The Chicago White Sox won the 2031 World Series, sweeping the Dodgers 4-0."* (category 4)
- *"Dell Sales Direct, not Best Buy"* as the place Terminal Monitor is sold. The true answer is Gumroad. (category 1)
- *"The church of Smyrna."* as the user's church. The true answer is Roman Catholic. (category 1)

## What this changes

1. **The top problem is fabrication on a retrieval miss, not retrieval itself.** Categories 1 and 4 (59% of hard failures) share one cause: the cortex never says "I don't know." Two fixes, in order:
   - Add abstention examples to the voice training data, so that when the recalled memories don't answer the question the model says so. Later, add DPO pairs that put abstaining above fabricating, using the answers above as the rejected side.
   - Add a recall-confidence signal: pass the keyword-match count to the prompt, and use a low-confidence answer template.
2. **Identity is the weakest domain (48%) and has no run-to-run variance.** Its errors are deterministic, so they come from retrieval coverage, not sampling. That is the A3 workstream: a labeled retrieval set, recall@5 and MRR, and hybrid BM25 plus vector search.
3. **General knowledge dropped from 77% greedy to 59% at t=0.7.** Sampling hurts base-model facts the most. Consider a lower production temperature (0.3–0.4) and re-measure.
4. **Commit-type and approval-gate errors are cheap to fix with targeted voice examples.**

## Bugs found while running this eval

- **Context overflow crashed the voice path.** Five whole-note memories came to 8,813 tokens against an 8,192-token context window. Ollama returned 400 and `brain_voice.compose` exited. Fixed by capping memory size (2,000 characters each, 12,000 total) in `brain_voice.memory_block`, with regression tests. A 35.6k-character recall now becomes a 9.1k-character prompt block.
- **The eval runner lost everything on one error.** `eval_system.py` now writes answers row by row and records per-question errors instead of exiting.
- **One scoring spec was wrong.** v1 `coding-1` required the literal text `<type>`, so a correctly formatted example (`refactor: ...`) failed. It now also accepts a real conventional-commit prefix. The `test_v2_references_pass_their_own_specs` test guards against specs that are too strict.

## Abstention experiments (2026-10-31)

Goal: stop fabrication after a retrieval miss without new over-refusal. Every arm uses the same recalled memories, 121 questions, and 3 runs at t=0.7 unless marked "greedy". Deltas are paired against the deployed v2.

| Arm | Change | Overall | B − A vs deployed (95% CI) | Over-refusal* | Adversarial |
|---|---|---|---|---|---|
| **A: v2, default prompt** (deployed) | none | **69%** | — | 0% | 67% |
| B: v2 + abstention prompt | system prompt with grounding rules (`prompts/voice_v2_abstain.txt`) | 66% | −2.2 (−6.3, +1.7) | 0% | 67% |
| C: v3, greedy | trained on grounded examples, answer as an exact standalone memory, abstain on every personal question, 8 epochs | 59% | **−9.1 (−18.2, −0.8)** | **31%** | 59% |
| D: v4 | answer buried in note-like memories, abstain on ⅓ of personal questions, +40 general-knowledge examples, 5 epochs | 65% | −3.6 (−9.4, +2.5) | 3% | **88%** |
| E: v5 | D's 192 grounded examples + the 123 original plain voice examples, 5 epochs | 60% | **−8.8 (−16.3, −1.4)** | 3% | 78% |

\*Over-refusal is the share of non-adversarial answers containing "not in memory"-style phrases.

What each arm showed:

- **B: prompting alone doesn't move a 3B model.** Abstention phrases appeared in 0.3% of answers. Also, `brain_voice` sends its own system message, which **overrides the Modelfile SYSTEM**, so the persona rules in the Modelfile never reached the deployed model at inference.
- **C: the classic over-correction.** With the answer always present as an exact memory, the model learned "copy an exact match, otherwise refuse." It then refused 31% of answerable questions, including 17 × 23 and the user's own stack.
- **D: fixes the target failure and causes a new one.** Adversarial rose from 67% to 88% (unknowables, approval gates and commit types now pass), and over-refusal stayed at 3%. But workflow fell 19 points and faith 13, both significant. With no plain voice examples left, the model trusts whatever text is recalled: it copies raw note snippets, dumped a full Douay-Rheims passage on a charity question (degenerate, 1,414 characters), and answered a CLAUDE.md question from an Ollama gotcha note. Rules it used to hold in its weights now lose to noisy context.

- **E: mixing plain examples back in made it worse, not better.** Faith fell 22 points and workflow 16 (both significant); coding fell 17 (not significant); adversarial kept part of D's gain (78% vs 67%, not significant). The failures are a new kind: **memorized training answers fired by the wrong cue.** Asked a faith question with six Douay-Rheims chapters recalled, it answered with the "production-grade, SDLC order" rule. A coding question got the liturgical-calendar answer. Neither answer came from the recalled memories; both are verbatim training outputs. Final train loss was about 0.11, and every plain example's instruction and output also appear in its grounded twin, so each target was seen twice per epoch (effectively 10 epochs on 315 rows). The adapter overfit into a lookup table.

**Decision:** keep v2 deployed. No arm beats it overall with significance, and v5 is significantly worse.

**What v5 rules out and what's next:** more SFT epochs on duplicated targets is the wrong lever. Next options, cheapest first:
1. Re-run E at 2 epochs and lr 1e-4, with plain and grounded copies split so no output appears twice. This tests the overfitting diagnosis directly.
2. Leave the weights alone and fix the retrieval miss upstream (A3: recall@5, hybrid search, recall-confidence in the prompt). 52% of hard failures start with a retrieval miss.
3. DPO with v2's fabrications as the rejected side, which targets abstention without new SFT targets to memorize.

## Retrieval A/B (A3, 2026-10-31)

52% of hard failures start with a retrieval miss, so this tests the retriever on its own. Four arms run against one frozen snapshot of the knowledge DB (25,759 active memories):
- **production:** the live keyword OR-merge. The offline replica matches live recall on 60/60 ids across 12 questions.
- **bm25:** one FTS5 OR query ranked by `bm25()`.
- **vector:** qwen3-embedding cosine, with an instruction prefix on queries only.
- **hybrid:** RRF (k=60) of bm25 and vector.

**Labels are silver.** For each of the 91 answerable personal questions, I pooled every arm's top 10 (2,325 pairs). A local qwen3:8b judge marked each pair relevant or not. A pair counted only if the judge said yes **and** the memory contains the question's required answer terms.

Spot check of 20 random positives: 14 clean, 3 borderline, 3 wrong. All 3 wrong ones belong to questions whose answer spec is empty or generic (a bare negation, "vault", no terms at all). The robustness row drops those 12 questions.

| Arm | hit@5 | MRR@10 | recall@5 | Δ hit@5 vs production |
|---|---|---|---|---|
| production | 0.70 | 0.56 | 0.24 | — |
| bm25 | 0.79 | 0.61 | 0.31 | +0.09 (−0.01, +0.21) |
| **vector** | **0.94** | **0.81** | **0.51** | **+0.24 (+0.14, +0.34)** |
| hybrid | 0.89 | 0.73 | 0.44 | +0.18 (+0.08, +0.29) |
| *robustness, n=75: vector* | 0.93 | 0.79 | 0.53 | +0.25 (+0.13, +0.37) |

n=87, paired bootstrap 95% CIs. Reproduce the per-domain tables with `score_retrieval.py`; the snapshot and labels stay private.

- **Vector wins, and the win holds on the strict subset.** A lenient judge would favor the on-topic memories that embeddings return. If that were the source of the lead, it would shrink once the weak specs are dropped. It doesn't.
- **Hybrid lands below vector.** The keyword side drags it down most on coding, where bm25 is worst (0.61 hit@5). The FTS OR query rewards notes that repeat "code" or "file" over the rule card that answers the question.
- **Identity gains the most:** production goes from 0.44 to 0.89 hit@5. These are the short questions ("Which church do I belong to?") where keyword AND-search fails.
- **Category-1 slice (n=10, underpowered):** production hits 0.30, vector 0.80, hybrid 1.00. The direction is clear, but the gap between vector and hybrid on 10 questions isn't evidence of anything.
- **4 questions are all-arm misses, not ingest gaps:** id-10, id-12, id-13 and wf-16. The facts are stored (for example, 7 memories mention Terminal Monitor and Gumroad), but no arm ranks them in its top 10 under the 22k SocialHub/Spector log rows. They're excluded, so every arm's absolute score is inflated by the same amount.
- **recall@5 is capped.** Questions average 5.7 relevant memories, so compare that column across arms, not against 1.0.
- **Serving cost:** a query embedding takes about 2.1 s warm (7 s cold) and 4.8 GB of VRAM. That's comparable to the whole 2.6 s p50 cortex answer, on the same 8 GB GPU, so Ollama will swap models. A smaller embedder (qwen3-embedding 0.6B) is the obvious next measurement before vector replaces production.

### Does better retrieval move answers? (2026-11-01)

Yes. Setup: the deployed v2 cortex on all 121 questions, 3 runs at t=0.7. `eval_system.py --retriever` switches the source of recalled memories, and both arms read the same snapshot, so the memories are the only thing that changes.

| Domain | production recall | vector recall | Δ (paired 95% CI) |
|---|---|---|---|
| **overall** | 68% | **75%** | **+6.6 (+0.3, +13.2), significant** |
| identity | 46% | 70% | **+23.8 (+7.9, +41.3), significant** |
| coding | 77% | 84% | +7.2 (n.s.) |
| workflow | 65% | 71% | +6.3 (n.s.) |
| faith | 85% | 87% | +2.6 (n.s.) |
| general | 56% | 59% | +2.6 (n.s.) |
| adversarial | 71% | 65% | −5.9 (n.s.) |

- **Always-failing questions drop from 28 to 20.** Fixed: id-3, id-4, id-8, id-9, id-14, id-21, wf-11, wf-15, coding-10, faith-9, faith-22, gen-8 and adv-14. Six of these are category-1 retrieval misses. Newly failing: adv-12, adv-17, faith-21, gen-12 and wf-3.
- **Adversarial slips, though not significantly.** Vector search always returns something on-topic, and the cortex treats it as evidence for a false premise. Keyword search returned nothing for those questions, which made it easier for the cortex to decline. Retrieval-confidence (a similarity floor, or a "no strong match" note in the prompt) is the matching fix.
- **The four all-arm misses still fail** (id-10, id-12, id-13 and wf-16). Their facts are buried under SocialHub/Terminal Monitor status logs, for example "Terminal Monitor auto-sync: 45 ports…". Down-weighting or excluding auto-generated log sources is the next retrieval change.
- **Latency:** recall p50 goes from ~0 s to 2.2 s. The cortex itself stays at 2.6 s, because the eval embeds every question up front. Live serving alternates the embedder and the cortex on one 8 GB GPU, so it will pay model-swap time on top.

**Decision:** vector retrieval is the better default on quality. Before switching brain_voice over, (1) measure a smaller embedder, (2) add the retrieval-confidence floor for false premises, and (3) filter log sources. Then re-run this eval.

## Caveats

- Scoring is regex-based, and grader agreement with human labels hasn't been measured yet. That's the next step: `calibrate_judge.py sample`, then hand-label 50 answers.
- All runs share one recall result per question (recall is deterministic). The variance shown is cortex sampling only.
