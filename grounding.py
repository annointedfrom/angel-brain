"""grounding.py - the cortex prompt format shared by training and serving.

At serve time brain_voice.compose sends a system prompt plus a user turn of recalled
memories and the question. Training on bare questions taught the voice but never showed
the model what "the memories don't cover this" looks like, so it fabricated instead of
abstaining. Grounded training examples are rendered here, in the identical format.

Mirrors NeuroLinked/brain-voice/brain_voice.py (memory_block + compose). Keep the two in
sync; tests/test_grounding.py checks them against each other when brain_voice is present.
Pure stdlib.
"""
from __future__ import annotations

MEMORY_CHARS_EACH = 2000
MEMORY_CHARS_TOTAL = 12000
NO_MEMORIES = "(no relevant memories found)"


def memory_block(memories: list[str]) -> str:
    if not memories:
        return NO_MEMORIES
    lines, used = [], 0
    for m in memories:
        room = min(MEMORY_CHARS_EACH, MEMORY_CHARS_TOTAL - used)
        if room <= 0:
            break
        text = m if len(m) <= room else m[:room].rstrip() + " ..."
        lines.append(f"- {text}")
        used += len(text)
    return "\n".join(lines)


def user_turn(question: str, memories: list[str]) -> str:
    return f"=== Recalled memories ===\n{memory_block(memories)}\n\n=== Question ===\n{question}"


def chat_messages(ex: dict) -> list[dict]:
    """Messages for one training example.

    Grounded examples carry "memories" (list) and optionally "system"; plain voice examples
    keep the original instruction/input rendering so older datasets train unchanged.
    """
    msgs = []
    if ex.get("system"):
        msgs.append({"role": "system", "content": ex["system"]})
    if "memories" in ex:
        user = user_turn(ex.get("instruction", ""), ex["memories"])
    else:
        instr, inp = ex.get("instruction", ""), ex.get("input", "")
        user = f"{instr}\n\n{inp}".strip() if inp else instr
    msgs.append({"role": "user", "content": user})
    msgs.append({"role": "assistant", "content": ex.get("output", "")})
    return msgs
