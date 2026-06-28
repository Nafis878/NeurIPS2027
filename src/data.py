"""Synthetic position-controlled retrieval dataset.

Each example is a long *digit-free* distractor context with exactly one planted
answer-critical fact ("The access code for Project Vega is 73914.") inserted at a
controlled normalized token position, followed by a question and a short exact answer.

Because the only number in the whole context is the answer code, exact-match retrieval
is a clean signal. Everything is token-precise and deterministic under a fixed seed.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List

# Digit-free vocabulary for distractor sentences.
_ADJ = ["quiet", "ancient", "distant", "golden", "weary", "restless", "hollow",
        "gentle", "silver", "narrow", "bright", "humble", "rugged", "solemn",
        "vivid", "frozen", "amber", "crimson", "stoic", "pale"]
_NOUN = ["harbor", "meadow", "lantern", "courier", "orchard", "ridge", "scholar",
         "engineer", "mariner", "garden", "foundry", "archive", "willow", "beacon",
         "tinker", "merchant", "valley", "cathedral", "printer", "weaver"]
_VERB = ["wandered", "lingered", "drifted", "gathered", "echoed", "settled",
         "vanished", "returned", "labored", "rested", "shimmered", "whispered",
         "circled", "departed", "remained", "stirred", "paused", "arrived"]
_PLACE = ["the old market", "a windswept cliff", "the river bend", "the eastern gate",
          "a copper mine", "the council hall", "a forgotten chapel", "the train yard",
          "the southern pier", "a mountain pass", "the city square", "the glass tower"]
_TIME = ["dawn", "dusk", "midsummer", "the long winter", "the harvest", "the festival",
         "an autumn rain", "the early frost", "a still evening", "the spring thaw"]

_PROJECT_NAMES = ["Vega", "Orion", "Lyra", "Draco", "Cygnus", "Aquila", "Perseus",
                  "Phoenix", "Hydra", "Corvus", "Pegasus", "Andromeda", "Cetus",
                  "Lynx", "Tucana", "Carina", "Volans", "Mensa", "Pictor", "Norma"]


def _sentence(rng: random.Random) -> str:
    return (f"The {rng.choice(_ADJ)} {rng.choice(_NOUN)} {rng.choice(_VERB)} near "
            f"{rng.choice(_PLACE)} during {rng.choice(_TIME)}.")


def _build_distractor_ids(tokenizer, rng: random.Random, min_tokens: int) -> List[int]:
    """Generate digit-free sentences and encode until we have >= min_tokens token ids."""
    ids: List[int] = []
    guard = 0
    while len(ids) < min_tokens:
        text = " ".join(_sentence(rng) for _ in range(8))
        ids.extend(tokenizer.encode(" " + text, add_special_tokens=False))
        guard += 1
        if guard > 100000:
            raise RuntimeError("Distractor generation did not reach target length.")
    return ids


def _fact_segments(tokenizer, name: str, code: str):
    """Return (prefix_ids, code_ids, suffix_ids) so the code span is unambiguous."""
    prefix = tokenizer.encode(f" The access code for Project {name} is",
                              add_special_tokens=False)
    code_ids = tokenizer.encode(f" {code}", add_special_tokens=False)
    suffix = tokenizer.encode(".", add_special_tokens=False)
    return prefix, code_ids, suffix


def generate_dataset(cfg: Dict[str, Any], tokenizer, seed: int) -> List[Dict[str, Any]]:
    """Generate the full dataset (list of example dicts)."""
    rng = random.Random(seed)
    buckets = cfg["data"]["position_buckets"]
    ctx_len = int(cfg["data"]["context_length"])
    n_per = int(cfg["data"]["examples_per_position"])

    examples: List[Dict[str, Any]] = []
    ex_id = 0
    for bucket in buckets:
        for _ in range(n_per):
            name = rng.choice(_PROJECT_NAMES)
            code = "".join(rng.choice("0123456789") for _ in range(5))
            prefix_ids, code_ids, suffix_ids = _fact_segments(tokenizer, name, code)
            fact_ids = prefix_ids + code_ids + suffix_ids
            fact_len = len(fact_ids)
            if fact_len >= ctx_len:
                raise ValueError(
                    f"context_length={ctx_len} too small for the fact ({fact_len} tokens).")

            # Token-precise insertion so the realized position matches the bucket.
            slots = ctx_len - fact_len
            insert_idx = int(round(bucket * slots))
            insert_idx = max(0, min(insert_idx, slots))

            distractor = _build_distractor_ids(tokenizer, rng, ctx_len)
            before = distractor[:insert_idx]
            after = distractor[insert_idx:slots]
            context_ids = before + fact_ids + after  # length == ctx_len

            # Location of the planted code within the context.
            code_start = len(before) + len(prefix_ids)
            code_end = code_start + len(code_ids)
            norm_pos = code_start / max(1, (ctx_len - 1))

            # Build the prompt = context + question scaffold.
            question = f"What is the access code for Project {name}?"
            tail_text = f"\n\nQuestion: {question}\nAnswer:"
            tail_ids = tokenizer.encode(tail_text, add_special_tokens=False)
            input_ids = context_ids + tail_ids  # prompt (no answer yet)

            answer_text = code
            answer_ids = tokenizer.encode(f" {code}", add_special_tokens=False)

            examples.append({
                "id": ex_id,
                "position_bucket": bucket,
                "norm_pos": norm_pos,
                "context_length": ctx_len,
                "project": name,
                "answer": answer_text,
                "question": question,
                "prompt_text": tokenizer.decode(input_ids),
                "input_ids": input_ids,           # prompt only
                "answer_ids": answer_ids,
                "context_token_len": ctx_len,
                "fact_code_span": [code_start, code_end],  # within input_ids / context
            })
            ex_id += 1
    return examples


def dataset_filename(cfg: Dict[str, Any]) -> str:
    ctx = int(cfg["data"]["context_length"])
    n = int(cfg["data"]["examples_per_position"])
    return f"dataset_ctx{ctx}_n{n}.jsonl"
