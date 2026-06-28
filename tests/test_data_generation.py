"""Tests for the synthetic dataset generator and the EM scorer.

Uses the gpt2 tokenizer (small, fast). If it cannot be downloaded (offline),
the tokenizer-dependent tests are skipped rather than failing spuriously.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data, evaluate  # noqa: E402


@pytest.fixture(scope="module")
def tokenizer():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"tokenizer unavailable (offline?): {exc}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _cfg(ctx=128, n=2):
    return {
        "data": {
            "position_buckets": [0.02, 0.10, 0.50, 0.90, 0.98],
            "context_length": ctx,
            "examples_per_position": n,
        }
    }


def test_extract_code():
    assert evaluate._extract_code(" 73914 is the code") == "73914"
    assert evaluate._extract_code("no digits here") == ""
    assert evaluate._extract_code("abc 42 def 99") == "42"


def test_fact_position_matches_bucket(tokenizer):
    cfg = _cfg(ctx=256, n=3)
    examples = data.generate_dataset(cfg, tokenizer, seed=0)
    for ex in examples:
        # Realized position should track the requested bucket (small forward offset
        # from the fact prefix tokens is expected).
        assert ex["norm_pos"] >= ex["position_bucket"] - 0.06
        assert ex["norm_pos"] <= ex["position_bucket"] + 0.12


def test_answer_span_recoverable(tokenizer):
    cfg = _cfg(ctx=192, n=2)
    examples = data.generate_dataset(cfg, tokenizer, seed=1)
    for ex in examples:
        s, e = ex["fact_code_span"]
        decoded = tokenizer.decode(ex["input_ids"][s:e]).strip()
        assert decoded == ex["answer"]
        # answer is a 5-digit code
        assert len(ex["answer"]) == 5 and ex["answer"].isdigit()


def test_context_length_exact(tokenizer):
    cfg = _cfg(ctx=200, n=2)
    examples = data.generate_dataset(cfg, tokenizer, seed=2)
    for ex in examples:
        assert ex["context_token_len"] == 200
        assert ex["fact_code_span"][1] <= 200  # code lies inside the context region


def test_only_answer_has_digits(tokenizer):
    cfg = _cfg(ctx=160, n=2)
    examples = data.generate_dataset(cfg, tokenizer, seed=3)
    for ex in examples:
        ctx_ids = ex["input_ids"][:ex["context_token_len"]]
        ctx_text = tokenizer.decode(ctx_ids)
        digits = "".join(c for c in ctx_text if c.isdigit())
        assert digits == ex["answer"], "context should contain only the answer code's digits"


def test_determinism(tokenizer):
    cfg = _cfg(ctx=160, n=3)
    a = data.generate_dataset(cfg, tokenizer, seed=7)
    b = data.generate_dataset(cfg, tokenizer, seed=7)
    assert [x["input_ids"] for x in a] == [x["input_ids"] for x in b]
    assert [x["answer"] for x in a] == [x["answer"] for x in b]


def test_count(tokenizer):
    cfg = _cfg(ctx=128, n=4)
    examples = data.generate_dataset(cfg, tokenizer, seed=0)
    assert len(examples) == 4 * len(cfg["data"]["position_buckets"])
