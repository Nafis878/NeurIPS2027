"""Experiment 1: retrieval accuracy vs. position.

Primary metric: exact-match (EM) of a greedy-decoded answer.
Secondary (sensitivity) metric: teacher-forced mean log-prob and mean rank of the
gold answer tokens -- a continuous signal that stays informative even when a tiny
model scores ~0 EM everywhere.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List

_DIGITS = re.compile(r"\d+")


def _extract_code(text: str) -> str:
    m = _DIGITS.search(text)
    return m.group(0) if m else ""


def _batches(seq: List[Any], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _greedy_generate(model, tokenizer, batch, device, max_new_tokens, prev_oom=None):
    """Left-pad a batch of prompts and greedily decode `max_new_tokens`. Returns list[str]."""
    import torch

    tokenizer.padding_side = "left"
    seqs = [ex["input_ids"] for ex in batch]
    maxlen = max(len(s) for s in seqs)
    pad_id = tokenizer.pad_token_id
    input_ids, attn = [], []
    for s in seqs:
        pad = maxlen - len(s)
        input_ids.append([pad_id] * pad + s)
        attn.append([0] * pad + [1] * len(s))
    input_ids = torch.tensor(input_ids, device=device)
    attn = torch.tensor(attn, device=device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad_id,
        )
    gen = out[:, input_ids.shape[1]:]
    return [tokenizer.decode(g, skip_special_tokens=True) for g in gen]


def _answer_logprob_rank(model, tokenizer, ex, device):
    """Teacher-forced mean log-prob and mean rank of the gold answer tokens."""
    import torch
    import torch.nn.functional as F

    full = ex["input_ids"] + ex["answer_ids"]
    ids = torch.tensor([full], device=device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0]  # [T, V]
    logprobs = F.log_softmax(logits.float(), dim=-1)
    n_ans = len(ex["answer_ids"])
    start = len(ex["input_ids"])
    total_lp, ranks = 0.0, []
    for k in range(n_ans):
        pos = start + k - 1            # predicting token at index start+k
        tgt = full[start + k]
        lp = logprobs[pos, tgt].item()
        total_lp += lp
        rank = int((logprobs[pos] > logprobs[pos, tgt]).sum().item())
        ranks.append(rank)
    return total_lp / max(1, n_ans), sum(ranks) / max(1, len(ranks))


def run_eval(model, tokenizer, examples, device, cfg, verbose=True) -> Dict[str, Any]:
    """Run EM + logprob/rank eval. Returns {'per_example': [...], 'by_bucket': [...]}."""
    from src.utils import is_cuda_oom

    model.eval()
    max_new = int(cfg["eval"]["max_new_tokens"])
    batch_size = int(cfg["eval"]["batch_size"])

    per_example: List[Dict[str, Any]] = []
    done = 0
    for batch in _batches(examples, batch_size):
        # Generation with CUDA-OOM-safe batch halving.
        while True:
            try:
                preds = _greedy_generate(model, tokenizer, batch, device, max_new)
                break
            except RuntimeError as exc:
                if is_cuda_oom(exc) and len(batch) > 1:
                    half = max(1, len(batch) // 2)
                    print(f"[eval] CUDA OOM -> retrying with batch_size={half}")
                    import torch
                    torch.cuda.empty_cache()
                    # process recursively in halves
                    preds = []
                    for sub in _batches(batch, half):
                        preds.extend(_greedy_generate(model, tokenizer, sub, device, max_new))
                    break
                raise
        for ex, pred in zip(batch, preds):
            pred_code = _extract_code(pred)
            em = int(pred_code == ex["answer"])
            mean_lp, mean_rank = _answer_logprob_rank(model, tokenizer, ex, device)
            per_example.append({
                "id": ex["id"],
                "position_bucket": ex["position_bucket"],
                "norm_pos": ex["norm_pos"],
                "context_length": ex["context_length"],
                "answer": ex["answer"],
                "prediction_raw": pred.strip(),
                "prediction_code": pred_code,
                "exact_match": em,
                "answer_mean_logprob": mean_lp,
                "answer_mean_rank": mean_rank,
            })
        done += len(batch)
        if verbose:
            print(f"[eval] {done}/{len(examples)} examples")

    by_bucket = aggregate_by_bucket(per_example)
    return {"per_example": per_example, "by_bucket": by_bucket}


def aggregate_by_bucket(per_example: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in per_example:
        groups[r["position_bucket"]].append(r)
    rows = []
    for bucket in sorted(groups):
        g = groups[bucket]
        n = len(g)
        rows.append({
            "position_bucket": bucket,
            "n": n,
            "context_length": g[0]["context_length"],
            "mean_norm_pos": sum(r["norm_pos"] for r in g) / n,
            "accuracy": sum(r["exact_match"] for r in g) / n,
            "mean_logprob": sum(r["answer_mean_logprob"] for r in g) / n,
            "mean_rank": sum(r["answer_mean_rank"] for r in g) / n,
        })
    return rows
