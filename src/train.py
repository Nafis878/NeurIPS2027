"""Experiments 3 & 4: fine-tuning with optional position-aware loss weighting.

The same `train` function does both: pass `weight_fn=None` for the ordinary
fine-tune (Experiment 3) and `weight_fn=middle_weight_fn(cfg)` for the
middle-weighted intervention (Experiment 4). Compute is matched because both use the
same data, epochs/steps, lr and batch size.
"""
from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional


def _example_loss(model, ex, device, loss_on: str):
    """Mean cross-entropy over the answer span (or full sequence) for one example."""
    import torch
    import torch.nn.functional as F

    full = ex["input_ids"] + ex["answer_ids"]
    ids = torch.tensor([full], device=device)
    logits = model(input_ids=ids).logits[0]  # [T, V]

    labels = torch.full((len(full),), -100, dtype=torch.long, device=device)
    if loss_on == "full":
        labels[1:] = ids[0, 1:]
    else:  # answer span only
        start = len(ex["input_ids"])
        for k in range(len(ex["answer_ids"])):
            labels[start + k] = full[start + k]

    shift_logits = logits[:-1, :]
    shift_labels = labels[1:]
    mask = shift_labels != -100
    if mask.sum() == 0:
        return logits.sum() * 0.0
    loss = F.cross_entropy(shift_logits[mask], shift_labels[mask])
    return loss


def train(model, tokenizer, examples, device, cfg,
          weight_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
          seed: int = 0, verbose: bool = True) -> List[Dict[str, Any]]:
    """Fine-tune in place. Returns per-step loss history."""
    import torch
    from src.utils import is_cuda_oom

    tcfg = cfg["train"]
    lr = float(tcfg["lr"])
    epochs = int(tcfg["epochs"])
    batch_size = int(tcfg["batch_size"])
    max_steps = tcfg.get("max_steps")
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    log_every = int(tcfg.get("log_every", 1))
    loss_on = tcfg.get("loss_on", "answer")

    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=lr,
                              weight_decay=float(tcfg.get("weight_decay", 0.0)))
    rng = random.Random(seed)
    history: List[Dict[str, Any]] = []
    step = 0
    stop = False

    for epoch in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            batch = [examples[j] for j in order[i:i + batch_size]]
            optim.zero_grad()
            try:
                weighted_sum = 0.0
                wnorm = 0.0
                plain_sum = 0.0
                for ex in batch:
                    loss = _example_loss(model, ex, device, loss_on)
                    w = float(weight_fn(ex)) if weight_fn else 1.0
                    weighted_sum = weighted_sum + w * loss
                    wnorm += w
                    plain_sum += loss.item()
                batch_loss = weighted_sum / max(wnorm, 1e-8)
                batch_loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optim.step()
            except RuntimeError as exc:
                if is_cuda_oom(exc):
                    print("[train] CUDA OOM during step; reduce train.batch_size in the config.")
                raise

            step += 1
            rec = {
                "step": step,
                "epoch": epoch,
                "loss": plain_sum / len(batch),          # unweighted (comparable across methods)
                "weighted_loss": float(batch_loss.item()),
            }
            history.append(rec)
            if verbose and step % log_every == 0:
                print(f"[train] step {step} epoch {epoch} loss {rec['loss']:.4f}")
            if max_steps is not None and step >= int(max_steps):
                stop = True
                break
        if stop:
            break

    model.eval()
    return history


def save_checkpoint(model, tokenizer, ckpt_dir: str) -> str:
    import os
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    return ckpt_dir
