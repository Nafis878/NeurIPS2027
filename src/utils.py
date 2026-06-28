"""Shared utilities: seeding, config loading, device selection, model loading, IO.

Everything here is deliberately dependency-light so it imports fast and runs the
same on a CPU laptop (smoke tests) and a Colab T4 (real runs).
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Repo paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo_path(*parts: str) -> str:
    return os.path.join(REPO_ROOT, *parts)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Config handling
# ---------------------------------------------------------------------------
def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = deepcopy(val)
    return out


def load_config(path: str, smoke: bool = False) -> Dict[str, Any]:
    """Load a YAML config; if ``smoke`` deep-merge the ``smoke:`` block on top."""
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if cfg is None:
        raise ValueError(f"Config at {path} is empty.")
    smoke_block = cfg.pop("smoke", {}) or {}
    if smoke:
        cfg = deep_merge(cfg, smoke_block)
    cfg["_meta"] = {"config_path": os.path.abspath(path), "smoke": smoke}
    return cfg


def apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply non-None CLI overrides to known nested keys. Unknown keys raise."""
    routes = {
        "examples_per_position": ("data", "examples_per_position"),
        "context_length": ("data", "context_length"),
        "model": ("model", "name"),
        "seed": ("seed",),
        "epochs": ("train", "epochs"),
        "max_steps": ("train", "max_steps"),
        "lr": ("train", "lr"),
        "max_examples": ("influence", "max_examples"),
    }
    cfg = deepcopy(cfg)
    for flag, value in overrides.items():
        if value is None:
            continue
        if flag not in routes:
            raise KeyError(f"Unknown override '{flag}'")
        path = routes[flag]
        node = cfg
        for part in path[:-1]:
            node = node[part]
        node[path[-1]] = value
    return cfg


def save_run_config(cfg: Dict[str, Any], out_dir: str, name: str = "run_config.json") -> str:
    ensure_dir(out_dir)
    meta = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": sys.argv,
        "python": sys.version.split()[0],
    }
    payload = {"config": cfg, "run_meta": meta}
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# Device + model loading
# ---------------------------------------------------------------------------
def pick_device(verbose: bool = True):
    import torch

    if torch.cuda.is_available():
        dev = "cuda"
        if verbose:
            name = torch.cuda.get_device_name(0)
            print(f"[device] CUDA available -> using GPU: {name}")
    else:
        dev = "cpu"
        if verbose:
            print("[device] CUDA NOT available -> using CPU (smoke-scale only).")
    return torch.device(dev)


def resolve_dtype(dtype_str: str, device) -> "object":
    import torch

    if dtype_str == "float16" and device.type == "cuda":
        return torch.float16
    if dtype_str == "float16":
        print("[dtype] float16 requested but not on CUDA; falling back to float32.")
    return torch.float32


def load_tokenizer(cfg: Dict[str, Any]):
    """Load just the tokenizer (with fallback). Returns (tokenizer, name)."""
    from transformers import AutoTokenizer

    name = cfg["model"]["name"]
    fallback = cfg["model"].get("fallback")
    try:
        tok = AutoTokenizer.from_pretrained(name)
        loaded = name
    except Exception as exc:  # noqa: BLE001
        if not fallback:
            raise
        print(f"[tokenizer] Failed to load '{name}' ({exc!r}); falling back to '{fallback}'.")
        tok = AutoTokenizer.from_pretrained(fallback)
        loaded = fallback
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, loaded


def load_model_tokenizer(cfg: Dict[str, Any], device, for_training: bool = False,
                         init_random: bool = False, seed: Optional[int] = None):
    """Load model + tokenizer with automatic fallback. Returns (model, tokenizer, name).

    If ``init_random`` is True the model is built fresh from the architecture config with
    *random* weights (no pretrained checkpoint) -- this is the "Step 0" / initialization
    state used to measure the architectural position-bias prior. The tokenizer is still
    taken from the pretrained name so token ids are meaningful. ``seed`` makes the random
    initialization reproducible.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    name = cfg["model"]["name"]
    fallback = cfg["model"].get("fallback")
    dtype = resolve_dtype(cfg["model"].get("dtype", "float32"), device)

    def _load(model_name: str):
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        if init_random:
            if seed is not None:
                torch.manual_seed(seed)
            config = AutoConfig.from_pretrained(model_name)
            mdl = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
        else:
            mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        mdl.to(device)
        return mdl, tok

    try:
        model, tokenizer = _load(name)
        loaded = name
    except Exception as exc:  # noqa: BLE001 - we want to surface and fall back
        if not fallback:
            raise
        print(f"[model] Failed to load '{name}' ({exc!r}); falling back to '{fallback}'.")
        model, tokenizer = _load(fallback)
        loaded = fallback

    model.train(for_training)
    tag = "RANDOM-INIT (Step 0)" if init_random else "pretrained"
    print(f"[model] Loaded '{loaded}' [{tag}] | params={sum(p.numel() for p in model.parameters()):,} "
          f"| dtype={dtype} | device={device}")
    return model, tokenizer, loaded


def is_cuda_oom(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower()


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> str:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=_json_default) + "\n")
    return path


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> str:
    ensure_dir(os.path.dirname(path) or ".")
    if not rows:
        # still write a header-less empty file so downstream code can detect it
        open(path, "w", encoding="utf-8").close()
        return path
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    return path


def write_json(path: str, obj: Any) -> str:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    return path


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)
