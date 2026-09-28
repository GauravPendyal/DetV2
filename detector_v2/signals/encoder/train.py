"""Training loop for the DeBERTa-v3 claim-level encoder.

Kept separate from ``model.py`` so the inference wrapper carries no training deps
and so a tiny synthetic model can be trained in a unit test. The loop is a plain
AdamW fine-tune with a linear warmup schedule and optional class weighting; no
``Trainer``/``accelerate`` dependency.
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .model import DebertaClaimEncoder, DEFAULT_BACKBONE, Pair


def _set_seed(seed: int) -> None:
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _encode_batch(encoder: DebertaClaimEncoder, pairs: Sequence[Pair]):
    a = [(p[0] or "") for p in pairs]
    b = [(p[1] or "") for p in pairs]
    enc = encoder.tokenizer(
        a, b, truncation=True, max_length=encoder.max_len, padding=True, return_tensors="pt",
    )
    return {k: v.to(encoder.device) for k, v in enc.items()}


def train_encoder(
    pairs: Sequence[Pair],
    labels: Sequence[int],
    *,
    model_name: str = DEFAULT_BACKBONE,
    encoder: Optional[DebertaClaimEncoder] = None,
    epochs: int = 2,
    lr: float = 2e-5,
    batch_size: int = 16,
    max_len: int = 128,
    weight_decay: float = 0.01,
    warmup_frac: float = 0.1,
    class_weight: bool = True,
    seed: int = 13,
    log: Callable[[str], None] = print,
) -> DebertaClaimEncoder:
    """Fine-tune a fresh classification head on ``(query, claim)`` pairs with
    binary labels (1 = hallucinated). Returns the trained encoder."""
    import torch
    from torch.optim import AdamW

    _set_seed(seed)
    encoder = encoder or DebertaClaimEncoder.from_pretrained(model_name, max_len=max_len)
    model, device = encoder.model, encoder.device

    y = np.asarray(labels, dtype=np.int64)
    n = len(pairs)
    if class_weight:
        pos = max(int(y.sum()), 1)
        neg = max(n - pos, 1)
        w = torch.tensor([n / (2.0 * neg), n / (2.0 * pos)], dtype=torch.float32, device=device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    steps_per_epoch = math.ceil(n / batch_size)
    total_steps = max(steps_per_epoch * epochs, 1)
    warmup_steps = int(total_steps * warmup_frac)
    optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 1.0 - prog)

    rng = np.random.RandomState(seed)
    step = 0
    model.train()
    for epoch in range(epochs):
        order = rng.permutation(n)
        running = 0.0
        for bi in range(steps_per_epoch):
            idx = order[bi * batch_size:(bi + 1) * batch_size]
            if len(idx) == 0:
                continue
            batch_pairs = [pairs[i] for i in idx]
            enc = _encode_batch(encoder, batch_pairs)
            yb = torch.tensor(y[idx], device=device)
            for g in optim.param_groups:
                g["lr"] = lr * lr_at(step)
            optim.zero_grad()
            logits = model(**enc).logits
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running += float(loss.detach().cpu())
            step += 1
            if step % 50 == 0:
                log(f"  epoch {epoch + 1} step {step}/{total_steps} loss={running / (bi + 1):.4f}")
        log(f"  epoch {epoch + 1} done — mean loss {running / max(steps_per_epoch, 1):.4f}")

    model.eval()
    return encoder
