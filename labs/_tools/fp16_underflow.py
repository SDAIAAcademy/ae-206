"""Figures for the float16 underflow slides of Day 2 (fp16-underflow-signs, loss-scaling) and the bank.

    cd labs && uv run python _tools/fp16_underflow.py

One backward pass of the finished Hasad model on 32 farmer questions (loss on the answer only,
as in its training), run three ways on the same batch:
    float32                      the reference gradients
    float16 autocast, scale S    the loss multiplied by S before backward, the gradients
                                 divided by S afterwards (what a gradient scaler does)
float16's smallest positive number is 2**-24 (about 6e-8); values below half of it round to zero,
and the largest finite number is 65504. For each scale it counts, per layer group, the gradient
entries that are nonzero in float32 and exactly zero in float16 ("lost"), the entries that are
infinite or NaN, and the relative error of the whole gradient against float32.
Writes data/reference/d2_fp16_underflow.json and prints the tables the slides quote.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))
import nama as U  # noqa: E402

SCALES = [1, 2 ** 8, 2 ** 16, 2 ** 24, 2 ** 32]
GROUPS = ["embeddings", "attention", "feed-forward", "norms"]


def group_of(name: str) -> str:
    if name.startswith(("tok.", "pos.")):
        return "embeddings"
    if ".qkv." in name or ".proj." in name:
        return "attention"
    if ".fc1." in name or ".fc2." in name:
        return "feed-forward"
    return "norms"


def batch(model, n: int = 32, seed: int = 206):
    tk = model.tokenizer
    qs = U.farmer_questions()
    pick = np.random.default_rng(seed).choice(len(qs), n, replace=False)
    seqs = [(tk.encode(U.prompt_for(qs[i]["question"])), tk.encode(qs[i]["answer"] + "\n")) for i in pick]
    T = max(len(p) + len(a) for p, a in seqs)
    X = torch.zeros(n, T, dtype=torch.long)
    Y = torch.full((n, T), -100, dtype=torch.long)
    for i, (p, a) in enumerate(seqs):
        s = p + a
        X[i, :len(s)] = torch.tensor(s)
        Y[i, len(p) - 1:len(s) - 1] = torch.tensor(s[len(p):])
    return X, Y


def gradients(dtype, scale: float) -> tuple[float, dict]:
    """Loss and per-parameter gradients (divided back by the scale) of one backward pass."""
    model = U.load_hasad(32)
    X, Y = batch(model)
    model.train()
    ctx = torch.autocast("cpu", dtype=dtype) if dtype else U._null()
    with ctx:
        logits = model(X)
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), Y.reshape(-1), ignore_index=-100)
    (loss * scale).backward()
    return float(loss.detach()), {n: (p.grad.double() / scale).flatten() for n, p in model.named_parameters()}


def main():
    loss32, g32 = gradients(None, 1)
    ref = torch.cat(list(g32.values()))
    out = {"model": "hasad_32", "questions": 32, "loss_float32": loss32, "runs": []}
    for s in SCALES:
        loss16, g16 = gradients(torch.float16, s)
        groups = {k: {"entries": 0, "lost": 0} for k in GROUPS}
        bad = 0
        for n, a in g32.items():
            b = g16[n]
            G = groups[group_of(n)]
            G["entries"] += int((a != 0).sum())
            G["lost"] += int(((a != 0) & (b == 0)).sum())
            bad += int((~torch.isfinite(b)).sum())
        g = torch.cat(list(g16.values()))
        rel = float((g - ref).norm() / ref.norm()) if bad == 0 else None
        lost = sum(G["lost"] for G in groups.values())
        entries = sum(G["entries"] for G in groups.values())
        out["runs"].append({"scale": s, "loss_float16": loss16, "lost": lost, "entries": entries,
                            "inf_or_nan": bad, "relative_error": rel, "groups": groups})
    print(f"Hasad, one backward pass on 32 questions: loss float32 {loss32:.6f}")
    print(f"{'scale':>12s} {'loss fp16':>10s} {'lost %':>7s} {'inf/NaN':>8s} {'rel err %':>9s}  "
          + "  ".join(f"{k} lost %" for k in GROUPS))
    for r in out["runs"]:
        rel = "-" if r["relative_error"] is None else f"{100 * r['relative_error']:.2f}"
        per = "  ".join(f"{100 * r['groups'][k]['lost'] / r['groups'][k]['entries']:{len(k) + 7}.2f}" for k in GROUPS)
        print(f"{r['scale']:12d} {r['loss_float16']:10.6f} {100 * r['lost'] / r['entries']:7.2f} "
              f"{r['inf_or_nan']:8d} {rel:>9s}  {per}")
    print("entries per group:", {k: out["runs"][0]["groups"][k]["entries"] for k in GROUPS})
    path = LABS / "data" / "reference" / "d2_fp16_underflow.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
