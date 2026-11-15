# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "pillow>=11.0", "psutil>=6.0", "torch>=2.6"]
# ///
"""Lina Al-Mohaisen's training script for Waraqa, as she handed it over: it works, and it is slow.

    uv run lina_train.py --steps 20          (from labs/; builds the PNG field export on first use)

It reads the field app's 224-pixel PNG export, rotates each leaf at full size, shrinks it to 64
pixels and trains in the main process. Labs 1.3 and 1.4 profile and fix it; the functions below
are what they import.
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

import nama as U


def make_loader(folder=None, batch: int = 64, workers: int = 0, dataset=None, seed: int = 0):
    """Lina's loader: the slow dataset, shuffled, in the main process unless workers > 0."""
    ds = dataset if dataset is not None else U.FieldExportSlow(folder or U.field_export())
    g = torch.Generator().manual_seed(seed)
    kw = {"num_workers": workers, "batch_size": batch, "shuffle": True, "generator": g, "drop_last": True}
    if workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=2)
    return torch.utils.data.DataLoader(ds, **kw)


def phase_step(model, opt, batches) -> dict:
    """One training step with a timer around each phase; returns seconds per phase."""
    t0 = time.perf_counter()
    x, y = next(batches)
    t1 = time.perf_counter()
    loss = F.cross_entropy(model(x), y)
    t2 = time.perf_counter()
    loss.backward()
    t3 = time.perf_counter()
    opt.step()
    opt.zero_grad(set_to_none=True)
    t4 = time.perf_counter()
    return {"loading": t1 - t0, "forward": t2 - t1, "backward": t3 - t2, "optimiser": t4 - t3}


def run(loader, steps: int, skip: int = 2, model=None):
    """Train for skip + steps steps; return the phase times of the last `steps` (the first ones
    pay for start-up: worker processes, caches, first allocations)."""
    model = model or U.load_model("waraqa_warm").train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    it = iter(loader)
    out = []
    for i in range(skip + steps):
        try:
            p = phase_step(model, opt, it)
        except StopIteration:
            it = iter(loader)
            p = phase_step(model, opt, it)
        if i >= skip:
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    t = time.perf_counter()
    phases = run(make_loader(), a.steps)
    total = sum(sum(p.values()) for p in phases)
    print(f"{a.steps} steps in {total:.1f} s ({total / a.steps * 1000:.0f} ms per step), wall {time.perf_counter() - t:.1f} s")


if __name__ == "__main__":
    main()
