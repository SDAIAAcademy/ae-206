# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "matplotlib>=3.9", "psutil>=6.0", "torch>=2.6", "pillow>=11.0"]
# ///
# %% [markdown]
# # Lab 2.3 · A large batch in fixed memory
#
# Day 2 · Session 3 (Accumulation and checkpointing) · 25 min.
#
# Mutaib allows the monthly retraining to use at most **400 MB** of memory above the Python
# process at rest on a farm-office laptop. Lina's recipe uses an effective batch of 256. You
# fit it in the budget twice: with gradient accumulation and with activation checkpointing,
# and you check that the first does not change the gradient.
#
# **Tasks**
# - [ ] Batch 256 against the budget · 4 min
# - [ ] Accumulation: four micro-batches of 64 · 6 min
# - [ ] Check: the same gradient as one large batch · 5 min
# - [ ] Activation checkpointing · 6 min
# - [ ] Memory and time per method · 4 min
# - [ ] ✱ Stretch · Checkpoint two segments against four and find the cheapest that fits
#
# AI policy for this lab: AI-tutor.

# %%
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    REPO_URL = "https://github.com/SDAIAAcademy/ae-206.git"
    if not Path("/content/ae-206").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/ae-206"], check=True)
    os.chdir("/content/ae-206/labs")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
else:
    while not Path("_tools").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")  # solutions/ notebooks run from labs/
sys.path.insert(0, os.getcwd())

import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
import nama as U

BUDGET_MB = 400
BATCH = 256
x, y, _ = U.leaves("val", BATCH)
xb, yb = U.to_input(x), torch.from_numpy(y).long()
results = {}


def measure(name, fn, model, reps=U.pick(3, 2)):
    """Peak memory (second run) and median seconds of one optimiser step."""
    fn()
    _, peak = U.peak_rss_mb(fn)
    secs = [U.time_ms(fn, n=1, warmup=0)[0] / 1000 for _ in range(reps)]
    results[name] = {"peak_mb": peak, "s_per_step": float(np.median(secs)), "fits": peak <= BUDGET_MB}
    print(f"{name:32s} peak {peak:7.0f} MB  {'fits' if peak <= BUDGET_MB else 'OVER'}   {np.median(secs):.2f} s per step")


print(f"mode {U.MODE}: budget {BUDGET_MB} MB, effective batch {BATCH}")

# %% [markdown]
# ## 1 · Batch 256 against the budget

# %%
model = U.load_model("waraqa").train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)


def plain_step():
    loss = F.cross_entropy(model(xb), yb)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


measure("one batch of 256", plain_step, model)

# %% [markdown]
# ## 2 · Accumulation: four micro-batches of 64
#
# Run forward and backward on each micro-batch, letting the gradients add up in `.grad`, and
# step the optimiser once. Divide each micro-batch's loss by the number of micro-batches, so
# the sum equals the mean over the whole batch.
#
# ✏️ **Exercise** · Compute the accumulated gradients of `model` on `(xs, ys)` split into `k`
# micro-batches. Do not call the optimiser here.

# %%
def accumulate(model, xs, ys, k):
    # BEGIN SOLUTION
    for xm, ym in zip(xs.chunk(k), ys.chunk(k)):
        (F.cross_entropy(model(xm), ym) / k).backward()
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    accumulate(model, xb[:8], yb[:8], 2)
    model.zero_grad(set_to_none=True)
except NotImplementedError:
    def accumulate(model, xs, ys, k):
        for xm, ym in zip(xs.chunk(k), ys.chunk(k)):
            (F.cross_entropy(model(xm), ym) / k).backward()


def accum_step():
    accumulate(model, xb, yb, 4)
    opt.step()
    opt.zero_grad(set_to_none=True)


measure("accumulation, 4 x 64", accum_step, model)

# %% [markdown]
# ## 3 · Check: the same gradient as one large batch
#
# Batch norm computes its statistics per batch in training mode, so four micro-batches are
# normalised differently from one batch of 256: the gradients differ for that reason alone. In
# evaluation mode batch norm uses its stored statistics and the two must agree to rounding.

# %%
ref = U.load_model("waraqa").eval()
F.cross_entropy(ref(xb), yb).backward()
g_full = [p.grad.clone() for p in ref.parameters()]
ref.zero_grad(set_to_none=True)
accumulate(ref, xb, yb, 4)
g_acc = [p.grad.clone() for p in ref.parameters()]
rel = max(float((a - b).abs().max() / (b.abs().max() + 1e-12)) for a, b in zip(g_acc, g_full))
print(f"largest relative difference, evaluation mode: {rel:.2e}")
assert rel < 1e-4, rel

ref.train()
ref.zero_grad(set_to_none=True)
F.cross_entropy(ref(xb), yb).backward()
g_full_t = [p.grad.clone() for p in ref.parameters()]
ref.zero_grad(set_to_none=True)
accumulate(ref, xb, yb, 4)
g_acc_t = [p.grad.clone() for p in ref.parameters()]
rel_t = max(float((a - b).abs().max() / (b.abs().max() + 1e-12)) for a, b in zip(g_acc_t, g_full_t))
print(f"largest relative difference, training mode (batch norm per micro-batch): {rel_t:.2e}")

# %% [markdown]
# **Look at** the second number. Accumulation reproduces a large batch exactly only for layers
# that treat each example independently. With batch norm the micro-batch is what the layer
# sees. It usually trains fine; it is not the same computation.

# %% [markdown]
# ## 4 · Activation checkpointing
#
# Keep only the activations at a few boundaries during the forward pass and recompute the rest,
# segment by segment, during the backward pass. It trades compute for memory.
#
# ✏️ **Exercise** · Return Waraqa's logits computed with `checkpoint_sequential` over
# `model.features` in `segments` segments (`use_reentrant=False`), followed by the global
# average and the head, as in `Waraqa.forward`.

# %%
def forward_checkpointed(model, xs, segments):
    # BEGIN SOLUTION
    feats = checkpoint_sequential(model.features, segments, xs, use_reentrant=False)
    return model.head(feats.mean(dim=(2, 3)))
    # END SOLUTION

# %%
try:
    with torch.no_grad():
        forward_checkpointed(model, xb[:2], 2)
except NotImplementedError:
    def forward_checkpointed(model, xs, segments):
        return model.head(checkpoint_sequential(model.features, segments, xs, use_reentrant=False).mean(dim=(2, 3)))

model.eval()
with torch.no_grad():
    assert torch.allclose(forward_checkpointed(model, xb[:4], 4), model(xb[:4]), atol=1e-5)
model.train()


def ckpt_step():
    loss = F.cross_entropy(forward_checkpointed(model, xb, 4), yb)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


measure("checkpointing, 4 segments, 256", ckpt_step, model)

# %% [markdown]
# ## 5 · Memory and time per method

# %%
base = results["one batch of 256"]
print(f"{'method':32s} {'peak MB':>8s} {'fits':>5s} {'s/step':>7s} {'time x':>7s}")
for k, r in results.items():
    print(f"{k:32s} {r['peak_mb']:8.0f} {('yes' if r['fits'] else 'no'):>5s} {r['s_per_step']:7.2f} {r['s_per_step'] / base['s_per_step']:6.2f}x")
U.save_reference("d2_3_fit", {"budget_mb": BUDGET_MB, "results": results, "grad_rel_eval": rel, "grad_rel_train": rel_t})

# %% [markdown]
# **Look at** the price of each method. Accumulation pays in steps that run one after another
# (less parallel work per call); checkpointing pays with a second forward pass. Which would you
# pick for the monthly retraining, and would the choice change on a GPU with little memory?

# %% [markdown]
# ## ✱ Stretch · How many segments?
#
# Repeat step 4 with 2 and 8 segments. Fewer segments keep fewer boundaries but recompute
# longer stretches. Does any number of segments fit the budget at batch 256? Compute the size of
# the first block's activations for the answer, then try checkpointing on top of accumulation.

# %% [markdown]
# ## Check yourself
#
# 1. Why divide each micro-batch's loss by k?
# 2. Accumulation over 4 micro-batches takes the same memory as batch 64. Why not exactly?
# 3. Checkpointing reduced memory and the step took longer. Where did the extra time go?
#
# <details><summary>Answers</summary>
#
# 1. So that the summed gradient is the gradient of the mean loss over the whole batch, as
#    with one batch of 256; otherwise the step is k times too large.
# 2. Nearly: the gradients and the full input batch are also alive, and the allocator keeps
#    some memory between micro-batches.
# 3. To the recomputation of each segment's forward pass during the backward pass.
# </details>
#
# ## Carry forward
#
# Peak memory and time per method feed the LO4 results table (`artifacts/d2_3_fit.json`).
