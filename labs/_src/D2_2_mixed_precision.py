# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "matplotlib>=3.9", "psutil>=6.0", "torch>=2.6", "pillow>=11.0"]
# ///
# %% [markdown]
# # Lab 2.2 · Reduced precision
#
# Day 2 · Session 2 (Mixed precision) · 25 min.
#
# A blog post tells Lina that "mixed precision makes training twice as fast". You test the
# claim on the machine you have: the same training steps from the same warm-start checkpoint,
# once in 32-bit floats and once with bfloat16 autocast, then time per step and validation
# accuracy side by side.
#
# **Tasks**
# - [ ] What this processor supports · 3 min
# - [ ] Train in 32-bit for a fixed number of steps · 6 min
# - [ ] Train the same steps with bfloat16 autocast · 6 min
# - [ ] Validation accuracy of both · 5 min
# - [ ] The results row for each setting · 5 min
# - [ ] ✱ Stretch · On a Colab GPU, float16 autocast with a gradient scaler
#
# AI policy for this lab: AI-tutor. An assistant may explain autocast; the verdict on the claim
# is yours, from your numbers.

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

import numpy as np
import torch
import nama as U

x_tr, y_tr, _ = U.leaves("train", U.pick(4000, 1500))
x_val, y_val, _ = U.leaves("val", U.pick(2000, 600))
STEPS = U.pick(12, 6)
results = {}
print(f"mode {U.MODE}: {STEPS} steps of 64 leaves from the warm-start checkpoint, {U.THREADS} threads")

# %% [markdown]
# ## 1 · What this processor supports
#
# Reduced precision is fast only where the hardware has instructions for it. PyTorch reports
# the widest vector instruction set it uses on this processor. Native bfloat16 arithmetic on
# x86 processors comes with the AVX-512 BF16 or AMX extensions; without them, bfloat16 is
# converted to float32 for the arithmetic and back.

# %%
cap = torch.backends.cpu.get_cpu_capability()
print("vector capability:", cap)
print("machine:", U.machine()["cpu"])
print("CUDA GPU:", torch.cuda.is_available())

# %% [markdown]
# **Look at** your capability and write down a prediction before step 2: faster, the same or
# slower with bfloat16?

# %% [markdown]
# ## 2 · Train in 32-bit
#
# `U.train_steps` trains for a number of optimiser steps on random batches (fixed seed) and
# returns seconds per step. Both runs start from the same checkpoint and see the same batches.

# %%
m32 = U.load_model("waraqa_warm")
t32 = U.train_steps(m32, x_tr, y_tr, STEPS, seed=7)
print(f"float32: median {np.median(t32[2:]) * 1000:.0f} ms per step")

# %% [markdown]
# ## 3 · The same steps with bfloat16 autocast
#
# `torch.autocast` runs selected operations (convolutions, matrix products) in the lower
# precision and keeps the rest (losses, reductions, batch norm statistics) in float32. The
# weights and the optimiser stay in float32: that is the "mixed" in mixed precision.
#
# ✏️ **Exercise** · Train a fresh copy of the warm-start model for the same steps and seed with
# `autocast_dtype=torch.bfloat16`. Return the model and the list of seconds per step.

# %%
def train_bf16(steps):
    # BEGIN SOLUTION
    m = U.load_model("waraqa_warm")
    return m, U.train_steps(m, x_tr, y_tr, steps, seed=7, autocast_dtype=torch.bfloat16)
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    m16, t16 = train_bf16(STEPS)
except NotImplementedError:
    m16 = U.load_model("waraqa_warm")
    t16 = U.train_steps(m16, x_tr, y_tr, STEPS, seed=7, autocast_dtype=torch.bfloat16)
assert len(t16) == STEPS
print(f"bfloat16 autocast: median {np.median(t16[2:]) * 1000:.0f} ms per step")
print(f"speed-up: {np.median(t32[2:]) / np.median(t16[2:]):.2f} x")

# %% [markdown]
# ## 4 · Validation accuracy of both
#
# A speed-up is accepted only with its quality. Evaluate both models, in float32, on the same
# validation leaves.
#
# ✏️ **Exercise** · Return the accuracy of a model on `x_val`, `y_val`.

# %%
def val_accuracy(model):
    # BEGIN SOLUTION
    return U.accuracy(model, x_val, y_val)
    # END SOLUTION

# %%
try:
    acc32, acc16 = val_accuracy(m32), val_accuracy(m16)
except NotImplementedError:
    acc32, acc16 = U.accuracy(m32, x_val, y_val), U.accuracy(m16, x_val, y_val)
start = U.accuracy(U.load_model("waraqa_warm"), x_val, y_val)
print(f"validation accuracy: start {start:.3f}, float32 {acc32:.3f}, bfloat16 {acc16:.3f}")

# %% [markdown]
# **Look at** the difference between the two accuracies against the number of validation
# leaves. With 600 leaves one leaf is 0.17 points: a difference of one or two leaves is noise.
# Day 4 puts an interval on such differences.

# %% [markdown]
# ## 5 · The results row for each setting

# %%
for name, t, acc, dt in (("float32", t32, acc32, None), ("bfloat16 autocast", t16, acc16, torch.bfloat16)):
    ms = float(np.median(t[2:]) * 1000)
    m = U.load_model("waraqa_warm")
    U.train_steps(m, x_tr, y_tr, 1, autocast_dtype=dt)            # optimiser state exists before measuring
    _, peak = U.peak_rss_mb(lambda: U.train_steps(m, x_tr, y_tr, 1, autocast_dtype=dt))
    results[name] = {"ms_per_step": ms, "leaves_per_s": 64 / ms * 1000, "peak_mb": peak, "val_accuracy": acc}
print(f"{'setting':20s} {'ms/step':>8s} {'leaves/s':>9s} {'peak MB':>8s} {'val acc':>8s}")
for k, r in results.items():
    print(f"{k:20s} {r['ms_per_step']:8.0f} {r['leaves_per_s']:9.0f} {r['peak_mb']:8.0f} {r['val_accuracy']:8.3f}")
U.save_reference("d2_2_precision", {"capability": cap, "results": results})

# %% [markdown]
# **Look at** your verdict on the blog post's claim. Write one sentence: on which hardware the
# claim holds, and what you measured here.

# %% [markdown]
# ## ✱ Stretch · float16 on a GPU
#
# In Colab, choose a GPU runtime and run the cell below. float16 has a narrow range, so small
# gradients underflow to zero; a gradient scaler multiplies the loss before the backward pass
# and divides the gradients before the step.

# %%
if torch.cuda.is_available():
    dev = torch.device("cuda")
    g = U.load_model("waraqa_warm").to(dev)
    opt = torch.optim.AdamW(g.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    xb, yb = U.to_input(x_tr[:64]).to(dev), torch.from_numpy(y_tr[:64]).long().to(dev)
    g.train()
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.cross_entropy(g(xb), yb)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    print("float16 with a gradient scaler ran; now time it as in step 3")
else:
    print("no GPU here: run this cell in a Colab GPU runtime")

# %% [markdown]
# ## Check yourself
#
# 1. bfloat16 and float16 both use 16 bits. What does each give up?
# 2. Why are the weights kept in float32 during mixed-precision training?
# 3. bfloat16 was slower on your laptop. Is mixed precision then useless for Nama?
#
# <details><summary>Answers</summary>
#
# 1. float16 keeps more precision (10 fraction bits) and a small range; bfloat16 keeps the
#    range of float32 (8 exponent bits) and only 7 fraction bits.
# 2. Small updates added to a large weight vanish in 16 bits; the float32 master copy keeps
#    them.
# 3. No: it halves the bytes of every activation it touches, and on hardware with native
#    support it is faster. On this laptop it buys nothing for speed; that is a measurement, not
#    a rule.
# </details>
#
# ## Carry forward
#
# The two rows go into the LO4 results table that Lab 2.4 completes (`artifacts/d2_2_precision.json`).
