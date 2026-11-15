# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "matplotlib>=3.9", "psutil>=6.0", "torch>=2.6", "pillow>=11.0"]
# ///
# %% [markdown]
# # Lab 2.1 · Memory at three batch sizes
#
# Day 2 · Session 1 (Where memory goes) · 25 min.
#
# Lina wants to train Waraqa at batch 256 on a farm-office laptop "because larger batches are
# faster". Before anyone tries, you estimate what a training step holds in memory (weights,
# gradients, optimiser state and activations), then measure the peak at batch 16, 64 and 256
# and explain every difference.
#
# **Tasks**
# - [ ] Count weights, gradients and optimiser state · 5 min
# - [ ] Estimate activations by hand · 6 min
# - [ ] Measure peak memory at three batch sizes · 7 min
# - [ ] Explain each difference · 4 min
# - [ ] Predict batch 512 before running it · 3 min
# - [ ] ✱ Stretch · Swap AdamW for SGD with momentum and predict the change first
#
# AI policy for this lab: AI-off for the estimates (tasks 1, 2 and 5); AI-tutor for reading the
# measurements.

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
import torch.nn as nn
import torch.nn.functional as F
import nama as U

model = U.load_model("waraqa").train()     # the supplied checkpoint
P = U.n_params(model)
BATCHES = (16, 64, 256)
print(f"mode {U.MODE}: Waraqa has {P:,} parameters")

# %% [markdown]
# ## 1 · Count weights, gradients and optimiser state
#
# In float32 each number is 4 bytes. Training keeps the weights, one gradient per weight, and
# the optimiser's state. AdamW keeps two numbers per weight (a running mean of the gradient and
# of its square); SGD with momentum keeps one; plain SGD none.
#
# ✏️ **Exercise** · Return a dict of bytes: `weights`, `gradients`, `optimiser` for `n`
# parameters and an optimiser that keeps `state_per_param` numbers per weight.

# %%
def static_bytes(n, state_per_param=2, bytes_per_number=4):
    # BEGIN SOLUTION
    return {"weights": n * bytes_per_number, "gradients": n * bytes_per_number,
            "optimiser": n * state_per_param * bytes_per_number}
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    static = static_bytes(P)
except NotImplementedError:
    static = {"weights": 4 * P, "gradients": 4 * P, "optimiser": 8 * P}
assert static["optimiser"] == 8 * P
print({k: f"{v / 1e6:.2f} MB" for k, v in static.items()}, f"total {sum(static.values()) / 1e6:.2f} MB")

# %% [markdown]
# **Look at** the total: under 5 MB. For a model this small the weights are not the problem.
# For a language model of a billion parameters the same rule gives 16 GB before a single
# activation.

# %% [markdown]
# ## 2 · Estimate activations by hand
#
# The backward pass needs what the forward pass computed. Each convolution block of Waraqa
# (convolution, batch norm, ReLU) keeps about **two tensors the size of its output**: the batch
# norm's input and the ReLU's output (the next convolution reuses that as its input). Below are
# the output shapes of the six convolutions for one leaf.

# %%
shapes = []
hooks = [c.register_forward_hook(lambda m, i, o: shapes.append(tuple(o.shape[1:]))) for c, _ in U.conv_layers(model)]
with torch.no_grad():
    model(torch.zeros(1, 3, 64, 64))
for h in hooks:
    h.remove()
for s in shapes:
    print(s, f"{np.prod(s):,} numbers")

# %% [markdown]
# ✏️ **Exercise** · Return the estimated activation bytes of a batch: two tensors per block,
# 4 bytes per number, times the batch size.

# %%
def activation_bytes(shapes, batch, tensors_per_block=2):
    # BEGIN SOLUTION
    return batch * tensors_per_block * 4 * sum(int(np.prod(s)) for s in shapes)
    # END SOLUTION

# %%
try:
    est_act = {b: activation_bytes(shapes, b) for b in BATCHES}
except NotImplementedError:
    est_act = {b: b * 2 * 4 * sum(int(np.prod(s)) for s in shapes) for b in BATCHES}
for b in BATCHES:
    print(f"batch {b:4d}: activations about {est_act[b] / 1e6:7.1f} MB")

# %% [markdown]
# ## 3 · Measure peak memory at three batch sizes
#
# Two meters. `U.saved_activation_mb` counts exactly what autograd keeps for the backward pass
# (each storage once). `U.fresh_peak_mb` runs one full step (forward, backward, optimiser) in a
# fresh Python process and samples how much memory that process holds above its start; on a GPU
# it reads the framework's own peak counter. A fresh process, because memory freed by an earlier
# step often stays with the process and would hide part of the next one. The child takes one
# step of two leaves first, so the optimiser state exists before the measurement.

# %%
x_all, y_all, _ = U.leaves("val", 512)
rows = []
for b in BATCHES:
    peak = U.fresh_peak_mb(b)
    saved = U.saved_activation_mb(model, U.to_input(x_all[:b]), torch.from_numpy(y_all[:b]).long())
    rows.append({"batch": b, "estimate_mb": (est_act[b] + sum(static.values())) / 1e6,
                 "saved_activations_mb": saved, "peak_mb": peak})
print(f"{'batch':>6s} {'estimate':>9s} {'saved act.':>11s} {'measured peak':>14s}")
for r in rows:
    print(f"{r['batch']:6d} {r['estimate_mb']:9.1f} {r['saved_activations_mb']:11.1f} {r['peak_mb']:14.1f}")

# %% [markdown]
# ## 4 · Explain each difference
#
# Three gaps to explain, in your own words, in the cell below:
#
# - estimate against saved activations: what did "two tensors per block" leave out? (Hint: max
#   pooling keeps the positions of its maxima, as 8-byte integers.)
# - saved activations against the measured peak: what else is alive during the backward pass?
#   (Gradients of activations, the convolution's work buffers, the input batch.)
# - why the ratio between batch 256 and batch 16 is close to 16 for activations but not for the
#   whole process.

# %%
EXPLANATION = """
estimate vs saved: pooling indices (int64) and the input batch were not counted
saved vs peak: activation gradients and convolution workspaces live during backward
small batches: fixed costs (weights, optimiser state, allocator) dominate, so peak is not proportional
"""
ratio = rows[-1]["saved_activations_mb"] / rows[0]["saved_activations_mb"]
print(f"saved activations, batch 256 / batch 16 = {ratio:.1f}")
assert 12 < ratio < 20, ratio

# %% [markdown]
# ## 5 · Predict batch 512 before running it
#
# ✏️ **Exercise** · From your batch-256 measurement alone, predict the peak at batch 512
# (activations double; the static part does not). Then run it.

# %%
def predict_peak(peak_256, static_mb):
    # BEGIN SOLUTION
    return 2 * (peak_256 - static_mb) + static_mb
    # END SOLUTION

# %%
static_mb = sum(static.values()) / 1e6
try:
    guess = predict_peak(rows[-1]["peak_mb"], static_mb)
except NotImplementedError:
    guess = 2 * (rows[-1]["peak_mb"] - static_mb) + static_mb
peak512 = U.fresh_peak_mb(512)
print(f"batch 512: predicted {guess:.0f} MB, measured {peak512:.0f} MB ({peak512 / guess - 1:+.0%})")
est512 = 512 * 2 * 4 * sum(int(np.prod(s)) for s in shapes) + sum(static.values())
rows.append({"batch": 512, "estimate_mb": est512 / 1e6, "saved_activations_mb": None,
             "peak_mb": peak512, "predicted_mb": guess})
U.save_reference("d2_1_memory", {"static_bytes": static, "rows": rows})

# %% [markdown]
# **Look at** the farm-office laptop: 8 GB of memory, of which the operating system, a browser
# and the field app already use most. Which of these batch sizes would you dare to run there?
# Lab 2.3 fits a large batch into a fixed budget.

# %% [markdown]
# ## ✱ Stretch · SGD with momentum
#
# Predict the change in the static part if AdamW is replaced by SGD with momentum (one number
# per weight instead of two). Then measure. Is the change visible at batch 256? Why not?

# %% [markdown]
# ## Check yourself
#
# 1. A model of 1 billion parameters is trained in float32 with AdamW. How much memory do
#    weights, gradients and optimiser state take before any activation?
# 2. Doubling the batch doubled the peak at batch 256 but not at batch 16. Why?
# 3. Which part of training memory does activation checkpointing reduce?
#
# <details><summary>Answers</summary>
#
# 1. 16 bytes per parameter: 4 + 4 + 8, so about 16 GB.
# 2. At batch 256 activations dominate and scale with the batch; at batch 16 the fixed part
#    (weights, optimiser state, allocator) dominates.
# 3. Activations only: it keeps fewer and recomputes the rest during the backward pass.
# </details>
#
# ## Carry forward
#
# The estimate table next to the measured peaks is your LO3 evidence for the vision model
# (`artifacts/d2_1_memory.json`). Lab 3.2 does the same for the language model's cache.
