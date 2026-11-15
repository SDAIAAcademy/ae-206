# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "matplotlib>=3.9", "psutil>=6.0", "torch>=2.6", "pillow>=11.0"]
# ///
# %% [markdown]
# # Lab 1.3 · Profile Lina's training step
#
# Day 1 · Session 3 (Profiling a training step) · 25 min.
#
# Lina Al-Mohaisen retrains Waraqa every month on the leaves the field app uploads. Her script,
# `lina_train.py`, reads the app's 224-pixel PNG export and "takes all night". Before anyone
# buys a GPU, you find out where a step's time goes: data loading, forward pass, backward pass
# or optimiser.
#
# **Tasks**
# - [ ] Run the slow script and time its first steps · 4 min
# - [ ] Time each phase with timers · 6 min
# - [ ] Profile one step with the framework profiler · 6 min
# - [ ] Rank the four phases · 5 min
# - [ ] Name the largest bottleneck · 4 min
# - [ ] ✱ Stretch · Record a flame graph of the same script with py-spy
#
# AI policy for this lab: AI-tutor. An assistant may explain a profiler row; the ranking and the
# diagnosis are yours.

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
import matplotlib.pyplot as plt
import nama as U
import lina_train as L

t0 = time.perf_counter()
folder = U.field_export()          # builds the PNG export on first use (about a minute)
print(f"mode {U.MODE}: field export {folder.name}, ready in {time.perf_counter() - t0:.0f} s")
STEPS = U.pick(12, 6)

# %% [markdown]
# ## 1 · Run the slow script and time its first steps
#
# `L.run(loader, steps)` trains Waraqa from its warm-start checkpoint and returns, for each
# step, the seconds of each phase. It skips the first two steps, which pay for start-up.

# %%
phases = L.run(L.make_loader(folder), STEPS)
step_ms = [sum(p.values()) * 1000 for p in phases]
print(f"{STEPS} steps of 64 leaves: median {np.median(step_ms):.0f} ms per step")
print(f"one epoch of 16,000 leaves would take about {np.median(step_ms) / 1000 * 16000 / 64 / 60:.1f} min")

# %% [markdown]
# ## 2 · Time each phase with timers
#
# `L.phase_step` already wraps each phase in `time.perf_counter()`. Read it (`L.phase_step??`
# in Jupyter). On a CPU the timers are honest. On a GPU they are not: the GPU runs
# asynchronously, so a timer must call `torch.cuda.synchronize()` before reading the clock.
#
# ✏️ **Exercise** · Return the median milliseconds of each phase over all steps, as a dict.

# %%
def median_phases(phases):
    # BEGIN SOLUTION
    return {k: float(np.median([p[k] for p in phases]) * 1000) for k in phases[0]}
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    med = median_phases(phases)
except NotImplementedError:
    med = {k: float(np.median([p[k] for p in phases]) * 1000) for k in phases[0]}
assert set(med) == {"loading", "forward", "backward", "optimiser"}
for k, v in med.items():
    print(f"{k:10s} {v:8.1f} ms")

# %% [markdown]
# ## 3 · Profile one step with the framework profiler
#
# Timers say how long; a profiler says what ran. `torch.profiler` records every operator of
# the step. Profile one full step, loading included, and print the ten rows with the largest
# **self** time (time in the operator itself, not in what it calls).

# %%
from torch.profiler import profile, record_function, ProfilerActivity

model = U.load_model("waraqa_warm").train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
it = iter(L.make_loader(folder, seed=1))
L.phase_step(model, opt, it)   # warm-up
with profile(activities=[ProfilerActivity.CPU]) as prof:
    with record_function("loading"):
        xb, yb = next(it)
    with record_function("forward"):
        loss = F.cross_entropy(model(xb), yb)
    with record_function("backward"):
        loss.backward()
    with record_function("optimiser"):
        opt.step()
        opt.zero_grad(set_to_none=True)
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

# %% [markdown]
# **Look at** where the loading time is. The data loader's work (PNG decoding, rotation,
# resizing) is Python and Pillow, not PyTorch operators, so it appears as one opaque row,
# `enumerate(DataLoader)`, with nothing inside it: the framework profiler shows that loading is
# slow, not why. A profiler shows what it can see. That is why you time phases as well, and why
# a Python profiler (the stretch) is the next tool.

# %%
ranges = {e.key: e.cpu_time_total / 1000 for e in prof.key_averages() if e.key in med}
print("profiler ranges (ms):", {k: round(v, 1) for k, v in ranges.items()})

# %% [markdown]
# ## 4 · Rank the four phases
#
# ✏️ **Exercise** · Return the phase names sorted from the largest to the smallest median
# time, and the share of the step each one takes (a dict of fractions that add up to 1).

# %%
def rank(med):
    # BEGIN SOLUTION
    order = sorted(med, key=med.get, reverse=True)
    total = sum(med.values())
    return order, {k: med[k] / total for k in order}
    # END SOLUTION

# %%
try:
    order, share = rank(med)
except NotImplementedError:
    order = sorted(med, key=med.get, reverse=True)
    share = {k: med[k] / sum(med.values()) for k in order}
assert abs(sum(share.values()) - 1) < 1e-9
for k in order:
    print(f"{k:10s} {share[k]:6.1%}")
U.save_reference("d1_3_profile", {"median_ms": med, "order": order, "share": share, "step_ms": float(np.median(step_ms))})

plt.figure(figsize=(6, 2.6))
plt.barh(order[::-1], [med[k] for k in order[::-1]], color="#1f4e79")
plt.xlabel("median ms per step")
plt.title("Where Lina's step goes")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5 · Name the largest bottleneck
#
# Write the diagnosis in one sentence: which phase, its share, and what in it costs the time
# (look at the loader in `nama.FieldExportSlow`: what does it do to each leaf, in which order,
# and in which process?). Then predict how fast a step could be if that phase cost nothing.

# %%
best_case = sum(v for k, v in med.items() if k != order[0])
print(f"largest: {order[0]} ({share[order[0]]:.0%}). With it free, a step would take about {best_case:.0f} ms "
      f"instead of {sum(med.values()):.0f} ms: at most {sum(med.values()) / best_case:.1f} x faster.")
# Rewrite this sentence from your own numbers
DIAGNOSIS = (
    f"The step is input-bound: {order[0]} takes {share[order[0]]:.0%} of it, decoding 224-pixel PNG files and "
    "rotating them at full size in the main process, one leaf at a time.")
print(DIAGNOSIS)

# %% [markdown]
# **Look at** the ceiling you just computed. It is Amdahl's law: speeding up one phase can
# never make the step faster than the sum of the others. If your largest phase is not
# loading (a very fast disk and processor can do that), the next lab still applies: measure
# again after each change.

# %% [markdown]
# ## ✱ Stretch · A flame graph with py-spy
#
# A sampling profiler looks at the whole Python program, including Pillow and the loader.
# Install it (`uv add py-spy` locally, `pip install py-spy` in Colab) and run, from `labs/`:
#
# `py-spy record -o artifacts/lina.svg -- python lina_train.py --steps 10`
#
# Open the SVG in a browser. The widest towers are where the time goes. On Linux you may need
# `sudo` or `--nonblocking`; on Windows run it from a normal terminal.

# %% [markdown]
# ## Check yourself
#
# 1. The largest operator row is `aten::convolution_backward`, yet loading is the largest
#    phase. How can both be true?
# 2. Why do the first steps not count?
# 3. Loading is 70% of the step. What is the best speed-up you can get by fixing only it?
#
# <details><summary>Answers</summary>
#
# 1. Loading is many small Python and Pillow calls that the operator table does not break
#    down; their sum is one phase. The largest operator is not the largest phase.
# 2. They pay for start-up: allocating memory, opening files, choosing kernels.
# 3. About 1 / 0.3 = 3.3 times, if loading cost nothing at all.
# </details>
#
# ## Carry forward
#
# Your ranking is in `artifacts/d1_3_profile.json`. Lab 1.4 starts from the same export and
# removes the largest phase.
