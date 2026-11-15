# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "matplotlib>=3.9", "psutil>=6.0", "torch>=2.6", "pillow>=11.0"]
# ///
# %% [markdown]
# # Lab 3.4 · Prune or shrink
#
# Day 3 · Session 4 (Pruning or a smaller model) · 25 min.
#
# Two ways to a model with half the channels: prune the trained Waraqa (remove the least
# important filters of every convolution, then fine-tune briefly), or train Waraqa-S, the same
# architecture at half the width, from scratch. Nama supplies Waraqa-S, trained with the same
# budget as Waraqa. You prune, fine-tune and compare.
#
# **Tasks**
# - [ ] Score channels by their L1 norm · 4 min
# - [ ] Prune half and measure accuracy · 5 min
# - [ ] Fine-tune briefly · 6 min
# - [ ] Compare with Waraqa-S · 5 min
# - [ ] Size, latency and accuracy table · 5 min
# - [ ] ✱ Stretch · Mask half the weights without removing channels: does it run faster?
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

import numpy as np
import torch
import nama as U

base = U.load_model("waraqa")
x_tr, y_tr, _ = U.leaves("train", U.pick(8000, 2000))
x_test, y_test, _ = U.leaves("test", U.pick(2000, 600))
OUT = U.ARTIFACTS / "d3_4"
OUT.mkdir(parents=True, exist_ok=True)
rows, models = [], {}


def row(name, m):
    models[name] = m
    leaf = U.to_input(x_test[:1])
    with torch.no_grad():
        s = U.summarise(U.time_ms(lambda: m(leaf), n=U.pick(100, 40), warmup=5))
    path = U.save_model(m, OUT / f"{name.replace(' ', '_')}.pt")
    r = {"model": name, "params": U.n_params(m), "file_mb": U.file_mb(path), "median_ms": s["median"],
         "p95_ms": s["p95"], "accuracy": U.accuracy(m, x_test, y_test)}
    rows.append(r)
    print(f"{name:28s} {r['params']:8,d} params  {r['file_mb']:5.2f} MB  median {r['median_ms']:5.2f} ms  accuracy {r['accuracy']:.3f}")
    return r


print(f"mode {U.MODE}")
row("Waraqa", base)

# %% [markdown]
# ## 1 · Score channels by their L1 norm
#
# A filter whose weights are all small contributes little to the next layer. The sum of the
# absolute values of a filter's weights (its L1 norm) is a cheap importance score.
# `U.channel_scores(model)` returns one tensor of scores per convolution.
#
# ✏️ **Exercise** · For each convolution, return the indices of the `ratio` share of filters
# with the highest scores, in ascending order (`torch.topk`, then `torch.sort`).

# %%
def keep_top(scores, ratio=0.5):
    # BEGIN SOLUTION
    return [torch.sort(torch.topk(s, max(1, int(len(s) * ratio))).indices).values for s in scores]
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
scores = U.channel_scores(base)
try:
    keep = keep_top(scores)
except NotImplementedError:
    keep = [torch.sort(torch.topk(s, len(s) // 2).indices).values for s in scores]
assert [len(k) for k in keep] == [w // 2 for w in base.widths]
for i, (s, k) in enumerate(zip(scores, keep)):
    print(f"conv {i + 1}: {len(s):3d} filters, keep {len(k):3d}; lowest kept score {s[k].min():.2f}, highest dropped {s[~torch.isin(torch.arange(len(s)), k)].max():.2f}")

# %% [markdown]
# ## 2 · Prune half and measure accuracy
#
# `U.prune_channels(model, keep)` builds a **smaller** network that holds only the kept filters
# (and the matching input channels of the next layer). Removing whole channels shrinks every
# matrix the processor multiplies, which is why structured pruning can be faster.

# %%
pruned = U.prune_channels(base, keep)
r_pruned = row("pruned, no fine-tune", pruned)

# %% [markdown]
# ## 3 · Fine-tune briefly
#
# The remaining filters adapt to the missing ones in a short fine-tune, at a low learning rate.
#
# ✏️ **Exercise** · Fine-tune `pruned` for `steps` steps of 64 leaves at learning rate 5e-4
# with `U.train_steps` (on `x_tr`, `y_tr`) and return it.

# %%
def fine_tune(m, steps):
    # BEGIN SOLUTION
    U.train_steps(m, x_tr, y_tr, steps, lr=5e-4, seed=11)
    return m
    # END SOLUTION

# %%
STEPS = U.pick(250, 60)
try:
    tuned = fine_tune(pruned, STEPS)
except NotImplementedError:
    U.train_steps(pruned, x_tr, y_tr, STEPS, lr=5e-4, seed=11)
    tuned = pruned
r_tuned = row(f"pruned + {STEPS} steps", tuned)
assert r_tuned["accuracy"] >= r_pruned["accuracy"] - 0.02

# %% [markdown]
# ## 4 · Compare with Waraqa-S
#
# Waraqa-S has the same widths as the pruned model and was trained from scratch for as many
# epochs as Waraqa. The course also supplies the pruned model it fine-tuned for a full epoch
# (`waraqa_pruned`).

# %%
row("Waraqa-S (from scratch)", U.load_model("waraqa_s"))
row("pruned + 1 epoch (supplied)", U.load_model("waraqa_pruned"))

# %% [markdown]
# **Look at** the price of each route. Pruning starts from a trained model and needs only a
# short fine-tune; training Waraqa-S cost a full training run. Distillation is a third route:
# train the small model to match the large model's output probabilities instead of the labels
# alone, so it learns which wrong classes are "nearly right".

# %% [markdown]
# ## 5 · Size, latency and accuracy table
#
# The timings above were taken at different moments: some before the fine-tune, some right
# after it, when the processor was still hot. Lab 1.2's rule says to compare under the same
# conditions, so time every model again here, in turn, several rounds, and pool the calls.

# %%
leaf = U.to_input(x_test[:1])
pooled = {r["model"]: [] for r in rows}
with torch.no_grad():
    for _ in range(U.pick(6, 3)):
        for r in rows:
            m = models[r["model"]]
            pooled[r["model"]] += U.time_ms(lambda: m(leaf), n=U.pick(40, 15), warmup=3)
for r in rows:
    s = U.summarise(pooled[r["model"]])
    r["median_ms"], r["p95_ms"] = s["median"], s["p95"]
print(f"{'model':28s} {'params':>8s} {'MB':>5s} {'median ms':>9s} {'p95 ms':>7s} {'accuracy':>8s}")
for r in rows:
    print(f"{r['model']:28s} {r['params']:8,d} {r['file_mb']:5.2f} {r['median_ms']:9.2f} {r['p95_ms']:7.2f} {r['accuracy']:8.3f}")
U.save_reference("d3_4_prune", rows)

# %% [markdown]
# ## ✱ Stretch · A mask is not a smaller model
#
# Set the half of each convolution's weights with the smallest magnitudes to zero (unstructured
# pruning, `torch.nn.utils.prune.l1_unstructured`, then `prune.remove`). Count the zeros, then
# time it. Dense kernels multiply zeros as fast as any other number: half the weights are zero
# and the time does not change, unless the runtime has sparse kernels for that pattern.

# %% [markdown]
# ## Check yourself
#
# 1. Why does removing channels speed Waraqa up while zeroing half its weights does not?
# 2. The pruned model lost 10 points before fine-tuning and recovered them. What happened?
# 3. When would you prefer a smaller architecture trained from scratch?
#
# <details><summary>Answers</summary>
#
# 1. Removing channels makes every matrix smaller; zeros in a dense matrix are still
#    multiplied.
# 2. The remaining filters were trained to work with the removed ones; a short fine-tune lets
#    them take over the missing part.
# 3. When you will train anyway (new data every month), when the large model is not yours to
#    prune, or when the small model reaches the same quality: then it is the simpler artefact.
# </details>
#
# ## Carry forward
#
# The pruned and small models are both supplied (`data/models/waraqa_pruned.*`,
# `waraqa_s.*`); Day 4 evaluates them against the quality floor.
