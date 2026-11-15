# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "scipy>=1.14", "matplotlib>=3.9", "psutil>=6.0", "torch>=2.6", "pillow>=11.0", "onnxruntime>=1.21"]
# ///
# %% [markdown]
# # Lab 4.3 · Quality by crop type
#
# Day 4 · Session 3 (Quality must survive) · 25 min.
#
# Nama's quality floor: a variant may lose at most **1 point** of accuracy on the held-out set
# overall and at most **2 points** on any crop. Compression changes errors unevenly, so the
# check is per crop, and because the held-out set is finite each difference gets an interval.
# You compare three compressed Waraqa variants with the original on the same leaves.
#
# **Tasks**
# - [ ] Predictions of each variant · 4 min
# - [ ] Paired differences overall · 6 min
# - [ ] An interval on each difference · 5 min
# - [ ] By crop type · 6 min
# - [ ] Accept or reject each variant · 4 min
# - [ ] ✱ Stretch · Which leaves flip from right to wrong? Look at twelve of them
#
# AI policy for this lab: AI-tutor. The accept or reject decision is yours.

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
from scipy.stats import binomtest
import nama as U

x, y, crop = U.leaves("test", U.pick(2000, 600))
FLOOR_ALL, FLOOR_CROP = 0.01, 0.02
VARIANTS = {"8-bit": "waraqa_int8.onnx", "pruned": "waraqa_pruned.onnx", "Waraqa-S": "waraqa_s.onnx"}
print(f"mode {U.MODE}: {len(y)} held-out leaves; crops {dict(zip(U.CROPS, np.bincount(crop, minlength=4).tolist()))}")

# %% [markdown]
# ## 1 · Predictions of each variant
#
# All four run in ONNX Runtime on the same leaves. Keep, per leaf, whether each one is right:
# a paired comparison compares the two models leaf by leaf.

# %%
right = {"original": U.onnx_predict(U.onnx_session(U.MODELS / "waraqa.onnx"), x) == y}
for name, f in VARIANTS.items():
    right[name] = U.onnx_predict(U.onnx_session(U.MODELS / f), x) == y
for k, v in right.items():
    print(f"{k:9s} accuracy {v.mean():.3f}")

# %% [markdown]
# ## 2 · Paired differences overall
#
# Only the leaves where the two models disagree carry information: `b` leaves the original gets
# right and the variant wrong, `c` the other way. The accuracy difference is `(c - b) / n`.
# McNemar's exact test asks whether b and c could come from a fair coin.
#
# ✏️ **Exercise** · Return `(b, c, difference, p_value)` for the variant's right/wrong array
# against the original's, with `binomtest(b, b + c, 0.5).pvalue` (p = 1 when b + c = 0).

# %%
def paired(orig, var):
    # BEGIN SOLUTION
    b = int((orig & ~var).sum())
    c = int((~orig & var).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, (c - b) / len(orig), float(p)
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
def _paired(orig, var):
    b, c = int((orig & ~var).sum()), int((~orig & var).sum())
    return b, c, (c - b) / len(orig), float(binomtest(b, b + c, 0.5).pvalue if b + c else 1.0)

try:
    paired(right["original"], right["original"])
except NotImplementedError:
    paired = _paired
for name in VARIANTS:
    b, c, d, p = paired(right["original"], right[name])
    print(f"{name:9s} lost {b:3d} leaves, gained {c:3d}: difference {d:+.3f}, McNemar p = {p:.3f}")

# %% [markdown]
# ## 3 · An interval on each difference
#
# A paired bootstrap resamples **leaves** (with both models' results attached) and recomputes
# the difference each time.
#
# ✏️ **Exercise** · Return the 95% interval `(low, high)` of the accuracy difference (variant
# minus original) from `n_boot` paired resamples (seed 206).

# %%
def paired_ci(orig, var, n_boot=2000):
    # BEGIN SOLUTION
    rng = np.random.default_rng(206)
    d = var.astype(float) - orig.astype(float)
    boots = d[rng.integers(0, d.size, (n_boot, d.size))].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    # END SOLUTION

# %%
try:
    paired_ci(right["original"], right["original"], 10)
except NotImplementedError:
    def paired_ci(orig, var, n_boot=2000):
        rng = np.random.default_rng(206)
        d = var.astype(float) - orig.astype(float)
        boots = d[rng.integers(0, d.size, (n_boot, d.size))].mean(axis=1)
        return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

overall = {}
for name in VARIANTS:
    d = right[name].mean() - right["original"].mean()
    lo, hi = paired_ci(right["original"], right[name])
    overall[name] = {"diff": float(d), "ci": [lo, hi]}
    print(f"{name:9s} {d:+.3f}  95% interval [{lo:+.3f}, {hi:+.3f}]")

# %% [markdown]
# ## 4 · By crop type

# %%
by_crop = {}
print(f"{'variant':9s} " + " ".join(f"{c:>24s}" for c in U.CROPS))
for name in VARIANTS:
    cells, by_crop[name] = [], {}
    for k, c in enumerate(U.CROPS):
        m = crop == k
        d = right[name][m].mean() - right["original"][m].mean()
        lo, hi = paired_ci(right["original"][m], right[name][m])
        by_crop[name][c] = {"n": int(m.sum()), "orig": float(right["original"][m].mean()), "diff": float(d), "ci": [lo, hi]}
        cells.append(f"{d:+.3f} [{lo:+.3f},{hi:+.3f}]")
    print(f"{name:9s} " + " ".join(f"{s:>24s}" for s in cells))
print("original accuracy by crop:", {c: round(by_crop['8-bit'][c]['orig'], 3) for c in U.CROPS})

# %% [markdown]
# **Look at** the date-palm column: the fewest leaves, so the widest intervals, and the crop
# where the original model is already weakest. An overall number can pass the floor while one
# crop fails it; that is the crop whose farmers would get the worse advice.

# %% [markdown]
# ## 5 · Accept or reject each variant
#
# ✏️ **Exercise** · Return `"accept"`, `"reject"` or `"uncertain"` for a variant. Reject when a
# point estimate breaks the floor (overall below `-FLOOR_ALL`, or any crop below `-FLOOR_CROP`).
# Otherwise, `"uncertain"` when any interval's lower end is below its floor; else `"accept"`.

# %%
def decide(overall_v, crops_v):
    # BEGIN SOLUTION
    if overall_v["diff"] < -FLOOR_ALL or any(v["diff"] < -FLOOR_CROP for v in crops_v.values()):
        return "reject"
    if overall_v["ci"][0] < -FLOOR_ALL or any(v["ci"][0] < -FLOOR_CROP for v in crops_v.values()):
        return "uncertain"
    return "accept"
    # END SOLUTION

# %%
def _decide(o, cv):
    if o["diff"] < -FLOOR_ALL or any(v["diff"] < -FLOOR_CROP for v in cv.values()):
        return "reject"
    if o["ci"][0] < -FLOOR_ALL or any(v["ci"][0] < -FLOOR_CROP for v in cv.values()):
        return "uncertain"
    return "accept"

try:
    verdict = {n: decide(overall[n], by_crop[n]) for n in VARIANTS}
except NotImplementedError:
    verdict = {n: _decide(overall[n], by_crop[n]) for n in VARIANTS}
for n, v in verdict.items():
    print(f"{n:9s} {v}")
U.save_reference("d4_3_quality", {"overall": overall, "by_crop": by_crop, "verdict": verdict})

# %% [markdown]
# **Look at** an "uncertain" verdict: it is a request for more evidence (more date-palm leaves
# in the evaluation set), not a pass. Write in one line what Nama should collect before
# deciding.

# %% [markdown]
# ## ✱ Stretch · The flipped leaves
#
# Find the leaves the original gets right and the 8-bit variant gets wrong
# (`right["original"] & ~right["8-bit"]`). Show twelve with matplotlib, with crop and class.
# Do they share something: crop, class, faint lesions?

# %% [markdown]
# ## Check yourself
#
# 1. Why is a paired comparison more sensitive than comparing two accuracies?
# 2. A variant is 0.4 points worse overall, interval [-1.3, +0.5]. Accept?
# 3. Why does date palm get the widest interval?
#
# <details><summary>Answers</summary>
#
# 1. Both models see the same leaves, so the leaves' difficulty cancels out; only the leaves
#    they disagree on matter.
# 2. Not yet by Nama's rule: the point estimate passes the 1-point floor but the interval
#    reaches below it: uncertain.
# 3. It has the fewest leaves in the held-out set (about one in ten).
# </details>
#
# ## Carry forward
#
# The verdicts and intervals feed the results table of Lab 4.4 and are LO5's "quality loss with
# an interval" (`artifacts/d4_3_quality.json`).
