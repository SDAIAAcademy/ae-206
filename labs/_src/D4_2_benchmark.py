# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "matplotlib>=3.9", "psutil>=6.0", "torch>=2.6", "pillow>=11.0", "onnxruntime>=1.21"]
# ///
# %% [markdown]
# # Lab 4.2 · A benchmark you can trust
#
# Day 4 · Session 2 (A benchmark you can trust) · 25 min.
#
# Ghadeer will choose between the 32-bit and the 8-bit Waraqa from your numbers, and Mutaib
# will re-run your benchmark on a farm-office laptop. So the benchmark must control what it
# can (warm-up, repetitions, the inputs, the batch, the threads), report what it cannot (the
# machine and its load), give intervals instead of single numbers, and give the same answer when
# someone else runs it.
#
# **Tasks**
# - [ ] Warm-up and repetitions · 5 min
# - [ ] A fixed input mix · 4 min
# - [ ] Percentiles with bootstrap intervals · 6 min
# - [ ] Both variants in one report · 5 min
# - [ ] Re-run and compare within 10% · 5 min
# - [ ] ✱ Stretch · Time the benchmark script from the command line with hyperfine
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

import json
import time
import numpy as np
import nama as U

VARIANTS = {"Waraqa 32-bit": U.MODELS / "waraqa.onnx", "Waraqa 8-bit": U.MODELS / "waraqa_int8.onnx"}
x_test, y_test, crop_test = U.leaves("test", U.pick(2000, 600))
REPS, WARMUP = U.pick(300, 100), U.pick(30, 10)
print(f"mode {U.MODE}: {REPS} timed calls after {WARMUP} warm-up calls, batch 1, {U.THREADS} threads")

# %% [markdown]
# ## 1 · Warm-up and repetitions
#
# ✏️ **Exercise** · Write `benchmark(run, inputs, warmup, reps)`: call `run(inputs[i % len])`
# `warmup` times untimed, then `reps` times, timing each call with `time.perf_counter()`, and
# return the list of milliseconds. Cycling through `inputs` in order keeps the input mix fixed.

# %%
def benchmark(run, inputs, warmup, reps):
    # BEGIN SOLUTION
    for i in range(warmup):
        run(inputs[i % len(inputs)])
    out = []
    for i in range(reps):
        x = inputs[i % len(inputs)]
        t0 = time.perf_counter()
        run(x)
        out.append((time.perf_counter() - t0) * 1000)
    return out
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
def _benchmark(run, inputs, warmup, reps):
    for i in range(warmup):
        run(inputs[i % len(inputs)])
    out = []
    for i in range(reps):
        x = inputs[i % len(inputs)]
        t0 = time.perf_counter()
        run(x)
        out.append((time.perf_counter() - t0) * 1000)
    return out

try:
    benchmark(lambda x: None, [0], 1, 2)
except NotImplementedError:
    benchmark = _benchmark

# %% [markdown]
# ## 2 · A fixed input mix
#
# A benchmark on one leaf measures one leaf. Nama's leaves come in four crops in known shares;
# the input mix is 40 held-out leaves with the crops in those shares, saved to a file so that
# Mutaib's run uses the same ones.
#
# ✏️ **Exercise** · Return the indices of `n` leaves of `crop_test` with crop k taking
# `round(n * share[k])` of them (the first ones of each crop, in index order).

# %%
SHARE = [0.35, 0.30, 0.25, 0.10]


def input_mix(crop, n=40):
    # BEGIN SOLUTION
    idx = []
    for k, s in enumerate(SHARE):
        idx += list(np.flatnonzero(crop == k)[: round(n * s)])
    return np.array(sorted(idx))
    # END SOLUTION

# %%
try:
    mix_idx = input_mix(crop_test)
except NotImplementedError:
    mix_idx = np.array(sorted(sum((list(np.flatnonzero(crop_test == k)[: round(40 * s)]) for k, s in enumerate(SHARE)), [])))
assert len(mix_idx) == 40
mix = U.to_input(x_test[mix_idx]).numpy()
mix_path = U.ARTIFACTS / "mix.npy"
mix_path.parent.mkdir(parents=True, exist_ok=True)
np.save(mix_path, mix)
print("crops in the mix:", {U.CROPS[k]: int((crop_test[mix_idx] == k).sum()) for k in range(4)}, "->", mix_path.name)

# %% [markdown]
# ## 3 · Percentiles with bootstrap intervals
#
# A median from 300 calls is an estimate. Resample the 300 timings with replacement many times,
# take the median of each resample, and the middle 95% of those medians is an interval for it.
#
# ✏️ **Exercise** · Return `(low, high)`, the 95% bootstrap interval of the `q`-th percentile of
# `ms` from `n_boot` resamples (seed 206).

# %%
def percentile_ci(ms, q, n_boot=2000):
    # BEGIN SOLUTION
    a = np.asarray(ms)
    rng = np.random.default_rng(206)
    boots = np.percentile(a[rng.integers(0, a.size, (n_boot, a.size))], q, axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    # END SOLUTION

# %%
try:
    percentile_ci([1.0, 2.0, 3.0], 50, 10)
except NotImplementedError:
    def percentile_ci(ms, q, n_boot=2000):
        a = np.asarray(ms)
        rng = np.random.default_rng(206)
        boots = np.percentile(a[rng.integers(0, a.size, (n_boot, a.size))], q, axis=1)
        return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

# %% [markdown]
# ## 4 · Both variants in one report

# %%
def report(label, path):
    sess = U.onnx_session(path)
    ms = benchmark(lambda x: sess.run(None, {"image": x[None]}), mix, WARMUP, REPS)
    return {"label": label, "file_mb": U.file_mb(path), "median_ms": float(np.median(ms)), "median_ci": percentile_ci(ms, 50),
            "p95_ms": float(np.percentile(ms, 95)), "p95_ci": percentile_ci(ms, 95), "p99_ms": float(np.percentile(ms, 99))}


def run_all():
    return {"machine": U.machine(), "batch": 1, "warmup": WARMUP, "reps": REPS, "inputs": "artifacts/mix.npy (40 leaves)",
            "results": [report(k, v) for k, v in VARIANTS.items()]}


def show(rep):
    for r in rep["results"]:
        print(f"{r['label']:14s} median {r['median_ms']:6.3f} ms [{r['median_ci'][0]:.3f}, {r['median_ci'][1]:.3f}]   "
              f"p95 {r['p95_ms']:6.3f} ms [{r['p95_ci'][0]:.3f}, {r['p95_ci'][1]:.3f}]   {r['file_mb']:.2f} MB")


first = run_all()
show(first)
U.save_reference("d4_2_report", first)

# %% [markdown]
# **Look at** the intervals of the two medians. If they overlap a lot, "8-bit is faster" is not
# yet established on this machine; if they are far apart, it is. The p95 interval is wider than
# the median's: the tail rests on fewer calls.

# %% [markdown]
# ## 5 · Re-run and compare within 10%
#
# The outcome asks for a benchmark another participant re-runs within 10% of the reported
# median. Run it again here, then swap notebooks with a neighbour (same `mix.npy`, their
# machine) and compare.
#
# ✏️ **Exercise** · Return, per variant, the relative difference of the second median from the
# first, `second / first - 1`.

# %%
def rerun_gap(first, second):
    # BEGIN SOLUTION
    return {a["label"]: b["median_ms"] / a["median_ms"] - 1 for a, b in zip(first["results"], second["results"])}
    # END SOLUTION

# %%
second = run_all()
show(second)
try:
    gap = rerun_gap(first, second)
except NotImplementedError:
    gap = {a["label"]: b["median_ms"] / a["median_ms"] - 1 for a, b in zip(first["results"], second["results"])}
for k, g in gap.items():
    print(f"{k:14s} re-run median {g:+.1%}  {'within 10%' if abs(g) <= 0.10 else 'OUTSIDE 10%: find out why before reporting'}")
U.save_reference("d4_2_rerun", {"second": second, "gap": gap})

# %% [markdown]
# **Look at** what to do when a re-run lands outside 10%: check the machine's load at start
# (`bench.py` records it), the power mode (a laptop on battery slows down), the threads, and
# whether the inputs were the same file. The course's command-line version of this benchmark is
# `bench.py`: `uv run bench.py data/models/waraqa_int8.onnx --inputs artifacts/mix.npy --out artifacts/b.json`.

# %% [markdown]
# ## ✱ Stretch · hyperfine
#
# `hyperfine` (a command-line benchmark tool) times whole commands with warm-up runs and
# reports a mean and spread. Time `uv run bench.py data/models/waraqa.onnx --reps 50` with
# `hyperfine --warmup 2 "<command>"`. What does it measure that `bench.py` does not (process
# start, imports, opening the session), and when is that the right number?

# %% [markdown]
# ## Check yourself
#
# 1. Why cycle through a fixed input mix instead of timing one leaf?
# 2. Two medians differ by 3% with overlapping intervals. What do you report?
# 3. Your neighbour's re-run is 25% slower. Name three causes to check.
#
# <details><summary>Answers</summary>
#
# 1. Real traffic is mixed, and a re-run must see the same inputs; one leaf can be an
#    unrepresentative case.
# 2. That on this machine the two are not distinguishable at batch 1; not that one is faster.
# 3. Load on the machine, power mode or thermal throttling, a different thread count, a
#    different runtime version, a different input file.
# </details>
#
# ## Carry forward
#
# The report (`artifacts/d4_2_report.json`) is the LO6 evidence; the capstone uses the same
# method with `bench.py`.
