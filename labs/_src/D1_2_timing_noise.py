# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "matplotlib>=3.9", "psutil>=6.0", "torch>=2.6", "pillow>=11.0"]
# ///
# %% [markdown]
# # Lab 1.2 · Noise in one inference
#
# Day 1 · Session 2 (Timers, profilers and noise) · 25 min.
#
# Mutaib Al-Dhafeeri, who looks after Nama's farm-office laptops, says Waraqa "takes 5 ms".
# Ghadeer Al-Shammasi, who sets the budget, asks: always? You time one inference 100 times,
# look at the whole distribution, and show how the first calls and a busy machine change it.
#
# **Tasks**
# - [ ] Time one inference 100 times · 5 min
# - [ ] Plot the distribution and read its percentiles · 5 min
# - [ ] Cold start: the first calls against the rest · 5 min
# - [ ] Background load: repeat while another process computes · 6 min
# - [ ] Write the timing rule for the course · 4 min
# - [ ] ✱ Stretch · Change the thread count and see which percentile moves most
#
# AI policy for this lab: AI-tutor. An assistant may explain an output; it does not write the
# timing rule for you.

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
import matplotlib.pyplot as plt
import nama as U

model = U.load_model("waraqa")                 # the supplied checkpoint
x_test, y_test, crop_test = U.leaves("test", U.pick(2000, 600))
leaf = U.to_input(x_test[:1])                   # one leaf, as the field app sends it
N = 100
print(f"mode {U.MODE}, {U.THREADS} threads, Waraqa with {U.n_params(model):,} parameters")

# %% [markdown]
# ## 1 · Time one inference 100 times
#
# Use `time.perf_counter()`: a monotonic clock with the finest resolution Python offers.
# `time.time()` is the wall clock: it can jump when the system adjusts it. Wrap the call in
# `torch.no_grad()`, as serving does.
#
# ✏️ **Exercise** · Return a list of `n` durations in milliseconds, one per call of `fn`, with
# no warm-up (we want to see the first calls).

# %%
def time_calls(fn, n):
    # BEGIN SOLUTION
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)
    return out
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
def infer():
    with torch.no_grad():
        return model(leaf)

try:
    quiet = time_calls(infer, N)
except NotImplementedError:
    quiet = U.time_ms(infer, n=N, warmup=0)
assert len(quiet) == N and all(t > 0 for t in quiet)
print(f"first call {quiet[0]:.2f} ms, median {np.median(quiet):.2f} ms")

# %% [markdown]
# ## 2 · Plot the distribution and read its percentiles
#
# ✏️ **Exercise** · Return a dict with the `median`, `p95` and `p99` of a list of
# milliseconds (use `np.percentile`).

# %%
def percentiles(ms):
    # BEGIN SOLUTION
    return {"median": float(np.median(ms)), "p95": float(np.percentile(ms, 95)),
            "p99": float(np.percentile(ms, 99))}
    # END SOLUTION

# %%
try:
    assert percentiles(list(range(1, 101)))["median"] == 50.5
except NotImplementedError:
    def percentiles(ms):
        return {"median": float(np.median(ms)), "p95": float(np.percentile(ms, 95)), "p99": float(np.percentile(ms, 99))}

warm = quiet[5:]
p_quiet = percentiles(warm)
print("quiet machine, after 5 calls:", {k: round(v, 2) for k, v in p_quiet.items()}, f"mean {np.mean(warm):.2f}")
plt.figure(figsize=(6, 3))
plt.hist(warm, bins=30, color="#2a9d8f")
for k, ls in (("median", "-"), ("p95", "--"), ("p99", ":")):
    plt.axvline(p_quiet[k], color="#1f4e79", ls=ls, label=k)
plt.xlabel("ms per inference (batch 1)")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Look at** the shape. Latency distributions are not bell curves: they have a hard floor and
# a long right tail. That is why the course reports the median and the 95th percentile, not
# the mean and a standard deviation.

# %% [markdown]
# ## 3 · Cold start: the first calls against the rest
#
# Load a fresh copy of the model and time its first ten calls. The first call pays for
# one-time work: memory allocation, choosing kernels, filling caches.

# %%
fresh = U.load_model("waraqa")
fresh_leaf = U.to_input(x_test[1:2])


def infer_fresh():
    with torch.no_grad():
        return fresh(fresh_leaf)


first = U.time_ms(infer_fresh, n=10, warmup=0)
print("first ten calls (ms):", [round(t, 2) for t in first])
print(f"call 1 is {first[0] / np.median(warm):.1f} x the warm median")

# %% [markdown]
# ## 4 · Background load
#
# A farm-office laptop runs other things: a browser, a sync client, an antivirus scan. Start
# two processes that compute in a loop, time the same inference again, then stop them.
#
# ✏️ **Exercise** · Start `k` background processes running `BUSY` with `subprocess.Popen`
# and return the list of processes.

# %%
BUSY = "import time\nt=time.time()\nwhile time.time()-t<120: sum(i*i for i in range(10000))"

def start_load(k):
    # BEGIN SOLUTION
    return [subprocess.Popen([sys.executable, "-c", BUSY]) for _ in range(k)]
    # END SOLUTION

# %%
try:
    procs = start_load(2)
except NotImplementedError:
    procs = [subprocess.Popen([sys.executable, "-c", BUSY]) for _ in range(2)]
try:
    time.sleep(1.0)
    busy = U.time_ms(infer, n=N, warmup=5)
finally:
    for p in procs:
        p.kill()
        p.wait()
p_busy = percentiles(busy)
print(f"{'':10s}{'median':>9s}{'p95':>9s}{'p99':>9s}")
for name, p in (("quiet", p_quiet), ("busy", p_busy)):
    print(f"{name:10s}{p['median']:9.2f}{p['p95']:9.2f}{p['p99']:9.2f}")
U.save_reference("d1_2_timing", {"quiet": p_quiet, "busy": p_busy, "first_calls": first})

plt.figure(figsize=(6, 3))
plt.hist(warm, bins=30, alpha=0.7, label="quiet", color="#2a9d8f")
plt.hist(busy, bins=30, alpha=0.7, label="two busy processes", color="#e76f51")
plt.xlabel("ms per inference (batch 1)")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Look at** which number moved most: the median or the tail. On a machine with more free
# cores than busy processes the median may hardly move while the 99th percentile does. On a
# two-core Colab runtime both move.

# %% [markdown]
# ## 5 · Write the timing rule for the course
#
# In two or three lines, the rule every benchmark in this course will follow. Fill it in.

# %%
TIMING_RULE = """
- warm up before timing; report the first call separately
- time each call with a monotonic clock; repeat at least 100 times
- report median and 95th percentile, the machine, the threads and what else was running
"""
print(TIMING_RULE)

# %% [markdown]
# ## ✱ Stretch · Threads
#
# Repeat the quiet measurement with `torch.set_num_threads(1)` and with all cores. Which
# percentile moves most? Set it back to `U.THREADS` afterwards.

# %% [markdown]
# ## Check yourself
#
# 1. Mean 6.1 ms, median 5.2 ms. Which way is the distribution skewed?
# 2. Why is the first call excluded from the steady-state numbers but still reported?
# 3. Your 95th percentile doubled under background load and your median did not. What does a
#    farm agent notice?
#
# <details><summary>Answers</summary>
#
# 1. Right: a few slow calls pull the mean above the median.
# 2. It is a different event (a cold start); a user who opens the app pays it once, so it
#    belongs in the report, but mixing it into 100 warm calls hides both numbers.
# 3. About one leaf in twenty is slow. Most feel normal, which is why a budget is written on
#    the tail.
# </details>
#
# ## Carry forward
#
# The timing rule is the base of Lab 4.2's benchmark. Your numbers are in
# `artifacts/d1_2_timing.json`.
