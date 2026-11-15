# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "matplotlib>=3.9", "psutil>=6.0", "torch>=2.6", "pillow>=11.0"]
# ///
# %% [markdown]
# # Lab 3.1 · Batch size against latency
#
# Day 3 · Session 1 (Latency, throughput and batching) · 25 min.
#
# Waraqa serves two kinds of work at Nama. In a farm office an agent photographs one leaf at a
# time and waits for the answer: **latency**. At night the office re-scores the day's photo
# archive: **throughput**. You serve Waraqa at batch sizes from 1 to 64, plot the trade, and add
# the time a request spends waiting for its batch to fill.
#
# **Tasks**
# - [ ] Time batch sizes from 1 to 64 · 6 min
# - [ ] Throughput per batch size · 4 min
# - [ ] Plot latency against throughput · 5 min
# - [ ] Add the wait for the batch to fill · 6 min
# - [ ] Pick the batch size for the farm office · 4 min
# - [ ] ✱ Stretch · Simulate dynamic batching with a timeout and find its tail latency
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
import matplotlib.pyplot as plt
import nama as U

model = U.load_model("waraqa")
x_test, _, _ = U.leaves("test", 64)
SIZES = (1, 2, 4, 8, 16, 32, 64)
BUDGET_P95_MS = 15          # Nama's latency budget for one leaf in a farm office
N = U.pick(60, 20)
print(f"mode {U.MODE}: {U.THREADS} threads, budget {BUDGET_P95_MS} ms at the 95th percentile")

# %% [markdown]
# ## 1 · Time batch sizes from 1 to 64
#
# ✏️ **Exercise** · For one batch size `b`, return the list of milliseconds of `N` calls of
# Waraqa on the first `b` leaves, after a warm-up (use `U.time_ms` and `torch.no_grad()`).

# %%
def time_batch(b):
    # BEGIN SOLUTION
    xb = U.to_input(x_test[:b])
    with torch.no_grad():
        return U.time_ms(lambda: model(xb), n=N, warmup=3)
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
def _time_batch(b):
    xb = U.to_input(x_test[:b])
    with torch.no_grad():
        return U.time_ms(lambda: model(xb), n=N, warmup=3)

try:
    lat = {b: time_batch(b) for b in SIZES}
except NotImplementedError:
    lat = {b: _time_batch(b) for b in SIZES}
for b in SIZES:
    s = U.summarise(lat[b])
    print(f"batch {b:3d}: median {s['median']:7.2f} ms   p95 {s['p95']:7.2f} ms")

# %% [markdown]
# ## 2 · Throughput per batch size
#
# ✏️ **Exercise** · Return leaves per second for a batch of `b` that takes `median_ms`.

# %%
def throughput(b, median_ms):
    # BEGIN SOLUTION
    return b / median_ms * 1000
    # END SOLUTION

# %%
try:
    thr = {b: throughput(b, np.median(lat[b])) for b in SIZES}
except NotImplementedError:
    thr = {b: b / np.median(lat[b]) * 1000 for b in SIZES}
for b in SIZES:
    print(f"batch {b:3d}: {thr[b]:7.0f} leaves per second, {np.median(lat[b]) / b:6.2f} ms per leaf")

# %% [markdown]
# **Look at** where throughput stops growing. Batching helps while a single leaf leaves the
# processor's cores or vector units idle (batch 1 is memory-bound, Lab 1.1); once they are
# busy, a larger batch only takes longer.

# %% [markdown]
# ## 3 · Plot latency against throughput

# %%
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot([thr[b] for b in SIZES], [np.median(lat[b]) for b in SIZES], "o-", color="#1f4e79")
for b in SIZES:
    ax.annotate(str(b), (thr[b], np.median(lat[b])), textcoords="offset points", xytext=(4, 4), fontsize=8)
ax.axhline(BUDGET_P95_MS, ls="--", color="#e76f51", label="latency budget")
ax.set_xlabel("throughput (leaves per second)")
ax.set_ylabel("median latency of a batch (ms)")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4 · Add the wait for the batch to fill
#
# A batch of b does not exist until b requests have arrived. If leaves arrive one by one at
# `rate` per second, the first leaf of a batch waits for b - 1 more, on average
# (b - 1) / rate seconds, and the average leaf about half of that. Its latency is that wait plus
# the batch's own time.
#
# ✏️ **Exercise** · Return the average end-to-end milliseconds of a leaf in a batch of `b`
# at `rate` leaves per second, given the batch's median time.

# %%
def end_to_end_ms(b, rate, batch_ms):
    # BEGIN SOLUTION
    return (b - 1) / (2 * rate) * 1000 + batch_ms
    # END SOLUTION

# %%
try:
    assert abs(end_to_end_ms(1, 10, 5.0) - 5.0) < 1e-9
except NotImplementedError:
    def end_to_end_ms(b, rate, batch_ms):
        return (b - 1) / (2 * rate) * 1000 + batch_ms

table = []
for rate in (2, 200):           # a farm office in the day; the archive re-scoring at night
    for b in SIZES:
        e2e = end_to_end_ms(b, rate, np.median(lat[b]))
        ok = e2e + (np.percentile(lat[b], 95) - np.median(lat[b])) <= BUDGET_P95_MS
        table.append({"rate": rate, "batch": b, "e2e_ms": e2e, "throughput": thr[b], "meets_budget": bool(ok)})
print(f"{'rate/s':>7s} {'batch':>6s} {'avg e2e ms':>11s} {'max leaves/s':>13s} {'budget':>7s}")
for r in table:
    print(f"{r['rate']:7d} {r['batch']:6d} {r['e2e_ms']:11.1f} {r['throughput']:13.0f} {'ok' if r['meets_budget'] else '-':>7s}")
U.save_reference("d3_1_batching", {"latency": {b: U.summarise(lat[b]) for b in SIZES}, "throughput": thr, "table": table})

# %% [markdown]
# **Look at** the two arrival rates. At 2 leaves per second a batch of 8 makes the first leaf
# wait almost two seconds: batching is useless when requests are rare. At 200 per second the
# wait is short, and batching is how the processor keeps up.
#
# Goodput is the rate of requests served **within** the objective. A server that answers 500
# leaves per second with half of them late has a goodput of 250. AE-205 measures it under load
# for a real service.

# %% [markdown]
# ## 5 · Pick the batch size for the farm office
#
# ✏️ **Exercise** · Among the rows with `rate == 2` that meet the budget, return the batch size
# with the highest throughput.

# %%
def pick_batch(table, rate):
    # BEGIN SOLUTION
    ok = [r for r in table if r["rate"] == rate and r["meets_budget"]]
    return max(ok, key=lambda r: r["throughput"])["batch"] if ok else None
    # END SOLUTION

# %%
try:
    office, night = pick_batch(table, 2), pick_batch(table, 200)
except NotImplementedError:
    def pick_batch(table, rate):
        ok = [r for r in table if r["rate"] == rate and r["meets_budget"]]
        return max(ok, key=lambda r: r["throughput"])["batch"] if ok else None
    office, night = pick_batch(table, 2), pick_batch(table, 200)
say = lambda b: f"batch {b}" if b is not None else "no batch size meets the budget on this machine now"
print(f"farm office (2 leaves/s): {say(office)};   archive at 200 leaves/s within the same budget: {say(night)}")
if office is None:
    print("Even batch 1 misses the budget here: write down what else was running and the thread count, "
          "then re-run on a quiet machine before you conclude (Lab 1.2's rule).")

# %% [markdown]
# ## ✱ Stretch · Dynamic batching
#
# Servers usually batch dynamically: wait until b requests are queued **or** t milliseconds
# have passed, whichever comes first. Simulate Poisson arrivals at 50 leaves per second with
# `np.random.default_rng(0).exponential(1 / 50, 5000)`, a maximum batch of 16 and a timeout of
# 5 ms, using your measured batch times. What are the median and 95th percentile of the
# end-to-end latency?

# %% [markdown]
# ## Check yourself
#
# 1. Batch 32 serves several times more leaves per second than batch 1. Why not use it in the
#    office?
# 2. Why does throughput stop growing after some batch size?
# 3. What is goodput, and why can it fall while throughput rises?
#
# <details><summary>Answers</summary>
#
# 1. Leaves arrive one at a time; the first would wait for 31 more, far beyond the budget.
# 2. Once the cores and vector units are busy, more work per call only takes longer.
# 3. The rate of requests served within the objective. A larger batch raises throughput and
#    can push more requests past the objective, so fewer count.
# </details>
#
# ## Carry forward
#
# The latency per batch size is in `artifacts/d3_1_batching.json`. Lab 4.2 benchmarks batch 1,
# the office's case.
