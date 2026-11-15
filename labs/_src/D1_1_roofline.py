# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "matplotlib>=3.9", "psutil>=6.0", "torch>=2.6", "pillow>=11.0"]
# ///
# %% [markdown]
# # Lab 1.1 · Two layers on the roofline
#
# Day 1 · Session 1 (Compute, memory and bandwidth) · 25 min.
#
# Before Nama's models are touched, you measure the machine. Two numbers describe it for this
# course: how many floating-point operations per second it can do (**peak compute**) and how
# many bytes per second it can move from memory (**bandwidth**). Then you take two layers,
# estimate by hand how much they compute and how much they move, predict which limit binds
# each, and time them.
#
# **Tasks**
# - [ ] Measure peak compute with a large matrix product · 5 min
# - [ ] Measure bandwidth with a large copy · 4 min
# - [ ] Estimate FLOPs and bytes of both layers by hand · 6 min
# - [ ] Predict, then time both layers · 5 min
# - [ ] Place both on the roofline · 5 min
# - [ ] ✱ Stretch · Repeat the matrix product at batch 1 and watch it slide under the bandwidth roof
#
# AI policy for this lab: AI-off for the estimates in task 3 (the course's rule for every first
# estimate). Close other heavy programs: you are measuring your processor.

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
import matplotlib
import matplotlib.pyplot as plt
import nama as U

print(f"mode {U.MODE}, {U.THREADS} threads")
print(U.machine())

# %% [markdown]
# ## 1 · Peak compute with a large matrix product
#
# A product of two n x n matrices does n³ multiplications and n³ additions: **2n³ FLOPs**.
# For n = 2048 the matrices (16 MB each in float32) are re-used so many times that the
# processor spends its time computing, not waiting for memory. That makes it the standard way
# to approach the compute roof.
#
# ✏️ **Exercise** · Return the FLOPs of an n x n by n x n matrix product.

# %%
def matmul_flops(n):
    # BEGIN SOLUTION
    return 2 * n ** 3
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    assert matmul_flops(10) == 2000
except NotImplementedError:
    def matmul_flops(n):
        return 2 * n ** 3

N = U.pick(2048, 1536)
A, B = torch.randn(N, N), torch.randn(N, N)
ms = U.time_ms(lambda: A @ B, n=U.pick(20, 8), warmup=3)
PEAK = matmul_flops(N) / (np.median(ms) / 1000) / 1e9      # GFLOP/s
print(f"{N} x {N} product: median {np.median(ms):.1f} ms  ->  peak compute about {PEAK:.0f} GFLOP/s")

# %% [markdown]
# ## 2 · Bandwidth with a large copy
#
# A copy computes nothing. Each element is read once and written once, so a copy of B bytes
# moves **2B bytes**. With 256 MB the data cannot stay in the processor's caches, so the copy
# measures main memory.
#
# ✏️ **Exercise** · Return the bytes moved by copying a float32 tensor of `numel` elements.

# %%
def copy_bytes(numel):
    # BEGIN SOLUTION
    return 2 * 4 * numel
    # END SOLUTION

# %%
try:
    assert copy_bytes(1) == 8
except NotImplementedError:
    def copy_bytes(numel):
        return 2 * 4 * numel

src = torch.randn(U.pick(64, 32) * 1024 * 1024)   # 256 MB, or 128 MB in quick mode
dst = torch.empty_like(src)
ms = U.time_ms(lambda: dst.copy_(src), n=U.pick(20, 8), warmup=2)
BW = copy_bytes(src.numel()) / (np.median(ms) / 1000) / 1e9  # GB/s
print(f"copy of {src.numel() * 4 / 1e6:.0f} MB: median {np.median(ms):.1f} ms  ->  bandwidth about {BW:.1f} GB/s")
print(f"ridge point: {PEAK / BW:.1f} FLOPs per byte")
del src, dst

# %% [markdown]
# **Look at** the ridge point: the arithmetic intensity at which the two roofs meet. A layer
# that does fewer FLOPs per byte than this waits for memory; one that does more waits for the
# arithmetic units.

# %% [markdown]
# ## 3 · Estimate FLOPs and bytes of both layers by hand (AI-off)
#
# The two layers, in float32 (4 bytes per number):
#
# - **Layer A**, a fully connected layer: input of batch x d_in, weight d_in x d_out, output
#   batch x d_out, with batch = 512, d_in = d_out = 1024. FLOPs: one multiply and one add per
#   weight per row of the batch. Bytes: read the input and the weight, write the output.
# - **Layer B**, an elementwise "add a bias, then ReLU" on a 4096 x 4096 activation. FLOPs:
#   two per element. Bytes: read each element once and write it once (the bias is tiny).
#
# Write both on paper first. Then fill in the functions.
#
# ✏️ **Exercise** · Return `(flops, bytes)` for each layer, and the arithmetic intensity.

# %%
def linear_cost(batch, d_in, d_out):
    # BEGIN SOLUTION
    flops = 2 * batch * d_in * d_out
    nbytes = 4 * (batch * d_in + d_in * d_out + batch * d_out)
    return flops, nbytes
    # END SOLUTION


def elementwise_cost(numel):
    # BEGIN SOLUTION
    return 2 * numel, 4 * 2 * numel
    # END SOLUTION


def intensity(flops, nbytes):
    # BEGIN SOLUTION
    return flops / nbytes
    # END SOLUTION

# %%
try:
    cost_a, cost_b = linear_cost(512, 1024, 1024), elementwise_cost(4096 * 4096)
    ai_a, ai_b = intensity(*cost_a), intensity(*cost_b)
except NotImplementedError:
    cost_a = (2 * 512 * 1024 * 1024, 4 * (512 * 1024 + 1024 * 1024 + 512 * 1024))
    cost_b = (2 * 4096 * 4096, 8 * 4096 * 4096)
    ai_a, ai_b = cost_a[0] / cost_a[1], cost_b[0] / cost_b[1]
assert abs(ai_a - 128.0) < 0.01 and abs(ai_b - 0.25) < 1e-9, (ai_a, ai_b)
print(f"layer A: {cost_a[0] / 1e9:.2f} GFLOP, {cost_a[1] / 1e6:.1f} MB, intensity {ai_a:.1f} FLOP/byte")
print(f"layer B: {cost_b[0] / 1e9:.3f} GFLOP, {cost_b[1] / 1e6:.1f} MB, intensity {ai_b:.2f} FLOP/byte")

# %% [markdown]
# ## 4 · Predict, then time both layers
#
# The roofline's prediction for a layer is: attainable throughput = min(peak, intensity x
# bandwidth), so its time is FLOPs / attainable.
#
# ✏️ **Exercise** · Return the predicted milliseconds and the binding limit (`"compute"` or
# `"memory"`) for a layer.

# %%
def predict(flops, nbytes, peak_gflops, bw_gbs):
    # BEGIN SOLUTION
    ai = flops / nbytes
    attainable = min(peak_gflops, ai * bw_gbs)
    bound = "compute" if peak_gflops <= ai * bw_gbs else "memory"
    return flops / (attainable * 1e9) * 1000, bound
    # END SOLUTION

# %%
try:
    pred_a, pred_b = predict(*cost_a, PEAK, BW), predict(*cost_b, PEAK, BW)
except NotImplementedError:
    def predict(flops, nbytes, peak_gflops, bw_gbs):
        ai = flops / nbytes
        att = min(peak_gflops, ai * bw_gbs)
        return flops / (att * 1e9) * 1000, ("compute" if peak_gflops <= ai * bw_gbs else "memory")
    pred_a, pred_b = predict(*cost_a, PEAK, BW), predict(*cost_b, PEAK, BW)
assert pred_b[1] == "memory", pred_b

xa, wa, ba = torch.randn(512, 1024), torch.randn(1024, 1024), torch.randn(1024)
act, bias = torch.randn(4096, 4096), torch.randn(4096)
out_b = torch.empty_like(act)
t_a = np.median(U.time_ms(lambda: xa @ wa, n=U.pick(50, 20)))
t_b = np.median(U.time_ms(lambda: torch.add(act, bias, out=out_b).relu_(), n=U.pick(50, 20)))
for name, cost, pred, t in (("A", cost_a, pred_a, t_a), ("B", cost_b, pred_b, t_b)):
    print(f"layer {name}: predicted {pred[0]:7.2f} ms ({pred[1]}-bound), measured {t:7.2f} ms, "
          f"achieved {cost[0] / t / 1e6:6.1f} GFLOP/s and {cost[1] / t / 1e6:5.1f} GB/s")
U.save_reference("d1_1_layers", {"peak_gflops": PEAK, "bw_gbs": BW, "a": [ai_a, pred_a[0], t_a], "b": [ai_b, pred_b[0], t_b]})

# %% [markdown]
# **Look at** the two ratios of measured to predicted. A roofline gives a floor on time, never
# a promise: a measured layer is slower than its roof by whatever the roof leaves out (start-up
# of the call, cache misses, threads waiting for each other). The useful question is which roof
# it sits under, and how far.

# %% [markdown]
# ## 5 · Place both on the roofline

# %%
ai = np.logspace(-2, 3, 200)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.loglog(ai, np.minimum(PEAK, ai * BW), color="#1f4e79", lw=2, label="roofline of this laptop")
for name, a_i, cost, t, c in (("A · linear", ai_a, cost_a, t_a, "#2a9d8f"), ("B · bias + ReLU", ai_b, cost_b, t_b, "#e76f51")):
    ax.scatter([a_i], [cost[0] / t / 1e6], s=60, color=c, zorder=3, label=f"{name} (measured)")
ax.axvline(PEAK / BW, ls=":", color="grey")
ax.set_xlabel("arithmetic intensity (FLOP per byte)")
ax.set_ylabel("GFLOP/s")
ax.legend(fontsize=8)
ax.set_title("Two layers under the roofline")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## ✱ Stretch · Batch 1
#
# Repeat layer A with batch = 1 (one leaf, one farmer question). Estimate its intensity first:
# the weight still has to be read once, but it is now used for one row only. Where does it land
# on the plot? This is why serving one request at a time is usually memory-bound.

# %% [markdown]
# ## Check yourself
#
# 1. A layer does twice the FLOPs of another and runs in the same time. What can you conclude?
# 2. Why does a copy measure bandwidth and not compute?
# 3. A model's decoding step reads all its weights to produce one token. Which roof binds it?
#
# <details><summary>Answers</summary>
#
# 1. Probably that both are memory-bound and move about the same bytes: FLOPs were not the
#    limit. Check their intensities against the ridge point.
# 2. It does no arithmetic: every nanosecond is spent moving bytes, so bytes per second is
#    the only rate it can show.
# 3. Bandwidth, at batch 1: each weight is read once for two FLOPs. Batching more requests
#    raises the intensity.
# </details>
#
# ## Carry forward
#
# Your peak and bandwidth figures are saved in `artifacts/d1_1_layers.json`. Keep them: the
# memory estimates of Day 2 and the language model of Day 3 use the same two numbers.
