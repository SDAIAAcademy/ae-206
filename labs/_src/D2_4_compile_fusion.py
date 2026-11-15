# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "matplotlib>=3.9", "psutil>=6.0", "torch>=2.6", "pillow>=11.0"]
# ///
# %% [markdown]
# # Lab 2.4 · Compile and fuse
#
# Day 2 · Session 4 (Compilation and fusion) · 25 min.
#
# Waraqa runs each layer as a separate operation: convolution, then batch norm, then ReLU, each
# reading and writing a whole activation. Fusion merges operations so the data is touched once;
# a compiler does it for a whole graph. You do one fusion by hand (batch norm into the
# convolution before it), then let a compiler try, and measure the first-call cost against the
# steady-state gain. Last, you complete the results table of the day.
#
# **Tasks**
# - [ ] Eager baseline · 4 min
# - [ ] Fold batch norm into the convolution · 7 min
# - [ ] Check that the outputs match · 4 min
# - [ ] Compile: first call and steady state · 6 min
# - [ ] Complete the results table · 4 min
# - [ ] ✱ Stretch · In Colab, compare the default compile mode with max-autotune
#
# AI policy for this lab: AI-off for the folding formula (task 2); AI-tutor for the rest.
# `torch.compile` needs a C++ compiler: Colab, Linux and macOS have one; a Windows laptop
# usually does not, and the lab then measures the compiler's capture step only and says so.

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

import copy
import json
import time
import numpy as np
import torch
import torch.nn as nn
import nama as U

model = U.load_model("waraqa")
x_test, y_test, _ = U.leaves("test", U.pick(2000, 600))
BATCH = 32
xb = U.to_input(x_test[:BATCH])
N = U.pick(50, 20)
results = {}


def bench(name, m, n=N):
    with torch.no_grad():
        ms = U.time_ms(lambda: m(xb), n=n, warmup=3)
        _, peak = U.peak_rss_mb(lambda: m(xb))
    acc = U.accuracy(m, x_test, y_test)
    results[name] = {"ms_per_batch": float(np.median(ms)), "leaves_per_s": BATCH / np.median(ms) * 1000,
                     "peak_mb": peak, "accuracy": acc}
    print(f"{name:26s} {np.median(ms):7.2f} ms per batch of {BATCH}   {BATCH / np.median(ms) * 1000:6.0f} leaves/s   "
          f"peak {peak:5.1f} MB   accuracy {acc:.3f}")


print(f"mode {U.MODE}: {len(x_test)} held-out leaves, batch {BATCH}, {U.THREADS} threads")

# %% [markdown]
# ## 1 · Eager baseline

# %%
bench("eager", model)

# %% [markdown]
# ## 2 · Fold batch norm into the convolution
#
# In evaluation mode batch norm is a fixed affine map per channel:
# `y = gamma * (z - mean) / sqrt(var + eps) + beta`, applied to the convolution's output
# `z = W * x`. Both are linear, so they combine into one convolution with
#
# - weight `W' = W * s` per output channel, where `s = gamma / sqrt(var + eps)`
# - bias `b' = beta - mean * s` (Waraqa's convolutions have no bias of their own)
#
# ✏️ **Exercise** · Return a new `nn.Conv2d` equal to `bn(conv(x))` (on paper first: AI-off).

# %%
def fold(conv, bn):
    # BEGIN SOLUTION
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=True)
    fused.weight.data = conv.weight.data * s.reshape(-1, 1, 1, 1)
    fused.bias.data = bn.bias.data - bn.running_mean * s
    return fused
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
def _fold(conv, bn):
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    f = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=True)
    f.weight.data = conv.weight.data * s.reshape(-1, 1, 1, 1)
    f.bias.data = bn.bias.data - bn.running_mean * s
    return f

try:
    c0, b0 = U.conv_layers(model)[0]
    fold(c0, b0)
except NotImplementedError:
    fold = _fold


def fold_all(m):
    """A copy of Waraqa with every batch norm folded into its convolution."""
    m = copy.deepcopy(m).eval()
    mods = list(m.features)
    for i in range(len(mods) - 1):
        if isinstance(mods[i], nn.Conv2d) and isinstance(mods[i + 1], nn.BatchNorm2d):
            with torch.no_grad():
                mods[i] = fold(mods[i], mods[i + 1])
            mods[i + 1] = nn.Identity()
    m.features = nn.Sequential(*mods)
    return m


folded = fold_all(model)

# %% [markdown]
# ## 3 · Check that the outputs match
#
# Folding changes the order of floating-point operations, so the outputs agree to rounding,
# not bit for bit. Compare the logits on 256 leaves and the predicted classes.

# %%
with torch.no_grad():
    a, b = model(U.to_input(x_test[:256])), folded(U.to_input(x_test[:256]))
max_diff = float((a - b).abs().max())
same = float((a.argmax(1) == b.argmax(1)).float().mean())
print(f"largest logit difference {max_diff:.2e}, same class on {same:.1%} of leaves")
assert max_diff < 1e-3 and same == 1.0
bench("batch norm folded", folded)

# %% [markdown]
# ## 4 · Compile: first call and steady state
#
# `torch.compile` traces the model into a graph on the first call, fuses operations and
# generates code for them, then reuses that code. The first call pays for all of it.
#
# ✏️ **Exercise** · Return `(compiled_model, backend_used)`. Try `torch.compile(m)` and run it
# once on `xb`. If that raises (no C++ compiler), fall back to
# `torch.compile(m, backend="aot_eager")`, which captures the graph without generating code,
# and return `"aot_eager"` as the backend.

# %%
def compile_model(m):
    # BEGIN SOLUTION
    try:
        c = torch.compile(m)
        with torch.no_grad():
            c(xb)
        return c, "inductor"
    except Exception:
        torch._dynamo.reset()
        return torch.compile(m, backend="aot_eager"), "aot_eager"
    # END SOLUTION

# %%
def _compile_model(m):
    try:
        c = torch.compile(m)
        with torch.no_grad():
            c(xb)
        return c, "inductor"
    except Exception:
        torch._dynamo.reset()
        return torch.compile(m, backend="aot_eager"), "aot_eager"

torch._dynamo.reset()
t0 = time.perf_counter()
try:
    compiled, backend = compile_model(folded)
except NotImplementedError:
    compiled, backend = _compile_model(folded)
with torch.no_grad():
    compiled(xb)
first_s = time.perf_counter() - t0
print(f"backend {backend}: first call (with compilation) {first_s:.2f} s")
bench(f"folded + compile ({backend})", compiled)
steady = results[f"folded + compile ({backend})"]["ms_per_batch"]
gain = results["batch norm folded"]["ms_per_batch"] - steady
if gain > 0:
    print(f"the first call is repaid after about {first_s * 1000 / gain:.0f} batches")
else:
    print("no steady-state gain on this machine: the first call is never repaid")

# %% [markdown]
# **Look at** the backend line. If it says `aot_eager`, this machine has no C++ compiler, so no
# fused code was generated: what you timed is the capture alone. Run the lab in Colab to see
# the inductor backend. Report which backend produced your number; a compile speed-up without
# its backend and machine is not a result.

# %% [markdown]
# ## 5 · Complete the results table
#
# The LO4 evidence is one table: time, throughput, peak memory and accuracy for each setting.
# The training rows come from the course's runs of Labs 2.2 and 2.3 (supplied in
# `data/reference/`), so this lab does not need yours; put your own numbers in if you have them.
# Lab 2.3's rows carry no accuracy (n/a): they compare memory and time for one step only.

# %%
ref = json.loads((U.REF / "d2_training_rows.json").read_text(encoding="utf-8"))
print(f"{'setting':34s} {'ms':>9s} {'per s':>8s} {'peak MB':>8s} {'accuracy':>9s}")
for r in ref["rows"]:
    acc = "      n/a" if r["accuracy"] is None else f"{r['accuracy']:9.3f}"
    print(f"{r['setting']:34s} {r['ms']:9.0f} {r['per_s']:8.0f} {r['peak_mb']:8.0f} {acc}   (training, {r['unit']})")
for k, r in results.items():
    print(f"{k:34s} {r['ms_per_batch']:9.2f} {r['leaves_per_s']:8.0f} {r['peak_mb']:8.1f} {r['accuracy']:9.3f}   (inference, batch {BATCH})")
U.save_reference("d2_4_compile", {"backend": backend, "first_call_s": first_s, "max_logit_diff": max_diff, "results": results})

# %% [markdown]
# ## ✱ Stretch · max-autotune
#
# In a Colab runtime, compile with `mode="max-autotune"`. The first call takes much longer
# (it tries several kernels per operation). Is the steady state faster, and after how many
# batches is the wait repaid?

# %% [markdown]
# ## Check yourself
#
# 1. Why can batch norm be folded at inference but not during training?
# 2. Compilation took 20 s and saves 2 ms per batch. When is it worth it?
# 3. Folding and compiling changed no prediction. Do you still re-measure accuracy?
#
# <details><summary>Answers</summary>
#
# 1. During training batch norm uses each batch's own statistics, which change every step; at
#    inference it uses fixed stored statistics, a constant affine map.
# 2. After 10,000 batches in one process: for a long-running service, yes; for a script that
#    classifies 50 leaves and exits, no. Caching the compiled code moves that line.
# 3. Yes, on the whole held-out set: equal outputs on 256 leaves is evidence, not proof, and a
#    later change (precision, export) can interact with this one.
# </details>
#
# ## Carry forward
#
# The completed table is the LO4 evidence (`artifacts/d2_4_compile.json`). Lab 4.1 exports the
# folded graph to a portable runtime, which performs this fusion by itself.
