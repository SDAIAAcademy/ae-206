# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "matplotlib>=3.9", "psutil>=6.0", "torch>=2.6", "pillow>=11.0", "onnx>=1.17", "onnxruntime>=1.21"]
# ///
# %% [markdown]
# # Lab 3.3 · Quantise Waraqa and Hasad
#
# Day 3 · Session 3 (Quantisation) · 25 min.
#
# Nama's budget allows a Waraqa file of at most 1 MB; the 32-bit file is larger. You quantise
# Waraqa to 8-bit integers with a calibration set, then load the supplied 8-bit and 4-bit files
# of Hasad. For each you measure size, speed and quality, and for Hasad quality by language
# variety: a compressed model can lose more on one kind of question than on another.
#
# **Tasks**
# - [ ] Calibrate and quantise Waraqa to 8 bits · 6 min
# - [ ] Size, latency and accuracy of both Waraqa files · 5 min
# - [ ] Load Hasad's 8-bit and 4-bit files · 4 min
# - [ ] Size, speed and right answers by language variety · 6 min
# - [ ] One comparison table · 4 min
# - [ ] ✱ Stretch · Quantise Waraqa dynamically instead and compare with the static file
#
# AI policy for this lab: AI-tutor. An assistant may explain the quantisation tool's options and
# its errors; the comparison and its reading are yours.

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

import shutil
import numpy as np
import torch
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quant_pre_process, quantize_static)
import nama as U

OUT = U.ARTIFACTS / "d3_3"
OUT.mkdir(parents=True, exist_ok=True)
fp32_path = OUT / "waraqa_fp32.onnx"
shutil.copy(U.MODELS / "waraqa.onnx", fp32_path)          # the supplied 32-bit export
x_test, y_test, crop_test = U.leaves("test", U.pick(2000, 600))
x_cal, _, _ = U.leaves("train", 256)
qs = U.question_set(U.pick(200, 60))
N = U.pick(100, 40)
table = []
print(f"mode {U.MODE}: {len(x_test)} held-out leaves, {len(qs)} questions")

# %% [markdown]
# ## 1 · Calibrate and quantise Waraqa to 8 bits
#
# Static quantisation stores weights **and** activations as 8-bit integers, each tensor with a
# scale (and a zero point) that maps integers back to real values. Weights have known ranges;
# activations do not, so the tool runs the model on a few real leaves (the **calibration
# set**) and records each activation's range. ONNX Runtime asks for those leaves through a
# reader with one method, `get_next`, which returns one input dict per call and `None` at the
# end.
#
# ✏️ **Exercise** · Write `get_next`: return `{"image": batch}` for the next 16 leaves of
# `self.x` (float32 array from `U.to_input(...).numpy()`), and `None` when they are used up.

# %%
class Reader(CalibrationDataReader):
    def __init__(self, x, batch=16):
        self.x, self.batch, self.i = x, batch, 0

    def get_next(self):
        # BEGIN SOLUTION
        if self.i >= len(self.x):
            return None
        out = {"image": U.to_input(self.x[self.i:self.i + self.batch]).numpy()}
        self.i += self.batch
        return out
        # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
class _Reader(CalibrationDataReader):
    def __init__(self, x, batch=16):
        self.it = iter([{"image": U.to_input(x[i:i + batch]).numpy()} for i in range(0, len(x), batch)])

    def get_next(self):
        return next(self.it, None)

try:
    Reader(x_cal[:16]).get_next()
except NotImplementedError:
    Reader = _Reader

pre_path, int8_path = OUT / "waraqa_pre.onnx", OUT / "waraqa_int8.onnx"
quant_pre_process(str(fp32_path), str(pre_path))
quantize_static(str(pre_path), str(int8_path), Reader(x_cal), quant_format=QuantFormat.QDQ, per_channel=True,
                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
print(f"32-bit {U.file_mb(fp32_path):.3f} MB  ->  8-bit {U.file_mb(int8_path):.3f} MB")

# %% [markdown]
# ## 2 · Size, latency and accuracy of both Waraqa files
#
# The budget is per leaf in the office: batch 1, 95th percentile. Accuracy is measured again
# after every change, on the same held-out leaves.

# %%
def waraqa_row(name, path):
    sess = U.onnx_session(path)
    leaf = U.to_input(x_test[:1]).numpy()
    s = U.summarise(U.time_ms(lambda: sess.run(None, {"image": leaf}), n=N, warmup=10))
    pred = U.onnx_predict(sess, x_test)
    by_crop = {c: float((pred[crop_test == k] == y_test[crop_test == k]).mean()) for k, c in enumerate(U.CROPS)}
    row = {"model": name, "file_mb": U.file_mb(path), "median_ms": s["median"], "p95_ms": s["p95"],
           "quality": float((pred == y_test).mean()), "by_group": by_crop}
    table.append(row)
    return row


for name, path in (("Waraqa 32-bit", fp32_path), ("Waraqa 8-bit", int8_path)):
    r = waraqa_row(name, path)
    print(f"{name:14s} {r['file_mb']:6.3f} MB   median {r['median_ms']:5.2f} ms   p95 {r['p95_ms']:5.2f} ms   accuracy {r['quality']:.3f}")
    print("   by crop:", {k: round(v, 3) for k, v in r["by_group"].items()})

# %% [markdown]
# **Look at** the accuracy by crop, not only overall. Date palm is the smallest crop in the data
# (one leaf in ten) and the hardest; a difference of a few leaves there is a large share, in
# either direction. With per-channel weights and calibration leaves from every crop, the 8-bit
# file usually keeps every crop within a leaf or two. Day 4 puts intervals on these differences
# before anyone accepts or rejects a file.

# %% [markdown]
# ## 3 · Load Hasad's 8-bit and 4-bit files
#
# The course quantised Hasad's linear layers in groups of 32 weights that share one 16-bit
# scale: 8-bit codes in one file, 4-bit codes (two per byte) in the other. The embedding table
# stays in 16 bits. Runtimes for small language models use the same idea of block-wise scales.
# `U.load_hasad(bits, mode)` can turn the codes back into float32 **once** at load, or keep them
# in memory and expand each weight **per call**.

# %%
for bits in (32, 8, 4):
    print(f"hasad_{bits}.pt: {U.file_mb(U.MODELS / f'hasad_{bits}.pt'):.2f} MB")

# %% [markdown]
# ## 4 · Size, speed and right answers by language variety
#
# ✏️ **Exercise** · For a loaded Hasad, return `(share right, {variety: share right},
# median decode tokens per second)` over the question set `qs`. A question is right when
# `U.grade(answer, q["key"])` is true.

# %%
def hasad_quality(model):
    # BEGIN SOLUTION
    ok, var, speed = [], {}, []
    for q in qs:
        r = model.generate(U.prompt_for(q["question"]))
        good = U.grade(r["text"], q["key"])
        ok.append(good)
        var.setdefault(q["variety"], []).append(good)
        speed.append(r["decode_tok_s"])
    return float(np.mean(ok)), {k: float(np.mean(v)) for k, v in var.items()}, float(np.median(speed))
    # END SOLUTION

# %%
def _hasad_quality(model):
    ok, var, speed = [], {}, []
    for q in qs:
        r = model.generate(U.prompt_for(q["question"]))
        g = U.grade(r["text"], q["key"])
        ok.append(g)
        var.setdefault(q["variety"], []).append(g)
        speed.append(r["decode_tok_s"])
    return float(np.mean(ok)), {k: float(np.mean(v)) for k, v in var.items()}, float(np.median(speed))

for bits, mode in ((32, "once"), (8, "once"), (4, "once"), (4, "per_call")):
    m = U.load_hasad(bits, mode)
    m.generate(U.prompt_for(qs[0]["question"]))
    try:
        acc, by_var, tps = hasad_quality(m)
    except NotImplementedError:
        acc, by_var, tps = _hasad_quality(m)
    name = f"Hasad {bits}-bit" + (" (per call)" if mode == "per_call" else "")
    table.append({"model": name, "file_mb": U.file_mb(U.MODELS / f"hasad_{bits}.pt"), "decode_tok_s": tps,
                  "quality": acc, "by_group": by_var})
    print(f"{name:24s} right {acc:.3f}   {tps:5.0f} tokens/s   by variety {({k: round(v, 3) for k, v in by_var.items()})}")

# %% [markdown]
# **Look at** two things. The 4-bit file is a quarter of the 32-bit file's linear weights, yet
# loaded "once" it runs at the same speed: it is float32 again in memory. Kept as codes and
# expanded per call it uses less memory and runs slower here, because plain PyTorch expands each
# weight before multiplying. Runtimes built for this (llama.cpp's kernels, for example) multiply
# directly on the blocks, which is where quantisation buys speed in a memory-bound decode.

# %% [markdown]
# ## 5 · One comparison table

# %%
print(f"{'model':26s} {'MB':>6s} {'p95 ms':>7s} {'tok/s':>6s} {'quality':>8s} {'worst group':>20s}")
for r in table:
    worst = min(r["by_group"], key=r["by_group"].get)
    print(f"{r['model']:26s} {r['file_mb']:6.2f} {r.get('p95_ms', float('nan')):7.2f} {r.get('decode_tok_s', float('nan')):6.0f} "
          f"{r['quality']:8.3f} {worst + ' ' + format(r['by_group'][worst], '.3f'):>20s}")
U.save_reference("d3_3_quant", table)

# %% [markdown]
# ## ✱ Stretch · Dynamic quantisation
#
# `onnxruntime.quantization.quantize_dynamic` quantises weights ahead of time and computes each
# activation's scale at run time: no calibration set. Quantise `waraqa_pre.onnx` dynamically
# and compare size, latency and accuracy with the static file. Which convolutions does the
# dynamic path cover on this runtime?

# %% [markdown]
# ## Check yourself
#
# 1. Why does static quantisation need calibration data, and what goes wrong with
#    unrepresentative leaves (say, only tomato)?
# 2. The 4-bit file is smaller but not faster. When would it be faster?
# 3. Overall quality fell by one point. Is the file acceptable?
#
# <details><summary>Answers</summary>
#
# 1. Activation ranges are learnt from it. Ranges from tomato leaves alone can clip or waste
#    the integer range for date palm leaves, which then lose most.
# 2. With kernels that compute on the quantised blocks, in a step bound by memory bandwidth
#    (decoding at batch 1): fewer bytes read per token.
# 3. Not yet: check it against the quality floor overall and by crop or variety, with an
#    interval (Lab 4.3).
# </details>
#
# ## Carry forward
#
# The comparison table is part of the LO5 evidence (`artifacts/d3_3_quant.json`).
