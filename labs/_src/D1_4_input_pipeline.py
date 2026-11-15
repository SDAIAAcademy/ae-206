# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "matplotlib>=3.9", "psutil>=6.0", "torch>=2.6", "pillow>=11.0"]
# ///
# %% [markdown]
# # Lab 1.4 · Fix the input pipeline
#
# Day 1 · Session 4 (Fixing an input-bound pipeline) · 25 min.
#
# Lab 1.3 found that Lina's step waits for its data. Here you remove that wait, one change at
# a time, and measure after each: parallel workers, a cache of decoded images, and a faster
# file format. A change you did not measure did not happen.
#
# **Tasks**
# - [ ] Baseline: step time and loader time · 4 min
# - [ ] Parallel workers · 6 min
# - [ ] Cache the decoded images · 6 min
# - [ ] A faster image format · 5 min
# - [ ] The before-and-after table · 4 min
# - [ ] ✱ Stretch · Find the worker count after which the step stops getting faster
#
# AI policy for this lab: AI-tutor. An assistant may explain a DataLoader option; the
# measurements are yours.

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

import io
import time
import numpy as np
import torch
from PIL import Image
import nama as U
import lina_train as L

folder = U.field_export()             # the same 224-pixel PNG export as Lab 1.3
labels = np.array([int(r.split(",")[1]) for r in (folder / "labels.csv").read_text().splitlines()[1:]])
STEPS = U.pick(12, 6)
WORKERS = min(4, max(1, (os.cpu_count() or 2) - 1))
results = {}


def measure(name, loader):
    """Median ms per step and per loading phase over STEPS steps (after two start-up steps)."""
    ph = L.run(loader, STEPS)
    results[name] = {"step_ms": float(np.median([sum(p.values()) for p in ph]) * 1000),
                     "loading_ms": float(np.median([p["loading"] for p in ph]) * 1000)}
    print(f"{name:28s} step {results[name]['step_ms']:7.0f} ms   loading {results[name]['loading_ms']:7.0f} ms")


print(f"mode {U.MODE}: {len(labels)} leaves, {STEPS} timed steps, up to {WORKERS} workers")

# %% [markdown]
# ## 1 · Baseline: step time and loader time

# %%
measure("baseline (Lina)", L.make_loader(folder))

# %% [markdown]
# ## 2 · Parallel workers
#
# With `num_workers=k` the DataLoader starts k processes that prepare the next batches while
# the main process trains. On Windows and macOS each worker starts a fresh Python, so the first
# batch is slow; `persistent_workers` keeps them alive between epochs, and `L.run` skips the
# first two steps.
#
# ✏️ **Exercise** · Return Lina's loader with `k` workers (use `L.make_loader`).

# %%
def parallel_loader(k):
    # BEGIN SOLUTION
    return L.make_loader(folder, workers=k)
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    loader = parallel_loader(WORKERS)
except NotImplementedError:
    loader = L.make_loader(folder, workers=WORKERS)
assert loader.num_workers == WORKERS
measure(f"{WORKERS} workers", loader)
del loader

# %% [markdown]
# **Look at** the loading time. Workers hide loading behind compute; they do not make it
# cheaper. The machine still decodes every PNG, now on more cores, which the training step
# also wanted.

# %% [markdown]
# ## 3 · Cache the decoded images
#
# Every epoch decodes the same files again. Decode each leaf once, shrink it to 64 pixels, and
# store all of them in one `uint8` array on disk: 64 x 64 x 3 bytes per leaf. Training then
# reads bytes, not PNG files.
#
# ✏️ **Exercise** · Decode every PNG in `folder` (in the order of `labels.csv`), resize to
# 64 x 64 with `Image.BILINEAR`, stack into an array of shape (n, 64, 64, 3) and dtype uint8,
# save it with `np.save` to `path` and return the path.

# %%
def build_cache(folder, path):
    # BEGIN SOLUTION
    names = [r.split(",")[0] for r in (folder / "labels.csv").read_text().splitlines()[1:]]
    arr = np.stack([np.array(Image.open(folder / n).convert("RGB").resize((64, 64), Image.BILINEAR)) for n in names])
    np.save(path, arr)
    return path
    # END SOLUTION

# %%
cache = U.CACHE / f"decoded_{len(labels)}.npy"
t0 = time.perf_counter()
try:
    build_cache(folder, cache)
except NotImplementedError:
    names = [r.split(",")[0] for r in (folder / "labels.csv").read_text().splitlines()[1:]]
    np.save(cache, np.stack([np.array(Image.open(folder / n).convert("RGB").resize((64, 64), Image.BILINEAR)) for n in names]))
build_s = time.perf_counter() - t0
arr = np.load(cache, mmap_mode="r")
assert arr.shape == (len(labels), 64, 64, 3) and arr.dtype == np.uint8
print(f"cache built once in {build_s:.1f} s: {cache.stat().st_size / 1e6:.1f} MB for {len(labels)} leaves")

measure("cache, main process", L.make_loader(dataset=U.CachedLeaves(cache, labels)))
measure(f"cache, {WORKERS} workers", L.make_loader(dataset=U.CachedLeaves(cache, labels), workers=WORKERS))

# %% [markdown]
# **Look at** two things. The cache changed the augmentation too: Lina rotated each leaf by
# any angle at full size, the cache rotates by multiples of 90 degrees. A speed-up that
# changes what the model learns must be checked for quality (Day 4 does that properly). And
# with the cache, are workers still worth their start-up cost?

# %% [markdown]
# ## 4 · A faster image format
#
# The cache works for training. New leaves still arrive from the field, so the format they
# arrive in matters. Compare, per leaf, the decode time and file size of the 224-pixel PNG with
# a 64-pixel JPEG of quality 90 made from it.
#
# ✏️ **Exercise** · Return `(ms_per_decode, bytes)` of decoding `data` (the bytes of one
# image file) `n` times with `Image.open(io.BytesIO(data)).convert("RGB")`.

# %%
def decode_cost(data, n=50):
    # BEGIN SOLUTION
    t0 = time.perf_counter()
    for _ in range(n):
        Image.open(io.BytesIO(data)).convert("RGB")
    return (time.perf_counter() - t0) / n * 1000, len(data)
    # END SOLUTION

# %%
png = (folder / "leaf_00000.png").read_bytes()
buf = io.BytesIO()
Image.open(io.BytesIO(png)).convert("RGB").resize((64, 64), Image.BILINEAR).save(buf, "JPEG", quality=90)
jpg = buf.getvalue()
try:
    fmt = {"png_224": decode_cost(png), "jpeg_64": decode_cost(jpg)}
except NotImplementedError:
    def decode_cost(data, n=50):
        t0 = time.perf_counter()
        for _ in range(n):
            Image.open(io.BytesIO(data)).convert("RGB")
        return (time.perf_counter() - t0) / n * 1000, len(data)
    fmt = {"png_224": decode_cost(png), "jpeg_64": decode_cost(jpg)}
for k, (ms, b) in fmt.items():
    print(f"{k:8s} {ms:6.3f} ms per decode   {b / 1000:6.1f} kB")
print(f"64-pixel JPEG: {fmt['png_224'][0] / fmt['jpeg_64'][0]:.0f} x faster to decode, "
      f"{fmt['png_224'][1] / fmt['jpeg_64'][1]:.0f} x smaller")

# %% [markdown]
# **Look at** what JPEG costs: it is lossy. Compare the decoded pixels of the two versions of
# the same leaf (mean absolute difference below). Whether that loss matters is a quality
# question, answered by evaluating the model on JPEG leaves, not by looking.

# %%
a = np.array(Image.open(io.BytesIO(png)).convert("RGB").resize((64, 64), Image.BILINEAR), dtype=float)
b = np.array(Image.open(io.BytesIO(jpg)).convert("RGB"), dtype=float)
print(f"mean absolute pixel difference PNG-resized vs JPEG: {np.abs(a - b).mean():.2f} of 255")

# %% [markdown]
# ## 5 · The before-and-after table

# %%
base = results["baseline (Lina)"]["step_ms"]
print(f"{'pipeline':28s} {'step ms':>8s} {'loading ms':>11s} {'speed-up':>9s}")
for k, v in results.items():
    print(f"{k:28s} {v['step_ms']:8.0f} {v['loading_ms']:11.0f} {base / v['step_ms']:8.1f}x")
U.save_reference("d1_4_pipeline", {"results": results, "formats": fmt, "cache_build_s": build_s})

# %% [markdown]
# **Look at** the last row against Lab 1.3's ceiling: the step cannot get faster than forward,
# backward and optimiser together. When loading is near zero, the step is compute-bound, and
# the next speed-ups (Day 2) work on compute and memory.

# %% [markdown]
# ## ✱ Stretch · How many workers?
#
# Measure the cached loader and Lina's loader with 1, 2, 4 and 8 workers. Where does each
# stop improving, and why are the two answers different?

# %% [markdown]
# ## Check yourself
#
# 1. Adding workers made loading 6 times faster and the step only 1.5 times faster. Why?
# 2. What did the cache change besides speed?
# 3. When is a cache of preprocessed data the wrong answer?
#
# <details><summary>Answers</summary>
#
# 1. Once loading hides behind compute, the step is as slow as forward plus backward plus
#    optimiser; and the workers took cores from the training step.
# 2. The augmentation (90-degree rotations instead of any angle) and therefore possibly the
#    model's accuracy. It must be re-evaluated.
# 3. When preprocessing must differ each epoch (random crops at full resolution), when the data
#    does not fit on disk, or when the data changes faster than the cache is rebuilt.
# </details>
#
# ## Carry forward
#
# The table is your LO2 evidence: a before-and-after profile and a measured speed-up with the
# change explained. It is in `artifacts/d1_4_pipeline.json`.
