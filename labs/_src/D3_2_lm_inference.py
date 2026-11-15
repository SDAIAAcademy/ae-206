# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "matplotlib>=3.9", "psutil>=6.0", "torch>=2.6", "pillow>=11.0"]
# ///
# %% [markdown]
# # Lab 3.2 · Hasad on a laptop
#
# Day 3 · Session 2 (Language model inference on a laptop) · 25 min.
#
# Hasad answers farmers' questions on the farm-office laptop. It is a small open-weight
# language model the course trained for Nama (1.3 million parameters, one token per
# character), shipped in `data/models/`: nothing is downloaded. You measure its two phases
# (reading the prompt, then writing the answer), size its key-value cache, and decide which
# questions may ever leave the laptop.
#
# **Tasks**
# - [ ] Time to first token at three context lengths · 6 min
# - [ ] Tokens per second while decoding · 4 min
# - [ ] Size the key-value cache by hand · 5 min
# - [ ] Check the estimate against the cache tensors · 4 min
# - [ ] Route questions by classification · 6 min
# - [ ] ✱ Stretch · Decode without the cache and measure what it costs
#
# AI policy for this lab: AI-off for the cache estimate (task 3); AI-tutor for the rest. The
# farmer questions are synthetic; no real person is described.

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

import re
import numpy as np
import torch
import nama as U

hasad = U.load_hasad(32)
cfg = hasad.cfg
qs = U.question_set(U.pick(60, 20))
CONTEXTS = (64, 256, 448)            # characters of farm notes before the question
REPS = U.pick(10, 4)
print(f"mode {U.MODE}: Hasad {U.n_params(hasad):,} parameters, config {cfg}")

# %% [markdown]
# ## 1 · Time to first token at three context lengths
#
# **Prefill** reads the whole prompt in one pass and produces the first token: its time is the
# time to first token (TTFT). Farm agents paste the farm's notes before the question, so the
# prompt grows. `hasad.generate(prompt)` returns `ttft_s`, `decode_tok_s`, the prompt length in
# tokens and the cache's bytes.
#
# ✏️ **Exercise** · For a context length `n`, build `REPS` prompts (farm notes of `n`
# characters, `U.farm_notes(n, seed=i)`, then a question from `qs`, with `U.prompt_for`), run
# `hasad.generate` on each, and return the list of results.

# %%
def run_context(n):
    # BEGIN SOLUTION
    out = []
    for i in range(REPS):
        q = qs[i % len(qs)]["question"]
        out.append(hasad.generate(U.prompt_for(q, U.farm_notes(n, seed=i))))
    return out
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
hasad.generate(U.prompt_for(qs[0]["question"]))          # warm-up
try:
    runs = {n: run_context(n) for n in CONTEXTS}
except NotImplementedError:
    runs = {n: [hasad.generate(U.prompt_for(qs[i % len(qs)]["question"], U.farm_notes(n, seed=i))) for i in range(REPS)] for n in CONTEXTS}
print(f"{'notes':>6s} {'prompt tokens':>14s} {'TTFT ms':>8s} {'decode tok/s':>13s} {'KV cache kB':>12s}")
for n in CONTEXTS:
    r = runs[n]
    print(f"{n:6d} {np.median([x['prompt_tokens'] for x in r]):14.0f} {np.median([x['ttft_s'] for x in r]) * 1000:8.1f} "
          f"{np.median([x['decode_tok_s'] for x in r]):13.0f} {np.median([x['kv_bytes'] for x in r]) / 1000:12.0f}")

# %% [markdown]
# ## 2 · Tokens per second while decoding
#
# **Decode** writes one token per pass, reading every weight and the whole cache each time:
# at batch 1 that is memory-bound (Lab 1.1). Look at how decode speed and TTFT change with the
# context in the table above.

# %%
ttft = {n: float(np.median([x["ttft_s"] for x in runs[n]]) * 1000) for n in CONTEXTS}
tps = {n: float(np.median([x["decode_tok_s"] for x in runs[n]])) for n in CONTEXTS}
print(f"TTFT grows {ttft[CONTEXTS[-1]] / ttft[CONTEXTS[0]]:.1f} x from {CONTEXTS[0]} to {CONTEXTS[-1]} characters of notes; "
      f"decode speed changes {tps[CONTEXTS[-1]] / tps[CONTEXTS[0]] - 1:+.0%}")

# %% [markdown]
# ## 3 · Size the key-value cache by hand (AI-off)
#
# For every layer the cache keeps one key vector and one value vector per token. Each vector
# has `n_head x head_dim = d` numbers (Hasad: `d` = 128, 6 layers, float32).
#
# ✏️ **Exercise** · Return the cache's bytes for `tokens` tokens.

# %%
def kv_bytes(tokens, n_layer, d, bytes_per_number=4):
    # BEGIN SOLUTION
    return 2 * n_layer * d * tokens * bytes_per_number
    # END SOLUTION

# %%
try:
    assert kv_bytes(1, 1, 1) == 8
except NotImplementedError:
    def kv_bytes(tokens, n_layer, d, bytes_per_number=4):
        return 2 * n_layer * d * tokens * bytes_per_number

# %% [markdown]
# ## 4 · Check the estimate against the cache tensors
#
# The cache at the end holds the prompt and every generated token but the last one, whose key
# and value were never needed.

# %%
rows = []
for n in CONTEXTS:
    r = runs[n][0]
    tokens = r["prompt_tokens"] + r["new_tokens"] - 1
    est = kv_bytes(tokens, cfg["n_layer"], cfg["d"])
    rows.append({"notes": n, "tokens": tokens, "estimate_bytes": est, "measured_bytes": r["kv_bytes"]})
    print(f"notes {n:4d}: {tokens:4d} tokens, estimate {est / 1000:7.1f} kB, cache tensors {r['kv_bytes'] / 1000:7.1f} kB")
    assert est == r["kv_bytes"], (est, r["kv_bytes"])

big = kv_bytes(4096, 16, 8 * 64, bytes_per_number=2)
print(f"for scale: a model with 16 layers and 8 key-value heads of 64 numbers, 4,096 tokens in 16-bit: {big / 1e6:.0f} MB per conversation")
U.save_reference("d3_2_hasad", {"ttft_ms": ttft, "decode_tok_s": tps, "kv": rows})

# %% [markdown]
# **Look at** the last line. Hasad's cache is tiny because the model is tiny. A real assistant
# model's cache grows to hundreds of megabytes per conversation, and on a laptop with 8 GB the
# cache, not the weights, often decides how many conversations or how long a context fits.

# %% [markdown]
# ## 5 · Route questions by classification
#
# Nama could send easy questions to a hosted model and answer the rest on the laptop. Reham
# Al-Barakati, Nama's data protection officer, set the rule: Nama applies the four levels of the
# NDMO Data Classification Policy (Top Secret, Secret, Restricted, Public) to farmer questions.
# The policy lists personally identifiable information such as names, addresses, phone
# numbers and identity numbers among its examples of Restricted data. Restricted questions are
# answered **on the device only**; Public ones may go to a hosted model.
#
# ✏️ **Exercise** · Return `"Restricted"` when a question contains a Saudi mobile number
# (05 followed by eight digits), a ten-digit identity number starting with 1 or 2, a person
# introducing themselves ("أنا" followed by a name and a comma) or a named farm ("مزرعة"
# followed by a name); otherwise `"Public"`.

# %%
def classify(question):
    # BEGIN SOLUTION
    patterns = [r"05\d{8}", r"\b[12]\d{9}\b", r"أنا\s+\S+،", r"مزرعة\s+\S+"]
    return "Restricted" if any(re.search(p, question) for p in patterns) else "Public"
    # END SOLUTION

# %%
allq = U.farmer_questions()
try:
    pred = [classify(q["question"]) for q in allq]
except NotImplementedError:
    def classify(question):
        pats = [r"05\d{8}", r"\b[12]\d{9}\b", r"أنا\s+\S+،", r"مزرعة\s+\S+"]
        return "Restricted" if any(re.search(p, question) for p in pats) else "Public"
    pred = [classify(q["question"]) for q in allq]
truth = [q["classification"] for q in allq]
leaks = sum(p == "Public" and t == "Restricted" for p, t in zip(pred, truth))
kept = sum(p == "Restricted" and t == "Public" for p, t in zip(pred, truth))
n_restricted = truth.count("Restricted")
print(f"{len(allq)} questions, {n_restricted} Restricted by the answer key")
print(f"would have left the device although Restricted: {leaks}   kept on the device although Public: {kept}")
print(f"share answered on the device: {pred.count('Restricted') / len(allq):.1%}")
assert leaks == 0, "a Restricted question would leave the device"
U.save_reference("d3_2_routing", {"leaks": leaks, "kept_public": kept, "restricted": n_restricted})

# %% [markdown]
# **Look at** the asymmetry. A Public question kept on the device costs a little speed; a
# Restricted question sent out is a breach of Nama's rule. A rule that errs is tuned to err
# on the side of the device. Patterns catch the formats they know: a phone number written in
# words passes through, which is why such a router is a control to test, not a guarantee.

# %% [markdown]
# ## ✱ Stretch · Without the cache
#
# `hasad.generate(prompt, use_cache=False)` recomputes the whole sequence for every new token.
# Measure decode tokens per second at 448 characters of notes with and without the cache. How
# does the gap grow with the context, and why?

# %% [markdown]
# ## Check yourself
#
# 1. TTFT grew with the context and decode speed fell far less. Why do the two behave
#    differently?
# 2. A model has 32 layers, 8 key-value heads of 128 numbers, 16-bit numbers. How large is
#    the cache for 8,000 tokens?
# 3. Why is a question with a phone number kept on the device even if the hosted model is
#    more accurate?
#
# <details><summary>Answers</summary>
#
# 1. Prefill processes every prompt token, so it grows with the prompt; each decode step
#    processes one new token and reads a cache that grows slowly by comparison.
# 2. 2 x 32 x 1,024 x 8,000 x 2 bytes, about 1.05 GB.
# 3. Nama's rule, taken from the NDMO levels, classifies it Restricted; accuracy does not
#    change the classification of the data.
# </details>
#
# ## Carry forward
#
# TTFT, decode speed and the cache table are LO3's evidence for the language model
# (`artifacts/d3_2_hasad.json`). Lab 3.3 quantises Hasad and measures the same numbers again.
