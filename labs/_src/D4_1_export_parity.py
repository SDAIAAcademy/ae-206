# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "matplotlib>=3.9", "psutil>=6.0", "torch>=2.6", "pillow>=11.0", "onnx>=1.17", "onnxscript>=0.3", "onnxruntime>=1.21"]
# ///
# %% [markdown]
# # Lab 4.1 · Export and check parity
#
# Day 4 · Session 1 (Export to a portable runtime) · 25 min.
#
# The farm-office laptops will run Waraqa in ONNX Runtime, not in PyTorch: a smaller install,
# no Python training stack, the same file on Windows, Linux and phones. Lina wrote an export
# and a runtime script and reports that "the ONNX model gives different answers". You find out
# why, export properly, and prove that the runtime's outputs match PyTorch's within a stated
# tolerance.
#
# **Tasks**
# - [ ] Run the supplied export · 4 min
# - [ ] Find why the outputs differ · 6 min
# - [ ] Export with a dynamic batch axis · 4 min
# - [ ] Parity within tolerance on 500 leaves · 6 min
# - [ ] Graph optimisation levels · 5 min
# - [ ] ✱ Stretch · Count the nodes of the graph before and after optimisation
#
# AI policy for this lab: AI-tutor. An assistant may explain an exporter error; the diagnosis is
# yours.

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
import onnx
import onnxruntime as ort
import torch
import nama as U

model = U.load_model("waraqa")
x_test, y_test, _ = U.leaves("test", U.pick(2000, 600))
OUT = U.ARTIFACTS / "d4_1"
OUT.mkdir(parents=True, exist_ok=True)
TOL = 1e-4              # largest absolute difference of a logit we accept
print(f"mode {U.MODE}: onnx {onnx.__version__}, onnxruntime {ort.__version__}")

# %% [markdown]
# ## 1 · Run the supplied export
#
# Lina's two functions, as she wrote them. Run them and compare the predicted classes with
# PyTorch's on 64 leaves.

# %%
def lina_export(model, path):
    x = torch.randn(1, 3, 64, 64)
    torch.onnx.export(model, (x,), str(path), input_names=["image"], output_names=["logits"],
                      opset_version=18, dynamo=False)
    return path


def lina_runtime(path, images_uint8):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = []
    for img in images_uint8:                        # one leaf at a time
        x = (img.astype(np.float32) / 255).transpose(2, 0, 1)[None]
        out.append(sess.run(None, {"image": x})[0][0])
    return np.stack(out)


lina_path = lina_export(model, OUT / "lina.onnx")
lina_logits = lina_runtime(lina_path, x_test[:64])
with torch.no_grad():
    torch_logits = model(U.to_input(x_test[:64])).numpy()
agree = (lina_logits.argmax(1) == torch_logits.argmax(1)).mean()
print(f"same class on {agree:.0%} of 64 leaves; largest logit difference {np.abs(lina_logits - torch_logits).max():.2f}")

# %% [markdown]
# ## 2 · Find why the outputs differ
#
# Parity has two halves: the **model** inside the file and the **preprocessing** around it.
# Test them separately: give the runtime exactly the tensor PyTorch got. If the outputs then
# match, the file is right and the preprocessing is wrong.
#
# ✏️ **Exercise** · Return the runtime's logits for the first 64 leaves, computed from
# `U.to_input(...)` (the course's preprocessing) one leaf at a time with Lina's file.

# %%
def runtime_with_course_input(path, images_uint8):
    # BEGIN SOLUTION
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return np.stack([sess.run(None, {"image": U.to_input(img[None]).numpy()})[0][0] for img in images_uint8])
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
try:
    fixed = runtime_with_course_input(lina_path, x_test[:64])
except NotImplementedError:
    s = ort.InferenceSession(str(lina_path), providers=["CPUExecutionProvider"])
    fixed = np.stack([s.run(None, {"image": U.to_input(i[None]).numpy()})[0][0] for i in x_test[:64]])
print(f"with the course's preprocessing: largest logit difference {np.abs(fixed - torch_logits).max():.2e}")
assert np.abs(fixed - torch_logits).max() < TOL
print("Lina's runtime divides by 255 but never subtracts the mean 0.5 and divides by 0.25: "
      "the model was trained on inputs centred on zero.")

# %% [markdown]
# **Look at** the safest fix: put the preprocessing **inside** the exported file, so that the
# runtime gives the file raw pixels and nobody can forget a step again. Ask what else differs
# between the training pipeline and the field app: colour order, resizing filter, JPEG
# compression. Each is a parity question.

# %% [markdown]
# ## 3 · Export with a dynamic batch axis
#
# Lina's file was traced with one leaf, so its batch axis is fixed at 1. Try a batch of 8:

# %%
sess_lina = ort.InferenceSession(str(lina_path), providers=["CPUExecutionProvider"])
try:
    sess_lina.run(None, {"image": U.to_input(x_test[:8]).numpy()})
    print("batch 8 ran")
except Exception as e:
    print("batch 8 fails:", type(e).__name__, str(e).splitlines()[0][:120])

# %% [markdown]
# ✏️ **Exercise** · Export `model` (in evaluation mode) to `path` with input name `image`,
# output name `logits`, opset 18, `dynamo=False`, and a dynamic axis 0 named `batch` on both
# the input and the output. Return the path.

# %%
def export(model, path):
    # BEGIN SOLUTION
    model.eval()
    torch.onnx.export(model, (torch.randn(2, 3, 64, 64),), str(path), input_names=["image"], output_names=["logits"],
                      opset_version=18, dynamo=False,
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
    return path
    # END SOLUTION

# %%
try:
    path = export(model, OUT / "waraqa.onnx")
except NotImplementedError:
    path = U.export_onnx(model, OUT / "waraqa.onnx")
onnx.checker.check_model(onnx.load(str(path)))
sess = U.onnx_session(path)
print("input shape in the file:", sess.get_inputs()[0].shape)
assert sess.run(None, {"image": U.to_input(x_test[:8]).numpy()})[0].shape == (8, 10)

# %% [markdown]
# ## 4 · Parity within tolerance on 500 leaves
#
# ✏️ **Exercise** · Return `(largest absolute logit difference, share of leaves with the same
# class)` between PyTorch and the runtime session on the leaves `x`.

# %%
def parity(model, sess, x):
    # BEGIN SOLUTION
    with torch.no_grad():
        a = torch.cat([model(U.to_input(x[i:i + 100])) for i in range(0, len(x), 100)]).numpy()
    b = U.onnx_logits(sess, x, batch=100)
    return float(np.abs(a - b).max()), float((a.argmax(1) == b.argmax(1)).mean())
    # END SOLUTION

# %%
xp = x_test[:500]
try:
    max_diff, same = parity(model, sess, xp)
except NotImplementedError:
    with torch.no_grad():
        a = torch.cat([model(U.to_input(xp[i:i + 100])) for i in range(0, len(xp), 100)]).numpy()
    b = U.onnx_logits(sess, xp, batch=100)
    max_diff, same = float(np.abs(a - b).max()), float((a.argmax(1) == b.argmax(1)).mean())
print(f"{len(xp)} leaves: largest logit difference {max_diff:.2e} (tolerance {TOL:.0e}), same class on {same:.1%}")
assert max_diff < TOL and same == 1.0

# %% [markdown]
# ## 5 · Graph optimisation levels
#
# ONNX Runtime rewrites the graph when it opens a session: it folds constants, fuses a
# convolution with the batch norm and activation after it (Lab 2.4 did one of these by hand),
# and more at higher levels. Time batch 1 at each level.

# %%
leaf = U.to_input(x_test[:1]).numpy()
levels = {}
for level in ("none", "basic", "extended", "all"):
    s = U.onnx_session(path, level=level)
    levels[level] = U.summarise(U.time_ms(lambda: s.run(None, {"image": leaf}), n=U.pick(200, 60), warmup=10))
    print(f"{level:9s} median {levels[level]['median']:6.3f} ms   p95 {levels[level]['p95']:6.3f} ms")
with torch.no_grad():
    t_torch = U.summarise(U.time_ms(lambda: model(U.to_input(x_test[:1])), n=U.pick(200, 60), warmup=10))
print(f"PyTorch eager median {t_torch['median']:6.3f} ms   p95 {t_torch['p95']:6.3f} ms")
U.save_reference("d4_1_export", {"max_diff": max_diff, "same": same, "levels": levels, "torch": t_torch})

# %% [markdown]
# ## ✱ Stretch · Count the nodes
#
# Set `SessionOptions.optimized_model_filepath` to save the optimised graph, open both files
# with `onnx.load` and count the nodes by type (`collections.Counter(n.op_type for n in
# m.graph.node)`). Which node types disappeared?

# %% [markdown]
# ## Check yourself
#
# 1. The runtime's classes differ from PyTorch's. What are the two halves you test separately?
# 2. Why is a tolerance on logits, not exact equality, the right test?
# 3. Why export with a dynamic batch axis when the office serves one leaf at a time?
#
# <details><summary>Answers</summary>
#
# 1. The model inside the file (feed both the identical tensor) and the preprocessing around
#    it (compare the tensors each pipeline builds from the same image).
# 2. The runtime fuses and reorders floating-point operations; results agree to rounding, not
#    bit for bit.
# 3. The night-time archive re-scoring batches (Lab 3.1), and a benchmark that varies the batch
#    needs the same file.
# </details>
#
# ## Carry forward
#
# The exported file and its parity check are LO5's export evidence. The course supplies the
# same export as `data/models/waraqa.onnx` for the next labs.
