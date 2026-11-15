# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "scipy>=1.14", "matplotlib>=3.9", "psutil>=6.0", "torch>=2.6", "pillow>=11.0", "onnx>=1.17", "onnxscript>=0.3", "onnxruntime>=1.21"]
# ///
# %% [markdown]
# # Setup check
#
# Run this once before Day 1. It checks every library the labs use, builds the synthetic data
# of Nama Agritech (about a minute), loads both supplied models and says whether this machine
# can compile models (Lab 2.4). Nothing downloads a model: both are in `labs/data/models/`.

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

import importlib
missing = []
for name in ("numpy", "scipy", "matplotlib", "psutil", "torch", "PIL", "onnx", "onnxruntime"):
    try:
        m = importlib.import_module(name)
        print(f"ok  {name:12s} {getattr(m, '__version__', '')}")
    except ImportError:
        missing.append(name)
        print(f"MISSING {name}")
assert not missing, f"install: {missing} (locally: uv sync; in Colab: pip install -r requirements-colab.txt)"

# %%
import time
import torch
import nama as U

print(U.machine())
t0 = time.perf_counter()
x, y, crop = U.leaves("test", U.pick(2000, 600))
qs = U.question_set()
print(f"data: {len(x)} held-out leaves, {len(qs)} farmer questions, built in {time.perf_counter() - t0:.0f} s")

waraqa = U.load_model("waraqa")
acc = U.accuracy(waraqa, x, y)
hasad = U.load_hasad(32)
r = hasad.generate(U.prompt_for(qs[0]["question"]))
print(f"Waraqa: {U.n_params(waraqa):,} parameters, held-out accuracy {acc:.3f}")
print(f"Hasad: {U.n_params(hasad):,} parameters; first question answered: {U.grade(r['text'], qs[0]['key'])}")
assert acc > 0.8

# %%
try:
    c = torch.compile(torch.nn.Linear(8, 8))
    c(torch.randn(2, 8))
    print("torch.compile works here: Lab 2.4 measures the compiler")
except Exception as e:
    print(f"torch.compile is not available here ({type(e).__name__}): Lab 2.4 uses its fallback; "
          "a Colab runtime has a compiler")
print("ready")
