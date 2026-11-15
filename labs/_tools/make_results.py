"""Measure every Waraqa variant on this machine and write data/supplied/results_table.csv,
the table Lab 4.4 starts from (run once by the course author, on a quiet machine).

    cd labs && uv run python _tools/make_results.py

Each row: batch 1, four threads, 300 timed calls after 30 warm-up calls on a fixed mix of 40
held-out leaves; peak memory of a serving process started for the purpose (the runtime, the
model and 100 calls, nothing else), read from the operating system's peak counter; accuracy
on the 2,000 held-out leaves and the paired difference against the 32-bit model, overall and
on the worst crop.
"""
import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

import nama as U

torch.set_num_threads(4)
x, y, crop = U.leaves("test")
SHARE = [0.35, 0.30, 0.25, 0.10]
mix_idx = np.array(sorted(sum((list(np.flatnonzero(crop == k)[: round(40 * s)]) for k, s in enumerate(SHARE)), [])))
mix = U.to_input(x[mix_idx]).numpy()
U.CACHE.mkdir(parents=True, exist_ok=True)
np.save(U.CACHE / "results_mix.npy", mix)

PREAMBLE = """
import sys
import numpy as np
import psutil
mix = np.load(sys.argv[1])
"""
REPORT = """
m = psutil.Process().memory_info()
if hasattr(m, "peak_wset"):
    print("PEAK", m.peak_wset / 1e6)
else:
    import resource
    print("PEAK", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3)
"""
SERVE_ONNX = """
import onnxruntime as ort
o = ort.SessionOptions()
o.intra_op_num_threads = 4
s = ort.InferenceSession(sys.argv[2], o, providers=["CPUExecutionProvider"])
for i in range(100):
    s.run(None, {"image": mix[i % 40][None]})
"""
SERVE_TORCH = """
import torch
import nama as U
torch.set_num_threads(4)
m = U.load_model("waraqa")
with torch.no_grad():
    for i in range(100):
        m(torch.from_numpy(mix[i % 40][None]))
"""


def serving_peak_mb(body, model_path=""):
    out = subprocess.run([sys.executable, "-c", PREAMBLE + body + REPORT, str(U.CACHE / "results_mix.npy"), str(model_path)],
                         cwd=U.LABS, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("PEAK "):
            return float(line[5:])
    raise RuntimeError(out.stderr[-2000:])


def timing(run):
    ms = []
    for i in range(30):
        run(mix[i % 40][None])
    for i in range(300):
        ms.append(U.time_ms(lambda: run(mix[i % 40][None]), n=1, warmup=0)[0])
    lo, hi = U.bootstrap_ci(ms, lambda a: np.percentile(a, 95))
    return float(np.median(ms)), float(np.percentile(ms, 95)), hi


def onnx_row(name, file):
    path = U.MODELS / file
    peak = serving_peak_mb(SERVE_ONNX, path)
    s = U.onnx_session(path)
    med, p95, p95hi = timing(lambda a: s.run(None, {"image": a}))
    pred = U.onnx_predict(s, x)
    return {"variant": name, "runtime": "ONNX Runtime", "file_mb": U.file_mb(path), "median_ms": med, "p95_ms": p95,
            "p95_ci_high": p95hi, "peak_mb": peak, "right": pred == y}


def torch_row():
    path = U.MODELS / "waraqa.pt"
    peak = serving_peak_mb(SERVE_TORCH)
    m = U.load_model("waraqa")
    with torch.no_grad():
        med, p95, p95hi = timing(lambda a: m(torch.from_numpy(a)))
    return {"variant": "Waraqa 32-bit", "runtime": "PyTorch eager", "file_mb": U.file_mb(path), "median_ms": med,
            "p95_ms": p95, "p95_ci_high": p95hi, "peak_mb": peak, "right": U.predict(m, x) == y}


rows = [torch_row(), onnx_row("Waraqa 32-bit", "waraqa.onnx"), onnx_row("Waraqa 8-bit", "waraqa_int8.onnx"),
        onnx_row("Waraqa pruned", "waraqa_pruned.onnx"), onnx_row("Waraqa-S", "waraqa_s.onnx")]
orig = rows[0]["right"]
mach = U.machine()
machine = f"{mach['cpu']}, {mach['threads_used']} threads, {mach['os'].split('-')[0]}"
out = []
for r in rows:
    diffs = {c: float(r["right"][crop == k].mean() - orig[crop == k].mean()) for k, c in enumerate(U.CROPS)}
    worst = min(diffs, key=diffs.get)
    out.append({"variant": r["variant"], "runtime": r["runtime"], "file_mb": round(r["file_mb"], 3),
                "median_ms": round(r["median_ms"], 3), "p95_ms": round(r["p95_ms"], 3), "p95_ci_high": round(r["p95_ci_high"], 3),
                "peak_mb": round(r["peak_mb"], 1), "accuracy": round(float(r["right"].mean()), 4),
                "diff_overall": round(float(r["right"].mean() - orig.mean()), 4), "worst_crop": worst,
                "worst_crop_diff": round(diffs[worst], 4), "machine": machine})
U.SUPPLIED.mkdir(parents=True, exist_ok=True)
with open(U.SUPPLIED / "results_table.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
for r in out:
    print(r)
