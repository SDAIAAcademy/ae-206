# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "onnxruntime>=1.21", "psutil>=6.0"]
# ///
"""The course's timing script: benchmark one ONNX model on this machine and write a JSON report.

    uv run bench.py data/models/waraqa.onnx --inputs artifacts/mix.npy --batch 1 --reps 300
    uv run bench.py <model.onnx> --label "my variant" --out artifacts/bench_mine.json

The input mix is a float32 .npy array whose first axis is the items (for Waraqa: N x 3 x 64 x
64, already preprocessed); the script cycles through it in a fixed order so that every run sees
the same inputs. Without --inputs it uses random normal data of the model's input shape and says
so in the report. It warms up, times each call with a monotonic clock, and reports the median,
95th and 99th percentiles with 95% bootstrap intervals, plus the machine and the settings.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np


def bootstrap(values, q, n_boot=2000, seed=206):
    a = np.asarray(values)
    rng = np.random.default_rng(seed)
    boots = np.percentile(a[rng.integers(0, a.size, (n_boot, a.size))], q, axis=1)
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def machine(threads):
    import psutil
    cpu = platform.processor() or platform.machine()
    if sys.platform == "win32":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif sys.platform.startswith("linux"):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    import onnxruntime as ort
    return {"cpu": cpu, "logical_cores": os.cpu_count(), "threads": threads, "os": platform.platform(),
            "ram_gb": round(psutil.virtual_memory().total / 1e9, 1), "onnxruntime": ort.__version__,
            "cpu_load_percent_at_start": psutil.cpu_percent(interval=0.5)}


def run(model, inputs=None, batch=1, reps=300, warmup=30, threads=4):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    if inputs is None:
        shape = [batch if not isinstance(d, int) else d for d in inp.shape]
        mix = np.random.default_rng(0).standard_normal([max(batch, 8)] + shape[1:]).astype(np.float32)
        source = "random normal (no --inputs given)"
    else:
        mix = np.load(inputs).astype(np.float32)
        source = str(inputs)
    batches = [mix[(i * batch) % len(mix):(i * batch) % len(mix) + batch] for i in range(max(1, len(mix) // batch))]
    batches = [b for b in batches if len(b) == batch]
    for i in range(warmup):
        sess.run(None, {inp.name: batches[i % len(batches)]})
    ms = []
    for i in range(reps):
        b = batches[i % len(batches)]
        t0 = time.perf_counter()
        sess.run(None, {inp.name: b})
        ms.append((time.perf_counter() - t0) * 1000)
    a = np.asarray(ms)
    return {"model": str(model), "file_mb": Path(model).stat().st_size / 1e6, "batch": batch, "reps": reps,
            "warmup": warmup, "inputs": source, "median_ms": float(np.median(a)),
            "median_ci": bootstrap(a, 50), "p95_ms": float(np.percentile(a, 95)), "p95_ci": bootstrap(a, 95),
            "p99_ms": float(np.percentile(a, 99)), "machine": machine(threads), "ms": [round(v, 4) for v in ms]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--inputs")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--label", default="")
    ap.add_argument("--out")
    a = ap.parse_args()
    r = run(a.model, a.inputs, a.batch, a.reps, a.warmup, a.threads)
    r["label"] = a.label
    print(f"{a.label or a.model}: median {r['median_ms']:.3f} ms [{r['median_ci'][0]:.3f}, {r['median_ci'][1]:.3f}]   "
          f"p95 {r['p95_ms']:.3f} ms [{r['p95_ci'][0]:.3f}, {r['p95_ci'][1]:.3f}]   ({r['reps']} calls, batch {r['batch']})")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
