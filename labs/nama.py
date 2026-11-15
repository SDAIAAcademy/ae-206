"""Helpers for the AE-206 labs: Nama Agritech's data, its two models, timers and memory meters.

import nama as U

Data      U.leaves(split, n), U.to_input(x), U.field_export(n), U.farmer_questions()
Models    U.Waraqa, U.WaraqaS, U.load_model(name), U.accuracy(model, x, y)
          U.load_hasad(bits, mode), U.Hasad.generate(...), U.grade(answer, key)
Timing    U.time_ms(fn, n, warmup), U.summarise(ms), U.bootstrap_ci(values, stat)
Memory    U.peak_rss_mb(fn), U.saved_activation_mb(model, x, y), U.tensor_mb(...)
Export    U.export_onnx(model, path), U.onnx_session(path, threads), U.onnx_predict(...)
Others    U.pick(full, quick), U.reference(name), U.save_reference(name, obj), U.machine()

Quick mode: set NAMA_QUICK=1 for smaller runs (fewer images, steps and repetitions).
"""
from __future__ import annotations

import gc
import io
import json
import math
import os
import platform
import re
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import prepare_data as P

QUICK = os.environ.get("NAMA_QUICK", "0") == "1"
MODE = "quick" if QUICK else "full"
LABS = Path(__file__).resolve().parent
DATA = LABS / "data"
MODELS = DATA / "models"
REF = DATA / "reference"
SUPPLIED = DATA / "supplied"
CACHE = P.CACHE
ARTIFACTS = LABS / "artifacts"
CLASSES, CROPS = P.CLASSES, P.CROPS
THREADS = min(4, os.cpu_count() or 1)
torch.set_num_threads(THREADS)


def pick(full, quick):
    """The full value, or the small one in quick mode."""
    return quick if QUICK else full


def machine() -> dict:
    """What a benchmark report must name: processor, threads, memory, software versions."""
    import psutil
    try:
        import onnxruntime as ort
        ort_v = ort.__version__
    except ImportError:
        ort_v = None
    cpu = platform.processor() or platform.machine()
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    elif sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            import subprocess
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or cpu
        except OSError:
            pass
    return {"cpu": cpu, "logical_cores": os.cpu_count(), "threads_used": torch.get_num_threads(),
            "ram_gb": round(psutil.virtual_memory().total / 1e9, 1), "os": platform.platform(),
            "python": platform.python_version(), "torch": torch.__version__, "onnxruntime": ort_v,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

# ------------------------------------------------------------------ reference values


def reference(name: str):
    """A reference result, for the safety-net cells."""
    return json.loads((REF / f"{name}.json").read_text(encoding="utf-8"))


def save_reference(name: str, obj, folder: Path | None = None) -> Path:
    """Save a lab result as JSON (into labs/artifacts/ unless a folder is given)."""
    folder = folder or ARTIFACTS
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.json"
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    return path

# ------------------------------------------------------------------ data

MEAN, STD = 0.5, 0.25


def leaves(split: str, n: int | None = None):
    """(images uint8 N x 64 x 64 x 3, class, crop) of a split: train, val or test."""
    x, y, crop, _ = P.leaf_arrays(split, n)
    return x, y, crop


def to_input(x: np.ndarray) -> torch.Tensor:
    """uint8 N x H x W x 3 to the model's input: float32 N x 3 x H x W, normalised."""
    t = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float()
    return (t / 255 - MEAN) / STD


def field_export(n: int | None = None) -> Path:
    """The folder of the field app's PNG export (224 pixels), built on first use."""
    return P.field_export(n or pick(4096, 768))


def farmer_questions() -> list[dict]:
    return P.farmer_questions()


def question_set(n: int | None = None) -> list[dict]:
    """The fixed question set Hasad's quality is measured on: the first n of 2,000 (default 200)."""
    return farmer_questions()[: (n or 200)]

# ------------------------------------------------------------------ the field-export loaders
# They live here, not in a notebook, so that DataLoader workers can import them on Windows and
# macOS, where workers start as new processes.


class FieldExportSlow(torch.utils.data.Dataset):
    """Lina's loader: decode the 224-pixel PNG, augment at full size, then shrink to 64."""

    def __init__(self, folder: Path, augment: bool = True):
        rows = (Path(folder) / "labels.csv").read_text(encoding="utf-8").splitlines()[1:]
        self.folder = Path(folder)
        self.items = [(r.split(",")[0], int(r.split(",")[1])) for r in rows]
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        name, cls = self.items[i]
        img = Image.open(self.folder / name).convert("RGB")
        if self.augment:
            img = img.rotate(float((i * 37) % 360), resample=Image.BILINEAR)
        img = img.resize((64, 64), Image.BILINEAR)
        x = (torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255 - MEAN) / STD
        return x, cls


class CachedLeaves(torch.utils.data.Dataset):
    """Decoded 64-pixel images from one .npy file (uint8), flipped and rotated by 90 degrees."""

    def __init__(self, npy: Path, labels: np.ndarray, augment: bool = True):
        self.npy, self.labels, self.augment = Path(npy), np.asarray(labels), augment
        self.x = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.x is None:  # opened lazily: each worker maps the file itself
            self.x = np.load(self.npy, mmap_mode="r")
        a = np.asarray(self.x[i])
        if self.augment:
            a = np.rot90(a, k=i % 4)
        t = torch.from_numpy(np.array(a)).permute(2, 0, 1).float()
        return (t / 255 - MEAN) / STD, int(self.labels[i])

# ------------------------------------------------------------------ Waraqa, the leaf classifier


def _block(cin, cout):
    return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU()]


class Waraqa(nn.Module):
    """A small convolutional network for 64-pixel leaves: six 3x3 convolutions with batch norm."""

    def __init__(self, widths=(32, 32, 64, 64, 128, 128), n_classes=10):
        super().__init__()
        w = list(widths)
        layers = _block(3, w[0]) + _block(w[0], w[1]) + [nn.MaxPool2d(2)]
        layers += _block(w[1], w[2]) + _block(w[2], w[3]) + [nn.MaxPool2d(2)]
        layers += _block(w[3], w[4]) + [nn.MaxPool2d(2)] + _block(w[4], w[5])
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(w[5], n_classes)
        self.widths = tuple(w)

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(2, 3)))


class WaraqaS(Waraqa):
    """The smaller architecture: half the channels in every layer."""

    def __init__(self, n_classes=10):
        super().__init__(widths=(16, 16, 32, 32, 64, 64), n_classes=n_classes)


def build(name: str) -> nn.Module:
    if name.startswith("waraqa_s"):
        return WaraqaS()
    return Waraqa()


def load_model(name: str = "waraqa") -> nn.Module:
    """A supplied checkpoint from data/models/<name>.pt, in evaluation mode."""
    state = torch.load(MODELS / f"{name}.pt", map_location="cpu", weights_only=True)
    widths = state.pop("_widths", None)
    model = Waraqa(widths=tuple(int(v) for v in widths)) if widths is not None else build(name)
    model.load_state_dict(state)
    return model.eval()


def save_model(model: Waraqa, path: Path) -> Path:
    state = {k: v for k, v in model.state_dict().items()}
    state["_widths"] = torch.tensor(model.widths)
    torch.save(state, path)
    return path


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch: int = 256) -> np.ndarray:
    model.eval()
    out = [model(to_input(x[i:i + batch])).argmax(1).numpy() for i in range(0, len(x), batch)]
    return np.concatenate(out)


def accuracy(model: nn.Module, x: np.ndarray, y: np.ndarray) -> float:
    return float((predict(model, x) == y).mean())


def file_mb(path) -> float:
    return Path(path).stat().st_size / 1e6

# ------------------------------------------------------------------ timing


def time_ms(fn, n: int = 100, warmup: int = 5) -> list[float]:
    """Call fn warmup times untimed, then n times; return each call's milliseconds."""
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)
    return out


def summarise(ms) -> dict:
    a = np.asarray(ms, dtype=float)
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "min": float(a.min()), "max": float(a.max())}


def bootstrap_ci(values, stat=np.median, n_boot: int = 2000, level: float = 0.95, seed: int = 206):
    """Percentile bootstrap interval of stat(values)."""
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boots = np.array([stat(a[rng.integers(0, a.size, a.size)]) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)

# ------------------------------------------------------------------ memory


def tensor_mb(*tensors) -> float:
    return sum(t.numel() * t.element_size() for t in tensors) / 1e6


def fresh_peak_mb(batch, model="waraqa_warm", lr=1e-4):
    """Peak memory (MB above the start) of one training step at this batch, in a fresh process.

    Memory freed by an earlier step often stays with the process, so a second measurement in
    the same process can hide part of the next step. The child process loads the model, takes
    one step of two leaves so the optimiser state exists, then measures one step of `batch`."""
    import subprocess
    code = (
        "import json, torch, torch.nn.functional as F, nama as U\n"
        f"m = U.load_model({model!r}); m.train()\n"
        f"opt = torch.optim.AdamW(m.parameters(), lr={lr})\n"
        f"x, y, _ = U.leaves('val', {int(batch)})\n"
        "def step(b):\n"
        "    loss = F.cross_entropy(m(U.to_input(x[:b])), torch.from_numpy(y[:b]).long())\n"
        "    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)\n"
        "step(2)\n"
        f"_, p = U.peak_rss_mb(lambda: step({int(batch)}))\n"
        "print('PEAK', json.dumps(p))\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=LABS, capture_output=True, text=True,
                         env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    for line in out.stdout.splitlines():
        if line.startswith("PEAK "):
            return float(line[5:])
    raise RuntimeError(out.stderr[-2000:])


def _trim():
    """Give memory that is held but unused back to the operating system, so a measurement starts
    from what is in use (freed memory kept by the process would otherwise hide part of a peak)."""
    import ctypes
    try:
        if sys.platform == "win32":
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            ctypes.windll.psapi.EmptyWorkingSet(ctypes.c_void_p(k32.GetCurrentProcess()))
        elif sys.platform.startswith("linux"):
            ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def peak_rss_mb(fn):
    """Run fn and return (its result, peak resident memory above the start, in MB).

    On a GPU it reports torch.cuda.max_memory_allocated instead. A sampling thread reads the
    process's resident set every half millisecond, so a peak shorter than that can be missed."""
    import psutil
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        out = fn()
        torch.cuda.synchronize()
        return out, (torch.cuda.max_memory_allocated() - base) / 1e6
    proc = psutil.Process()
    _trim()
    base = proc.memory_info().rss
    peak, stop = [base], [False]

    def sample():
        while not stop[0]:
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.0005)

    t = threading.Thread(target=sample, daemon=True)
    t.start()
    try:
        out = fn()
    finally:
        stop[0] = True
        t.join()
    peak[0] = max(peak[0], proc.memory_info().rss)
    return out, (peak[0] - base) / 1e6


def saved_activation_mb(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Bytes the autograd graph keeps for the backward pass (each storage counted once)."""
    seen = {}

    def pack(t):
        try:
            key = t.untyped_storage().data_ptr()
            seen[key] = max(seen.get(key, 0), t.untyped_storage().nbytes())
        except RuntimeError:
            pass
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = F.cross_entropy(model(x), y)
    loss.backward()
    model.zero_grad(set_to_none=True)
    params = {p.untyped_storage().data_ptr() for p in model.parameters()}
    return sum(v for k, v in seen.items() if k not in params) / 1e6

# ------------------------------------------------------------------ training


def train_steps(model, x, y, steps, batch=64, lr=2e-4, autocast_dtype=None, seed=0, accumulate=1,
                checkpoint_segments=0, optimizer=None):
    """Train for a number of optimiser steps on uint8 images; return seconds per step (list)."""
    g = torch.Generator().manual_seed(seed)
    opt = optimizer or torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    dev = next(model.parameters()).device
    times = []
    micro = batch // accumulate
    for _ in range(steps):
        t0 = time.perf_counter()
        idx = torch.randint(0, len(x), (batch,), generator=g).numpy()
        xb, yb = to_input(x[idx]).to(dev), torch.from_numpy(y[idx]).long().to(dev)
        for k in range(accumulate):
            xs, ys = xb[k * micro:(k + 1) * micro], yb[k * micro:(k + 1) * micro]
            ctx = torch.autocast(dev.type, dtype=autocast_dtype) if autocast_dtype else _null()
            with ctx:
                if checkpoint_segments:
                    from torch.utils.checkpoint import checkpoint_sequential
                    feats = checkpoint_sequential(model.features, checkpoint_segments, xs, use_reentrant=False)
                    out = model.head(feats.mean(dim=(2, 3)))
                else:
                    out = model(xs)
                loss = F.cross_entropy(out.float(), ys) / accumulate
            loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    model.eval()
    return times


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

# ------------------------------------------------------------------ export and runtime


def export_onnx(model: nn.Module, path, dynamic_batch: bool = True, size: int = 64) -> Path:
    """Export to ONNX with a dynamic batch axis (the model must already be in eval mode)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = torch.randn(2, 3, size, size)
    kw = {"input_names": ["image"], "output_names": ["logits"], "opset_version": 18}
    if dynamic_batch:
        kw["dynamic_axes"] = {"image": {0: "batch"}, "logits": {0: "batch"}}
    torch.onnx.export(model, (x,), str(path), dynamo=False, **kw)
    return path


def onnx_session(path, threads: int = THREADS, level: str = "all"):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = {"none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                                   "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                                   "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                                   "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL}[level]
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def onnx_logits(sess, x: np.ndarray, batch: int = 256) -> np.ndarray:
    out = [sess.run(None, {"image": to_input(x[i:i + batch]).numpy()})[0] for i in range(0, len(x), batch)]
    return np.concatenate(out)


def onnx_predict(sess, x: np.ndarray) -> np.ndarray:
    return onnx_logits(sess, x).argmax(1)

# ------------------------------------------------------------------ Hasad, the small language model
# A character-level decoder-only transformer trained by the course (_tools/train_models.py) on
# the synthetic advisory facts. One token is one character.


class Tokenizer:
    def __init__(self, chars: str):
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}

    def encode(self, s: str) -> list[int]:
        unk = self.stoi["?"]
        return [self.stoi.get(c, unk) for c in s]

    def decode(self, ids) -> str:
        return "".join(self.chars[i] for i in ids)


class QLinear(nn.Module):
    """A linear layer whose weight is stored in 8 or 4 bits, in groups of `group` values that
    share one 16-bit scale. It turns the weight back into float32 on every call."""

    def __init__(self, q: torch.Tensor, scale: torch.Tensor, bits: int, shape, group: int, bias=None):
        super().__init__()
        self.register_buffer("q", q)
        self.register_buffer("scale", scale)
        self.bits, self.shape, self.group = bits, tuple(shape), group
        self.bias = bias

    def weight(self):
        return dequantise(self.q, self.scale, self.bits, self.shape, self.group)

    def forward(self, x):
        return F.linear(x, self.weight(), self.bias)


def quantise(w: torch.Tensor, bits: int, group: int = 32):
    """Symmetric group-wise quantisation of a 2-D weight: (codes, fp16 scales).

    8 bits: codes are int8 in -127..127. 4 bits: codes in -7..7, two packed in each uint8."""
    out_f, in_f = w.shape
    g = w.float().reshape(out_f, in_f // group, group)
    qmax = 127 if bits == 8 else 7
    scale = (g.abs().amax(dim=2, keepdim=True) / qmax).clamp(min=1e-8)
    codes = torch.round(g / scale).clamp(-qmax, qmax).to(torch.int8)
    if bits == 4:
        u = (codes.reshape(-1) + 8).to(torch.uint8)              # 1..15
        codes = (u[0::2] | (u[1::2] << 4)).contiguous()
    return codes, scale.squeeze(2).half()


def dequantise(codes, scale, bits, shape, group=32):
    out_f, in_f = shape
    if bits == 4:
        lo = (codes & 0x0F).to(torch.int8) - 8
        hi = (codes >> 4).to(torch.int8) - 8
        codes = torch.stack([lo, hi], dim=1).reshape(-1)
    g = codes.reshape(out_f, in_f // group, group).float()
    return (g * scale.float().unsqueeze(2)).reshape(out_f, in_f)


class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.n_head = n_head

    def forward(self, x, cache=None):
        B, T, D = x.shape
        h = self.n_head
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (t.view(B, T, h, D // h).transpose(1, 2) for t in (q, k, v))
        if cache is not None:
            if cache["k"] is not None:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        causal = cache is None or T > 1
        a = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class Hasad(nn.Module):
    def __init__(self, vocab: int, d: int = 128, n_layer: int = 6, n_head: int = 4, ctx: int = 512):
        super().__init__()
        self.cfg = {"vocab": vocab, "d": d, "n_layer": n_layer, "n_head": n_head, "ctx": ctx}
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, n_head) for _ in range(n_layer))
        self.ln = nn.LayerNorm(d)
        self.tokenizer: Tokenizer | None = None
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, idx, caches=None, start: int = 0):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(start, start + T))
        for i, b in enumerate(self.blocks):
            x = b(x, None if caches is None else caches[i])
        return self.ln(x) @ self.tok.weight.T

    def new_cache(self):
        return [{"k": None, "v": None} for _ in self.blocks]

    @torch.no_grad()
    def generate(self, prompt: str, max_new: int = 80, use_cache: bool = True, stop: str = "\n") -> dict:
        """Greedy decoding. Returns text, time to first token (s), decode tokens per second,
        and the key-value cache's bytes at the end."""
        tk = self.tokenizer
        ids = tk.encode(prompt)[-(self.cfg["ctx"] - max_new):]
        t0 = time.perf_counter()
        caches = self.new_cache() if use_cache else None
        logits = self(torch.tensor([ids]), caches)
        nxt = int(logits[0, -1].argmax())
        ttft = time.perf_counter() - t0
        out = [nxt]
        t1 = time.perf_counter()
        while len(out) < max_new and tk.chars[nxt] != stop:
            if use_cache:
                logits = self(torch.tensor([[nxt]]), caches, start=len(ids) + len(out) - 1)
            else:
                seq = ids + out
                logits = self(torch.tensor([seq]))
            nxt = int(logits[0, -1].argmax())
            out.append(nxt)
        dt = time.perf_counter() - t1
        kv = sum(t.numel() * t.element_size() for c in caches for t in (c["k"], c["v"])) if use_cache else 0
        text = tk.decode(out).rstrip(stop)
        return {"text": text, "ttft_s": ttft, "decode_tok_s": (len(out) - 1) / dt if dt > 0 else float("nan"),
                "prompt_tokens": len(ids), "new_tokens": len(out), "kv_bytes": int(kv)}


def prompt_for(question: str, notes: str = "") -> str:
    """Hasad's prompt format: optional farm notes, then the question and the answer marker."""
    return (f"ملاحظات: {notes}\n" if notes else "") + f"س: {question}\nج: "


def farm_notes(n_chars: int, seed: int = 0) -> str:
    """Synthetic farm-log notes of about n_chars characters, to lengthen a prompt."""
    rng = np.random.default_rng([P.SEED, 7000, seed])
    words = ["ري", "الصباح", "تم", "فحص", "الأوراق", "سماد", "البيت", "المحمي", "رقم", "الحرارة",
             "مرتفعة", "اليوم", "عامل", "المضخة", "تعمل", "جيداً", "الحصاد", "قريب", "تنظيف", "القنوات"]
    s = ""
    while len(s) < n_chars:
        s += words[int(rng.integers(len(words)))] + " "
    return s[:n_chars]


def load_hasad(bits: int = 32, mode: str = "once") -> Hasad:
    """Load Hasad from data/models/hasad_<bits>.pt.

    bits 32: float32 weights. bits 8 or 4: the quantised file. mode "once" turns every weight
    back into float32 at load (small file, same memory and speed as float32); mode "per_call"
    keeps the codes in memory and expands each weight when it is used (small memory, slower)."""
    blob = torch.load(MODELS / f"hasad_{bits}.pt", map_location="cpu", weights_only=False)
    cfg, chars = blob["cfg"], blob["chars"]
    model = Hasad(**cfg)
    if bits == 32:
        model.load_state_dict(blob["state"])
    else:
        state = dict(blob["state"])
        for name, q in blob["quant"].items():
            state[name] = dequantise(q["codes"], q["scale"], bits, q["shape"], q["group"])
        model.load_state_dict({k: v.float() for k, v in state.items()})
        if mode == "per_call":
            for name, q in blob["quant"].items():
                mod_name = name.rsplit(".", 1)[0]
                parent_name, attr = mod_name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
                lin = getattr(parent, attr)
                setattr(parent, attr, QLinear(q["codes"], q["scale"], bits, q["shape"], q["group"], lin.bias))
    model.tokenizer = Tokenizer(chars)
    return model.eval()


def save_hasad_quantised(model: Hasad, bits: int, path: Path, group: int = 32) -> Path:
    """Write the quantised file: every Linear weight in `bits`, the rest in float16."""
    quant, state = {}, {}
    for name, t in model.state_dict().items():
        if name.endswith(".weight") and t.dim() == 2 and not name.startswith(("tok", "pos")):
            codes, scale = quantise(t, bits, group)
            quant[name] = {"codes": codes, "scale": scale, "shape": tuple(t.shape), "group": group}
        else:
            state[name] = t.half()
    torch.save({"cfg": model.cfg, "chars": model.tokenizer.chars, "state": state, "quant": quant}, path)
    return path


NUM = re.compile(r"\d+")


def grade(answer: str, key: str) -> bool:
    """An answer is right when the first number in it is the key fact."""
    m = NUM.search(answer)
    return bool(m) and m.group(0) == key

# ------------------------------------------------------------------ energy


def energy_per_1000(ms_per_prediction: float, watts: float) -> float:
    """Watt-hours per 1,000 predictions: an estimate from time and an assumed power draw."""
    return watts * ms_per_prediction / 1000 * 1000 / 3600

# ------------------------------------------------------------------ structured pruning


def conv_layers(model: Waraqa):
    """The (conv, batch norm) pairs of Waraqa, in order."""
    mods = list(model.features)
    return [(mods[i], mods[i + 1]) for i in range(len(mods)) if isinstance(mods[i], nn.Conv2d)]


def channel_scores(model: Waraqa) -> list[torch.Tensor]:
    """L1 norm of each output filter of each convolution: a cheap importance score."""
    return [c.weight.detach().abs().sum(dim=(1, 2, 3)) for c, _ in conv_layers(model)]


def prune_channels(model: Waraqa, keep: list[torch.Tensor]) -> Waraqa:
    """A smaller Waraqa that keeps, in each convolution, the output channels listed in keep[i].
    Weights are copied, so the result works before any fine-tuning."""
    widths = tuple(len(k) for k in keep)
    small = Waraqa(widths=widths)
    old, new = conv_layers(model), conv_layers(small)
    prev = torch.arange(3)
    for (c0, b0), (c1, b1), k in zip(old, new, keep):
        k = torch.as_tensor(k).long()
        c1.weight.data = c0.weight.data[k][:, prev].clone()
        for name in ("weight", "bias", "running_mean", "running_var"):
            getattr(b1, name).data = getattr(b0, name).data[k].clone()
        prev = k
    small.head.weight.data = model.head.weight.data[:, prev].clone()
    small.head.bias.data = model.head.bias.data.clone()
    return small.eval()
