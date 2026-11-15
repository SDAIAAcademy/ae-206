"""Train and write the supplied model files of AE-206 (run once by the course author).

    cd labs && uv run python _tools/train_models.py [waraqa|variants|hasad|all]

Writes data/models/: waraqa.pt (the classifier), waraqa_warm.pt (after one epoch, Lab 2.2),
waraqa_s.pt (half the channels, trained from scratch on the same budget), waraqa_pruned.pt
(half the channels pruned by L1 norm, then one epoch of fine-tuning), the ONNX file of each,
waraqa_int8.onnx (static 8-bit quantisation), and hasad_32.pt, hasad_8.pt, hasad_4.pt.
Seeds are fixed; times differ by machine.
"""
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F

import nama as U
import prepare_data as P

torch.set_num_threads(12)
OUT = U.MODELS
OUT.mkdir(parents=True, exist_ok=True)


def train_classifier(model, x, y, epochs, lr=2e-3, batch=128, seed=0, log=""):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(x) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    g = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        perm = g.permutation(len(x))
        t0 = time.time()
        for i in range(0, len(x) - batch + 1, batch):
            idx = perm[i:i + batch]
            xb = np.rot90(x[idx], k=int(g.integers(4)), axes=(1, 2))
            if g.random() < 0.5:
                xb = xb[:, :, ::-1]
            loss = F.cross_entropy(model(U.to_input(xb)), torch.from_numpy(y[idx]).long())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        print(f"{log} epoch {ep + 1}/{epochs} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
    return model.eval()


def waraqa():
    x, y, _ = U.leaves("train")
    xv, yv, _ = U.leaves("val")
    torch.manual_seed(1)
    warm = train_classifier(U.Waraqa(), x, y, 1, seed=1, log="warm")
    U.save_model(warm, OUT / "waraqa_warm.pt")
    print("warm val acc", U.accuracy(warm, xv, yv))
    torch.manual_seed(0)
    m = train_classifier(U.Waraqa(), x, y, 6, seed=0, log="waraqa")
    U.save_model(m, OUT / "waraqa.pt")
    print("waraqa val acc", U.accuracy(m, xv, yv))


def variants():
    x, y, _ = U.leaves("train")
    xv, yv, _ = U.leaves("val")
    torch.manual_seed(2)
    s = train_classifier(U.WaraqaS(), x, y, 6, seed=2, log="small")
    U.save_model(s, OUT / "waraqa_s.pt")
    print("small val acc", U.accuracy(s, xv, yv))
    base = U.load_model("waraqa")
    keep = [torch.sort(torch.topk(sc, len(sc) // 2).indices).values for sc in U.channel_scores(base)]
    pruned = U.prune_channels(base, keep)
    print("pruned before fine-tune", U.accuracy(pruned, xv, yv))
    pruned = train_classifier(pruned, x, y, 1, lr=1e-3, seed=3, log="prune-ft")
    U.save_model(pruned, OUT / "waraqa_pruned.pt")
    print("pruned val acc", U.accuracy(pruned, xv, yv))
    for name in ("waraqa", "waraqa_s", "waraqa_pruned"):
        U.export_onnx(U.load_model(name), OUT / f"{name}.onnx")
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quant_pre_process, quantize_static)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"image": U.to_input(x[i:i + 16]).numpy()} for i in range(0, 256, 16)])

        def get_next(self):
            return next(self.it, None)

    pre = OUT / "waraqa_pre.onnx"
    quant_pre_process(str(OUT / "waraqa.onnx"), str(pre))
    quantize_static(str(pre), str(OUT / "waraqa_int8.onnx"), Reader(), quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    pre.unlink()
    for f in sorted(OUT.glob("*.onnx")):
        print(f.name, round(U.file_mb(f), 3), "MB")


def hasad_corpus():
    items = []
    for f in P.facts():
        for tmpls in P.Q_TEMPLATES[f["topic"]].values():
            for t in tmpls:
                q = t.format(c=P.CROPS_AR[f["crop"]], lc="لل" + P.CROPS_AR[f["crop"]].removeprefix("ال"),
                             e=P.CROPS_EN[f["crop"]], r=P.REGIONS_AR.get(f["region"] or "", ""))
                items.append((q, f["answer"]))
    return items


def hasad(steps=3000):
    rng = np.random.default_rng(206)
    items = hasad_corpus()
    names = P.FIRST_NAMES
    text_all = "".join(q + a for q, a in items) + "".join(names) + U.farm_notes(4000, 1)
    text_all += "0123456789أنا، جوالي. مزرعة شمال الطريق السريع: رقم هويتي للتسجيل." + U.prompt_for("x", "y")
    chars = "".join(sorted(set(text_all + "?")))
    tk = U.Tokenizer(chars)
    torch.manual_seed(0)
    model = U.Hasad(len(chars))
    model.tokenizer = tk
    print("hasad params", U.n_params(model), "vocab", len(chars), "questions", len(items))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    ctx = model.cfg["ctx"]

    def digits(n):
        return "".join(str(int(d)) for d in rng.integers(0, 10, n))

    def sample(long):
        q, a = items[int(rng.integers(len(items)))]
        r = rng.random()
        name = names[int(rng.integers(len(names)))]
        if r < 0.1:
            q = f"أنا {name}، جوالي 05{digits(8)}. " + q
        elif r < 0.15:
            q = q + f" رقم هويتي 1{digits(9)} للتسجيل."
        elif r < 0.2:
            q = f"مزرعة {name} شمال الطريق السريع: " + q
        notes = U.farm_notes(int(rng.integers(20, ctx - 200)), int(rng.integers(10**6))) if long else ""
        return tk.encode(U.prompt_for(q, notes)), tk.encode(a + "\n")

    t0 = time.time()
    for step in range(steps):
        long = step % 4 == 0
        B = 8 if long else 32
        seqs = [sample(long) for _ in range(B)]
        T = max(len(p) + len(a) for p, a in seqs)
        X = torch.zeros(B, T, dtype=torch.long)
        Y = torch.full((B, T), -100, dtype=torch.long)
        for i, (p, a) in enumerate(seqs):
            s = p + a
            X[i, :len(s)] = torch.tensor(s)
            Y[i, len(p) - 1:len(s) - 1] = torch.tensor(s[len(p):])   # loss on the answer only
        logits = model(X)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), Y.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0:
            print(f"hasad step {step} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
    model.eval()
    torch.save({"cfg": model.cfg, "chars": chars, "state": model.state_dict()}, OUT / "hasad_32.pt")
    for bits in (8, 4):
        U.save_hasad_quantised(model, bits, OUT / f"hasad_{bits}.pt")
    qs = U.question_set(200)
    for bits in (32, 8, 4):
        m = U.load_hasad(bits)
        ok = sum(U.grade(m.generate(U.prompt_for(q["question"]))["text"], q["key"]) for q in qs)
        print(f"hasad_{bits}: {U.file_mb(OUT / f'hasad_{bits}.pt'):.2f} MB, {ok}/200 right", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("waraqa", "all"):
        waraqa()
    if what in ("variants", "all"):
        variants()
    if what in ("hasad", "all"):
        hasad(int(sys.argv[2]) if len(sys.argv) > 2 else 2000)
