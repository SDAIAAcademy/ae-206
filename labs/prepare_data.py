"""Synthetic data of Nama Agritech, the case study of AE-206.

Everything is computed from a fixed seed, so every machine gets the same images and questions:

- leaf images: 20,000 photographs of single leaves, ten disease classes, four crops. Image i is
  drawn from its own random stream (seed 206, i), so any subset can be rebuilt alone.
  Indices 0-15,999 are the training split, 16,000-17,999 validation, 18,000-19,999 the held-out
  evaluation set that Nama's quality floor is measured on.
- the field export: the same leaves as 224-pixel PNG files, the format the field app uploads
  (the input-bound pipeline of Labs 1.3 and 1.4 reads these).
- farmer questions: 2,000 synthetic Arabic questions to Hasad, in Modern Standard Arabic, Gulf
  dialect and mixed Arabic and English, with the expected answer, the key fact that grades it,
  and a data classification (some questions carry a phone or an identity number).

Run from labs/:  uv run python prepare_data.py            (what the labs need, in quick mode less)
                 uv run python prepare_data.py --export    (also the PNG field export)
No real person, farm or company is described. Names and numbers are invented.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

SEED = 206
N_IMAGES = 20_000
SPLITS = {"train": (0, 16_000), "val": (16_000, 18_000), "test": (18_000, 20_000)}
CLASSES = ["healthy", "leaf_spot", "rust", "powdery_mildew", "early_blight",
           "late_blight", "mosaic", "bacterial_streak", "chlorosis", "scorch"]
CROPS = ["tomato", "cucumber", "citrus", "date_palm"]
CROP_SHARE = [0.35, 0.30, 0.25, 0.10]
LABS = Path(__file__).resolve().parent
CACHE = LABS / "cache"

# ---------------------------------------------------------------- leaf images

# leaf shape per crop: half-length, half-width (in a [-1, 1] frame), base colour (RGB 0-1)
SHAPE = {
    "tomato": (0.62, 0.40, (0.22, 0.48, 0.16)),
    "cucumber": (0.62, 0.55, (0.32, 0.58, 0.20)),
    "citrus": (0.66, 0.30, (0.10, 0.36, 0.12)),
    "date_palm": (0.85, 0.13, (0.36, 0.48, 0.30)),
}


def label_of(i: int) -> tuple[int, int]:
    """The class and crop of image i (cheap: no drawing)."""
    rng = np.random.default_rng([SEED, i, 0])
    crop = int(rng.choice(4, p=CROP_SHARE))
    cls = 0 if rng.random() < 0.2 else int(rng.integers(1, 10))
    return cls, crop


def render(i: int, size: int = 128) -> np.ndarray:
    """Image i as uint8 (size, size, 3)."""
    cls, crop = label_of(i)
    rng = np.random.default_rng([SEED, i, 1])
    half_len, half_w, base = SHAPE[CROPS[crop]]
    s = rng.uniform(0.85, 1.1)
    half_len, half_w = half_len * s, half_w * s * rng.uniform(0.9, 1.1)
    ang = rng.uniform(0, np.pi)
    cx, cy = rng.uniform(-0.12, 0.12, 2)

    g = (np.arange(size) + 0.5) / size * 2 - 1
    X, Y = np.meshgrid(g, g)
    ca, sa = np.cos(ang), np.sin(ang)
    u = ((X - cx) * ca + (Y - cy) * sa) / half_len          # along the midrib, -1..1 inside
    v = (-(X - cx) * sa + (Y - cy) * ca) / half_w           # across the leaf
    width = np.sqrt(np.clip(1 - u ** 2, 0, 1)) * (1 - 0.25 * np.clip(u, 0, 1))   # pointed tip
    if crop == 0:  # serrated edge
        width = width * (1 + 0.06 * np.sin(u * 40))
    if crop == 1:  # lobes
        width = width * (1 + 0.12 * np.cos(u * 9))
    inside = np.abs(v) < width
    edge = np.clip((width - np.abs(v)) / (width + 1e-6), 0, 1)   # 0 at the margin, 1 on the midrib

    bg = np.array(rng.choice([(0.55, 0.45, 0.33), (0.75, 0.72, 0.66), (0.42, 0.36, 0.28)]))
    img = np.empty((size, size, 3))
    img[:] = bg + rng.normal(0, 0.03, 3)

    leaf = np.array(base) * rng.uniform(0.85, 1.15)
    shade = 0.85 + 0.25 * edge                               # darker towards the margin
    col = leaf[None, None, :] * shade[..., None]
    vein = np.exp(-(v / 0.04) ** 2) * (np.abs(u) < 0.95)
    col = col * (1 - 0.25 * vein[..., None]) + 0.25 * vein[..., None] * np.array([0.7, 0.8, 0.5])

    def blobs(n, r_lo, r_hi):
        pu = rng.uniform(-0.8, 0.8, n)
        pv = rng.uniform(-0.7, 0.7, n)
        r = rng.uniform(r_lo, r_hi, n)
        vn = v / max(width.max(), 1e-3)
        du = ((u[None] - pu[:, None, None]) * half_len) ** 2
        dv = ((vn[None] - pv[:, None, None]) * half_w) ** 2
        return (np.sqrt(du + dv) / r[:, None, None]).min(axis=0)

    def paint(mask, rgb, alpha=1.0):
        nonlocal col
        m = np.clip(mask, 0, 1)[..., None] * alpha
        col = col * (1 - m) + np.array(rgb)[None, None, :] * m

    # severity: early, faint cases are common; on the narrow date-palm leaflets more so
    sev = rng.uniform(0.25, 0.75) if crop == 3 else rng.uniform(0.3, 0.95)
    name = CLASSES[cls]
    # look-alikes on any leaf: specks of soil and natural ageing (yellowing) confuse spot and chlorosis
    if rng.random() < 0.4:
        d = blobs(int(rng.integers(2, 6)), 0.015, 0.03)
        paint(d < 1.0, (0.40, 0.30, 0.18), rng.uniform(0.3, 0.8))
    if rng.random() < 0.3:
        paint(np.clip(1 - edge, 0, 1) * 0.6, (0.70, 0.70, 0.30), rng.uniform(0.2, 0.6))
    if name == "leaf_spot":
        d = blobs(int(rng.integers(5, 10)), 0.035, 0.06)
        paint((d < 1.8) * 0.6, (0.85, 0.80, 0.25), sev)
        paint(d < 1.0, (0.30, 0.18, 0.08), sev)
    elif name == "rust":
        d = blobs(int(rng.integers(25, 45)), 0.012, 0.022)
        paint(d < 1.0, (0.85, 0.45, 0.10), sev)
    elif name == "powdery_mildew":
        d = blobs(int(rng.integers(4, 8)), 0.08, 0.16)
        paint(np.exp(-d ** 2) * 1.2, (0.92, 0.92, 0.88), 0.8 * sev)
    elif name == "early_blight":
        d = blobs(int(rng.integers(2, 4)), 0.09, 0.13)
        ring = 0.5 + 0.5 * np.cos(d * 12)
        paint((d < 1.0) * (0.55 + 0.45 * ring), (0.35, 0.22, 0.10), sev)
    elif name == "late_blight":
        side = rng.choice([-1, 1])
        patch = np.clip(1.6 - np.sqrt((u - rng.uniform(-0.4, 0.4)) ** 2 * 2 + (v / max(width.max(), 1e-3) - side * 0.6) ** 2) * 1.4, 0, 1)
        paint(patch * 2.0, (0.18, 0.15, 0.10), sev)
    elif name == "mosaic":
        f1, f2 = rng.uniform(6, 10, 2)
        p1, p2 = rng.uniform(0, 6.3, 2)
        mott = 0.5 + 0.5 * np.sin(u * f1 * 2 + p1) * np.sin(v * f2 + p2)
        paint(mott * 0.85, (0.75, 0.78, 0.25), sev)
    elif name == "bacterial_streak":
        for _ in range(int(rng.integers(3, 6))):
            vv = rng.uniform(-0.7, 0.7)
            line = np.exp(-((v / max(width.max(), 1e-3) - vv) / 0.035) ** 2) * (np.abs(u - rng.uniform(-0.3, 0.3)) < rng.uniform(0.3, 0.6))
            paint(line * 1.3, (0.20, 0.16, 0.08), sev)
    elif name == "chlorosis":
        between = 1 - np.exp(-(v / 0.10) ** 2)
        paint(between * 0.85, (0.78, 0.74, 0.30), sev)
    elif name == "scorch":
        margin = np.clip(1 - edge / rng.uniform(0.25, 0.4), 0, 1)
        paint(margin * 1.4, (0.45, 0.28, 0.12), sev)

    img = np.where(inside[..., None], col, img)
    light = rng.uniform(0.75, 1.2)
    gx = rng.uniform(-0.25, 0.25)
    img = img * (light + gx * X[..., None])
    img = img + rng.normal(0, 0.05, img.shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def render_small(i: int, size: int = 64) -> np.ndarray:
    """Image i at training size: drawn at twice the size, then box-averaged (anti-aliased)."""
    big = render(i, size * 2).astype(np.float32)
    return big.reshape(size, 2, size, 2, 3).mean(axis=(1, 3)).round().astype(np.uint8)


def leaf_arrays(split: str, n: int | None = None, size: int = 64):
    """(images uint8 N x size x size x 3, class, crop, index) of a split, cached in labs/cache."""
    lo, hi = SPLITS[split]
    n = hi - lo if n is None else min(n, hi - lo)
    path = CACHE / f"leaves_{split}_{n}_{size}.npz"
    if path.exists():
        z = np.load(path)
        return z["x"], z["y"], z["crop"], z["idx"]
    idx = np.arange(lo, lo + n)
    x = np.stack([render_small(int(i), size) for i in idx])
    lab = np.array([label_of(int(i)) for i in idx])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, x=x, y=lab[:, 0], crop=lab[:, 1], idx=idx)
    return x, lab[:, 0], lab[:, 1], idx


def field_export(n: int, size: int = 224) -> Path:
    """The field app's export: n training leaves as PNG files plus labels.csv."""
    from PIL import Image
    out = CACHE / f"field_export_{n}"
    done = out / "labels.csv"
    if done.exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    rows = ["file,cls,crop"]
    for i in range(n):
        Image.fromarray(render(i, size)).save(out / f"leaf_{i:05d}.png", compress_level=6)
        c, k = label_of(i)
        rows.append(f"leaf_{i:05d}.png,{c},{k}")
    done.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return out

# ---------------------------------------------------------------- farmer questions

CROPS_AR = {"tomato": "الطماطم", "cucumber": "الخيار", "lemon": "الليمون", "date_palm": "النخيل",
            "wheat": "القمح", "potato": "البطاطس", "pepper": "الفلفل", "alfalfa": "البرسيم"}
CROPS_EN = {"tomato": "tomato", "cucumber": "cucumber", "lemon": "lemon", "date_palm": "palm",
            "wheat": "wheat", "potato": "potato", "pepper": "pepper", "alfalfa": "alfalfa"}
REGIONS_AR = {"qassim": "القصيم", "ahsa": "الأحساء", "jazan": "جازان", "tabuk": "تبوك"}
VARIETIES = ["msa", "gulf", "mixed"]
VARIETY_SHARE = [0.5, 0.35, 0.15]


def facts() -> list[dict]:
    """The advisory facts Hasad was trained on (synthetic agronomy, not advice)."""
    rng = np.random.default_rng([SEED, 9000])
    out = []
    for crop in CROPS_AR:
        for region in REGIONS_AR:
            days = int(rng.integers(2, 9))
            out.append({"topic": "irrigation", "crop": crop, "region": region, "value": days,
                        "answer": f"اسقِ {CROPS_AR[crop]} في {REGIONS_AR[region]} كل {days} أيام في الصيف."})
        kg = int(rng.integers(4, 31)) * 5
        out.append({"topic": "fertiliser", "crop": crop, "region": None, "value": kg,
                    "answer": f"سمّد {CROPS_AR[crop]} بنحو {kg} كغ نيتروجين للهكتار."})
        dd = int(rng.integers(6, 25)) * 5
        out.append({"topic": "harvest", "crop": crop, "region": None, "value": dd,
                    "answer": f"يُحصد {CROPS_AR[crop]} بعد نحو {dd} يوماً من الزراعة."})
        t = int(rng.integers(16, 33))
        out.append({"topic": "temperature", "crop": crop, "region": None, "value": t,
                    "answer": f"أنسب حرارة لنمو {CROPS_AR[crop]} حوالي {t} درجة."})
    return out


Q_TEMPLATES = {
    "irrigation": {
        "msa": ["كم مرة أسقي {c} في {r} في الصيف؟", "ما الفترة المناسبة بين ريّ {c} في منطقة {r} صيفاً؟",
                "متى أروي {c} في {r} خلال الصيف؟"],
        "gulf": ["كل كم يوم أسقي {c} في {r} بالصيف؟", "{c} عندي في {r}، كم يوم بين السقية والثانية بالصيف؟",
                 "ابي اعرف سقي {c} في {r} وقت الصيف كل كم؟"],
        "mixed": ["irrigation {lc} في {r} بالصيف كل كم يوم؟", "كم يوم بين كل irrigation لـ{e} في {r} صيفاً؟"],
    },
    "fertiliser": {
        "msa": ["كم كمية النيتروجين المناسبة {lc}؟", "ما مقدار السماد النيتروجيني {lc} للهكتار؟"],
        "gulf": ["كم أحط سماد نيتروجين {lc}؟", "{c} وش يبي من السماد النيتروجيني؟"],
        "mixed": ["كم nitrogen يحتاج {c} للهكتار؟", "fertiliser لـ{e}، كم كغ نيتروجين؟"],
    },
    "harvest": {
        "msa": ["متى يُحصد {c} بعد الزراعة؟", "كم يوماً يحتاج {c} حتى الحصاد؟"],
        "gulf": ["{c} متى يصير جاهز للحصاد؟", "بعد كم يوم أحصد {c}؟"],
        "mixed": ["متى harvest {lc} بعد الزراعة؟", "كم يوم لين harvest الـ{e}؟"],
    },
    "temperature": {
        "msa": ["ما درجة الحرارة المناسبة لنمو {c}؟", "ما الحرارة المثلى لزراعة {c}؟"],
        "gulf": ["{c} وش أحسن حرارة له؟", "كم الحرارة اللي تناسب {c}؟"],
        "mixed": ["أفضل temperature {lc} كم؟", "ideal temperature لـ{e} كم درجة؟"],
    },
}
FIRST_NAMES = ["سالم", "مريم", "عبدالله", "هيا", "فهد", "منى", "خالد", "نوال", "سعد", "أمل"]


def farmer_questions() -> list[dict]:
    """2,000 questions: id, question, answer, key fact, variety, classification, contains."""
    path = CACHE / "farmer_questions.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rng = np.random.default_rng([SEED, 9001])
    F = facts()
    out = []
    for qid in range(2000):
        f = F[int(rng.integers(len(F)))]
        var = VARIETIES[int(rng.choice(3, p=VARIETY_SHARE))]
        tmpl = Q_TEMPLATES[f["topic"]][var]
        t = tmpl[int(rng.integers(len(tmpl)))]
        q = t.format(c=CROPS_AR[f["crop"]], lc="لل" + CROPS_AR[f["crop"]].removeprefix("ال"),
                     e=CROPS_EN[f["crop"]], r=REGIONS_AR.get(f["region"] or "", ""))
        contains = []
        roll = rng.random()
        if roll < 0.08:   # a phone number in the question: personal data
            phone = "05" + "".join(str(int(d)) for d in rng.integers(0, 10, 8))
            q = f"أنا {FIRST_NAMES[int(rng.integers(10))]}، جوالي {phone}. " + q
            contains = ["name", "phone"]
        elif roll < 0.12:  # an identity number: personal data
            nid = str(int(rng.choice([1, 2]))) + "".join(str(int(d)) for d in rng.integers(0, 10, 9))
            q = q + f" رقم هويتي {nid} للتسجيل."
            contains = ["id_number"]
        elif roll < 0.15:  # a named farm location: personal data with the name
            q = f"مزرعة {FIRST_NAMES[int(rng.integers(10))]} شمال الطريق السريع: " + q
            contains = ["name", "location"]
        out.append({"id": qid, "question": q, "answer": f["answer"], "key": str(f["value"]),
                    "topic": f["topic"], "crop": f["crop"], "region": f["region"], "variety": var,
                    "template": t, "classification": "Restricted" if contains else "Public",
                    "contains": contains})
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    return out


def main(argv):
    quick = os.environ.get("NAMA_QUICK", "0") == "1"
    n_test = 600 if quick else 2000
    leaf_arrays("test", n_test)
    farmer_questions()
    if "--export" in argv:
        field_export(768 if quick else 4096)
    print(f"data ready in {CACHE} (quick={quick})")


if __name__ == "__main__":
    main(sys.argv[1:])
