"""Write data/reference/d2_training_rows.json, the training rows Lab 2.4 prints, from the
course's full runs of Labs 2.2 and 2.3 (their results in labs/artifacts/).

    cd labs && uv run python _tools/make_d2_rows.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nama as U

p22 = json.loads((U.ARTIFACTS / "d2_2_precision.json").read_text(encoding="utf-8"))
p23 = json.loads((U.ARTIFACTS / "d2_3_fit.json").read_text(encoding="utf-8"))
rows = []
for name, r in p22["results"].items():
    rows.append({"setting": name, "ms": r["ms_per_step"], "per_s": r["leaves_per_s"], "peak_mb": r["peak_mb"],
                 "accuracy": r["val_accuracy"], "unit": "64 leaves per step"})
for name, r in p23["results"].items():
    rows.append({"setting": name, "ms": r["s_per_step"] * 1000, "per_s": 256 / r["s_per_step"], "peak_mb": r["peak_mb"],
                 "accuracy": None, "unit": "256 leaves per step"})
machine = U.machine()
out = {"machine": f"{machine['cpu']}, {machine['threads_used']} threads", "rows": rows}
U.REF.mkdir(parents=True, exist_ok=True)
(U.REF / "d2_training_rows.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
