# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0", "matplotlib>=3.9", "psutil>=6.0", "torch>=2.6", "pillow>=11.0"]
# ///
# %% [markdown]
# # Lab 4.4 · Choose under the budget
#
# Day 4 · Session 4 (Budgets and trade-offs) · 25 min.
#
# Ghadeer Al-Shammasi set the budget for Waraqa on the farm-office laptops. The course measured
# every variant of the week on one laptop and supplies the table (`data/supplied/results_table.csv`),
# so this lab needs none of your earlier results. You filter by the budget and the quality
# floor, add an energy estimate, write the choice in one paragraph, and choose your capstone
# model and budget.
#
# **Tasks**
# - [ ] Read the supplied results table · 4 min
# - [ ] Filter by the budget and the quality floor · 5 min
# - [ ] Energy per 1,000 predictions · 5 min
# - [ ] The one-paragraph justification · 6 min
# - [ ] Capstone model and budget · 5 min
# - [ ] ✱ Stretch · Tighten the latency budget by a third: what is left?
#
# AI policy for this lab: AI-tutor for the table; the justification is written by you.

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

import csv
import json
import nama as U

BUDGET = json.loads((U.SUPPLIED / "budget.json").read_text(encoding="utf-8"))
print("Nama's budget for Waraqa on a farm-office laptop:")
for k, v in BUDGET.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## 1 · Read the supplied results table

# %%
with open(U.SUPPLIED / "results_table.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
NUM = ("file_mb", "median_ms", "p95_ms", "p95_ci_high", "peak_mb", "accuracy", "diff_overall", "worst_crop_diff")
for r in rows:
    for k in NUM:
        r[k] = float(r[k])
cols = ("variant", "runtime", "file_mb", "median_ms", "p95_ms", "peak_mb", "accuracy", "diff_overall", "worst_crop", "worst_crop_diff")
print(" | ".join(cols))
for r in rows:
    print(" | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols))
print("\nmeasured on:", rows[0]["machine"])

# %% [markdown]
# ## 2 · Filter by the budget and the quality floor
#
# ✏️ **Exercise** · Return the rows that meet every line of the budget: `p95_ms` at most
# `p95_ms_max`, `file_mb` at most `file_mb_max`, `peak_mb` at most `peak_mb_max`,
# `diff_overall` at least `-quality_floor_overall`, `worst_crop_diff` at least
# `-quality_floor_per_crop`.

# %%
def meets(rows, b):
    # BEGIN SOLUTION
    return [r for r in rows if r["p95_ms"] <= b["p95_ms_max"] and r["file_mb"] <= b["file_mb_max"]
            and r["peak_mb"] <= b["peak_mb_max"] and r["diff_overall"] >= -b["quality_floor_overall"]
            and r["worst_crop_diff"] >= -b["quality_floor_per_crop"]]
    # END SOLUTION

# %%
# Check, with a safety net: the cells below still run when the exercise is not done yet
def _meets(rows, b):
    return [r for r in rows if r["p95_ms"] <= b["p95_ms_max"] and r["file_mb"] <= b["file_mb_max"]
            and r["peak_mb"] <= b["peak_mb_max"] and r["diff_overall"] >= -b["quality_floor_overall"]
            and r["worst_crop_diff"] >= -b["quality_floor_per_crop"]]

try:
    ok = meets(rows, BUDGET)
except NotImplementedError:
    ok = _meets(rows, BUDGET)
for r in rows:
    why = [k for k, bad in (("latency", r["p95_ms"] > BUDGET["p95_ms_max"]), ("size", r["file_mb"] > BUDGET["file_mb_max"]),
                            ("memory", r["peak_mb"] > BUDGET["peak_mb_max"]),
                            ("quality", r["diff_overall"] < -BUDGET["quality_floor_overall"] or r["worst_crop_diff"] < -BUDGET["quality_floor_per_crop"])) if bad]
    print(f"{r['variant']:26s} {'meets the budget' if not why else 'fails: ' + ', '.join(why)}")
assert ok == _meets(rows, BUDGET)

# %% [markdown]
# ## 3 · Energy per 1,000 predictions
#
# No laptop in the room has a power meter, so this is an **estimate**: the time a prediction
# keeps the processor busy, times an assumed power draw. Use your laptop's processor power from
# its specification sheet if you know it; the course assumes 15 W for a thin laptop's processor
# under load. Write the assumption next to the number, always.
#
# ✏️ **Exercise** · Return the watt-hours of 1,000 predictions that take `ms` milliseconds each
# at `watts`.

# %%
WATTS = 15


def energy_wh(ms, watts=WATTS):
    # BEGIN SOLUTION
    return watts * (ms / 1000) * 1000 / 3600
    # END SOLUTION

# %%
try:
    energy_wh(1.0)
except NotImplementedError:
    energy_wh = U.energy_per_1000
for r in rows:
    r["wh_per_1000"] = energy_wh(r["median_ms"], WATTS)
    print(f"{r['variant']:26s} {r['wh_per_1000'] * 1000:7.2f} mWh per 1,000 leaves (at an assumed {WATTS} W)")

# %% [markdown]
# **Look at** the scale. The differences are real in proportion and tiny in absolute terms for
# one office; they matter across thousands of devices and for battery-powered use in the field,
# which is how SDAIA's AI Ethics Principles' attention to social and environmental benefits
# shows up in an engineering choice: as a number with its assumption.

# %% [markdown]
# ## 4 · The one-paragraph justification
#
# ✏️ **Exercise** · Choose the variant among `ok` with the highest accuracy (break a tie by
# the lower p95). Then rewrite the paragraph below from your own reading of the table: which
# variant, which budget lines it meets with how much margin, what it gave up, and the
# conditions of the measurement.

# %%
def choose(ok):
    # BEGIN SOLUTION
    return max(ok, key=lambda r: (r["accuracy"], -r["p95_ms"])) if ok else None
    # END SOLUTION

# %%
try:
    pick = choose(ok)
except NotImplementedError:
    pick = max(ok, key=lambda r: (r["accuracy"], -r["p95_ms"])) if ok else None
if pick is None:
    print("No variant meets the budget: report the nearest and the gap, line by line.")
else:
    paragraph = (
        f"We recommend {pick['variant']} for the farm-office laptops. On {pick['machine']}, it answers one leaf in "
        f"{pick['median_ms']:.2f} ms at the median and {pick['p95_ms']:.2f} ms at the 95th percentile "
        f"(budget {BUDGET['p95_ms_max']} ms), from a {pick['file_mb']:.2f} MB file (budget {BUDGET['file_mb_max']} MB) "
        f"with a peak of {pick['peak_mb']:.0f} MB (budget {BUDGET['peak_mb_max']} MB). Its accuracy on the held-out set "
        f"is {pick['accuracy']:.3f}, {pick['diff_overall'] * 100:+.1f} points against the 32-bit model, and its worst crop, "
        f"{pick['worst_crop']}, moves {pick['worst_crop_diff'] * 100:+.1f} points. Energy is about "
        f"{pick['wh_per_1000'] * 1000:.1f} mWh per 1,000 leaves at an assumed {WATTS} W.")
    print(paragraph)
    U.save_reference("d4_4_choice", {"pick": pick, "paragraph": paragraph, "meets": [r["variant"] for r in ok]})

# %% [markdown]
# ## 5 · Capstone model and budget
#
# Your capstone is an optimisation study on a model of your choice: Waraqa, Hasad, or a model
# of your own that runs on a laptop processor or a free notebook GPU. Fill in the plan; it is
# saved to `artifacts/capstone_plan.md` and is the first file of your capstone repository
# (the template is `capstone/README_template.md`).

# %%
PLAN = {
    "model": "Hasad (small language model), 32-bit file",
    "target device": "my laptop: " + U.machine()["cpu"],
    "latency budget": "time to first token at most 50 ms and at least 300 tokens per second at a 256-character context",
    "memory budget": "peak at most 200 MB above the Python process at rest",
    "quality floor": "right answers within 2 points of the 32-bit model overall and in every language variety",
    "first measurement": "baseline profile of prefill and decode with torch.profiler and timers",
}
text = "# Capstone plan\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in PLAN.items()) + "\n"
(U.ARTIFACTS / "capstone_plan.md").write_text(text, encoding="utf-8")
print(text)

# %% [markdown]
# ## ✱ Stretch · A tighter budget
#
# Cut `p95_ms_max` by a third and filter again. If nothing is left, which change from the week
# would you try next, and what would you measure first?

# %% [markdown]
# ## Check yourself
#
# 1. Variant A is fastest and fails the quality floor on date palm. Variant B meets everything
#    with a 0.1 ms margin. Which do you recommend, and what do you add?
# 2. Why is the energy figure an estimate, and what makes it honest?
# 3. In what order does an optimisation log list its changes?
#
# <details><summary>Answers</summary>
#
# 1. B, with the margin stated and a re-measurement on the target laptop: a 0.1 ms margin is
#    inside the noise of another machine.
# 2. Power was assumed, not measured; it is honest when the assumption and the method are
#    written next to it.
# 3. By measured impact, largest first, each with its before-and-after numbers and its quality
#    check.
# </details>
#
# ## Carry forward
#
# The choice paragraph is LO7's evidence on Waraqa; the capstone repeats the whole loop on the
# model you chose.
