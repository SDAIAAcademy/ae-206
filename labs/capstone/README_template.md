# Optimisation study: (your model) on (your device)

Copy this file into your capstone repository as `README.md` and fill every section. Every
number states the machine and the conditions it was measured under. Commit the profiler
outputs, the benchmark logs (`bench.py` JSON reports or your own) and the scripts that produced
them; the commit history is part of the evidence.

## 1. The model, the device and the budget

| | |
|---|---|
| Model | (name, parameters, file size, where the weights come from and their licence) |
| Target device | (processor, cores and threads used, memory, operating system, power mode) |
| Latency budget | (for example: at most 15 ms at the 95th percentile per request at batch 1) |
| Memory budget | (for example: peak at most 300 MB above the process at rest) |
| Quality floor | (for example: within 1 point overall and 2 points in every group, on a named held-out set) |
| Groups checked | (for example: crop type, language variety) |

## 2. Baseline profile

Where time and memory go before any change: a table of phases (or operators) with their share,
the profiler you used, and the first conclusion. Link the raw output files.

## 3. Changes, ordered by measured impact

| # | Change | Median before → after (interval) | p95 before → after | Memory | Quality change (interval) | Kept? |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 2 | | | | | | |

One paragraph per change: what you expected, what you measured, why it differs.

## 4. Quality re-evaluation

The final model against the baseline on the held-out set: a paired comparison overall and per
group, with intervals, and the verdict against the floor.

## 5. Final benchmark

The benchmark method (warm-up, repetitions, input mix, batch, threads), the final numbers with
95% intervals, and the budget line by line: met, or the gap and its explanation. An energy
estimate per 1,000 predictions with its assumed power.

## 6. Reproduce

The exact commands, from a fresh clone, that rebuild the data, re-run the benchmark and
reproduce the table, and how long each takes.
