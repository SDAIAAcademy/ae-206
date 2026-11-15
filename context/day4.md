---
course: "Machine Learning Systems on One Machine"
code: "AE-206"
version: "2027.1.0"
deck: "day4"
day: 4
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-206/slides/day4.html"
---

# Machine Learning Systems on One Machine

## Day 4: Export and benchmark

A portable file, an honest benchmark, quality that survives, a choice

#### [day4#title] Export and benchmark

*SDAIA Academy · Machine Learning Systems on One Machine*

Day 4: A portable file, an honest benchmark, quality that survives, a choice

AI Engineering Track · 4 teaching days + capstone day · 4 sessions · 4 labs · AE-206

#### [day4#agenda] Export and benchmark

*Today*

- **Session 1 · Export to a portable runtime**: Exchange formats, graph optimisation and runtimes for processors, edge devices and small language models
- **Session 2 · A benchmark you can trust**: Benchmark design: warm-up, repetitions, a fixed input mix, intervals and a re-run
- **Session 3 · Quality must survive**: Paired comparison of a compressed model with the original, overall and by group
- **Session 4 · Budgets and trade-offs**: Budgets, ordering changes by measured impact, energy per prediction and the capstone brief

#### [day4#review-day-3] Why was the 4-bit file not faster?

*Review of Day 3*

**Question:** Hasad's 4-bit file is about a sixth of the 32-bit file's size. Loaded with "expand once", it decodes no faster than the 32-bit model once the machine is quiet. Why?

- A. The 4-bit file holds the same bytes as the 32-bit one
- B. At load the weights became float32 again, so each decode step reads as many bytes as before
- C. The key-value cache got larger in 4 bits
- D. The tokenizer is slower for quantised models

### Session 1: Export to a portable runtime

#### [day4#s1] Export to a portable runtime

*Session 1*

- Export a model to an exchange format with a dynamic batch axis
- Check runtime outputs against the original within a tolerance
- Compare graph optimisation levels of a runtime

#### [day4#why-export] An exported file carries the graph and the weights, without the training stack

*Exchange formats*

- **Smaller install.** A runtime instead of a training framework on every farm laptop.
- **Optimised once.** Fusions and constant folding happen when the session opens, not at every start-up compile.
- **One file, many devices.** The same file on Windows, Linux and phones, with the runtime for each.

#### [day4#runtimes-map] Choose the runtime by the device and the model family

*Runtimes by target*

| Target | Examples | What they add |
| --- | --- | --- |
| Processors in laptops and servers | ONNX Runtime | graph optimisation, 8-bit kernels, many execution providers |
| Phones and embedded devices | ExecuTorch, LiteRT | small binaries, mobile accelerators |
| Small language models on a laptop | llama.cpp, Ollama | quantised GGUF files, kernels on quantised blocks, a key-value cache |

#### [day4#parity-two-halves] Parity has two halves: the model in the file and the preprocessing around it

*Lab 4.1: Lina's export "gives different answers"*

- symptom: same class on 6 of 64 leaves
- test 1: give the runtime PyTorch's own input tensor: difference 4.8e-06
- cause: her runtime divides by 255 and never subtracts the mean 0.5 or divides by 0.25

> **The robust fix.** Export the preprocessing inside the file, so the runtime takes raw pixels and nobody can forget a step.

#### [day4#export-dynamic] Export in evaluation mode, with a dynamic batch axis

*The second fault: a fixed batch*

```python
# file: Lab 4.1
model.eval()
torch.onnx.export(model, (torch.randn(2, 3, 64, 64),), "waraqa.onnx",
                  input_names=["image"], output_names=["logits"],
                  opset_version=18, dynamo=False,
                  dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
```

Lina's file was traced with one leaf: batch 8 fails at run time. The night archive needs batches (Day 3).

#### [day4#parity-tolerance] State the tolerance, then prove it on many inputs

*Measured in Lab 4.1 on 500 held-out leaves*

- **4.8e-06** largest absolute logit difference
- **1e-4** the tolerance stated before the test
- **100%** leaves with the same class

#### [day4#graph-levels] The runtime's graph optimisations do Day 2's fusion for you

*Measured in Lab 4.1: batch 1, four threads*

| Setting | Median ms | p95 ms |
| --- | --- | --- |
| No graph optimisation | 10.5 | 22.2 |
| Basic | 9.5 | 17.7 |
| Extended | 8.2 | 17.8 |
| All | 4.5 | 8.7 |
| PyTorch eager, for comparison | 12.3 | 30.0 |

#### [day4#checkpoint-1] Where do you look first?

*Checkpoint*

**Question:** The exported Waraqa gives different classes from PyTorch on field photos. Feeding both the identical input tensor, their logits agree within 1e-6. Where is the fault?

- A. In the exported file: re-export with another opset
- B. In the preprocessing: the two pipelines build different tensors from the same photo
- C. In floating-point rounding inside the runtime
- D. In the model's training, which must be redone

#### [day4#lab-4-1] Export and check parity

Lab 4.1 · 25 min

Export Waraqa to ONNX, find why the supplied export does not match, run it in ONNX Runtime and check that outputs agree within tolerance.

[D4_1_export_parity.ipynb](https://sdaiaacademy.github.io/ae-206/labs/D4_1_export_parity.ipynb)

- [ ] Run the supplied export (4 min)
- [ ] Find why the outputs differ (6 min)
- [ ] Export with a dynamic batch axis (4 min)
- [ ] Parity within tolerance on 500 leaves (6 min)
- [ ] Graph optimisation levels (5 min)
- [ ] ✱ Stretch: Count the nodes of the graph before and after optimisation

### Session 2: A benchmark you can trust

#### [day4#s2] A benchmark you can trust

*Session 2*

- Write a benchmark that controls warm-up, repetitions and input mix
- Report percentiles with bootstrap confidence intervals
- Re-run a benchmark and judge the agreement

#### [day4#benchmark-design] Control what you can, record what you cannot

*Benchmark design*

- **Controlled.** Warm-up calls, number of repetitions, a fixed input mix in a file, batch size, threads, the model file.
- **Recorded.** Processor, memory, operating system, runtime version, power mode, the load at start.

> **The test of a benchmark.** Another person runs it on the same kind of machine and gets a median within 10%.

#### [day4#input-mix] Time 40 leaves in the field's crop shares, saved to a file

*A fixed input mix*

| Item | Value |
| --- | --- |
| Tomato | 14 |
| Cucumber | 12 |
| Citrus | 10 |
| Date palm | 4 |

- One leaf measures one leaf; real traffic is mixed.
- The file (`mix.npy`) makes a re-run see the same inputs.
- For a language model the mix is prompts of the field's lengths.

#### [day4#bootstrap-interval] A bootstrap interval says how far the median could move on a re-run

*Percentiles with intervals*

```python
# file: Lab 4.2
rng = np.random.default_rng(206)
boots = np.percentile(
    ms[rng.integers(0, n, (2000, n))], q, axis=1)
low, high = np.percentile(boots, [2.5, 97.5])
```

- Resample the timings with replacement, recompute the percentile, 2,000 times.
- The middle 95% of those values is the interval.
- The 95th percentile's interval is wider: it rests on fewer calls.

#### [day4#benchmark-report] The 8-bit file is about twice as fast at the median, and the two intervals do not overlap

*Measured in Lab 4.2: batch 1, 300 calls after 30 warm-up, four threads*

| Variant | Median ms [95% interval] | p95 ms [95% interval] | MB |
| --- | --- | --- | --- |
| Waraqa 32-bit | 4.18 [4.08, 4.37] | 7.99 [7.60, 9.02] | 1.16 |
| Waraqa 8-bit | 2.17 [2.06, 2.30] | 5.15 [4.25, 5.76] | 0.31 |

Re-run on the same laptop: medians moved +7.5% (32-bit) and +8.6% (8-bit), both within 10%.

#### [day4#rerun-outside] Find the difference in conditions before reporting either number

*When a re-run disagrees*

- **The machine.** Load at start, power mode, battery, temperature, another processor.
- **The settings.** Threads, batch, runtime version, graph optimisation level.
- **The inputs.** The same mix file? The same model file, byte for byte?

#### [day4#checkpoint-2] What do you report?

*Checkpoint*

**Question:** The 8-bit median is 3% lower than the 32-bit median, and the two 95% intervals overlap by most of their width. What is the honest report?

- A. On this machine at batch 1 the two are not distinguishable; the 8-bit file is smaller
- B. The 8-bit model is 3% faster
- C. The two models are exactly as fast
- D. Repeat until the intervals separate, then report the difference

#### [day4#lab-4-2] A benchmark you can trust

Lab 4.2 · 25 min

Write a benchmark of two Waraqa variants with warm-up, repetitions and a fixed input mix, report percentiles with intervals, and re-run it.

[D4_2_benchmark.ipynb](https://sdaiaacademy.github.io/ae-206/labs/D4_2_benchmark.ipynb)

- [ ] Warm-up and repetitions (5 min)
- [ ] A fixed input mix (4 min)
- [ ] Percentiles with bootstrap intervals (6 min)
- [ ] Both variants in one report (5 min)
- [ ] Re-run and compare within 10% (5 min)
- [ ] ✱ Stretch: Time the benchmark script from the command line with hyperfine

### Session 3: Quality must survive

#### [day4#s3] Quality must survive

*Session 3*

- Compare two models on the same items with a paired test
- Put an interval on an accuracy difference
- Accept or reject a variant on the quality floor by crop type

#### [day4#quality-floor] At most 1 point lost overall and 2 points on any crop

*Nama's quality floor*

- Compression changes errors **unevenly**: faint lesions, narrow leaflets, the rarest crop.
- So every variant is checked overall and per crop, and Hasad per language variety (Lab 3.3).
- Fairness, the first of SDAIA's AI Ethics Principles, shows up here as a per-group check.

- **The farmers behind a group.** Date palm is one leaf in ten in the data and the main crop of many farms in some regions.

#### [day4#paired-comparison] Compare two models on the same leaves; only the disagreements carry information

*A paired comparison*

|  | Variant right | Variant wrong |
| --- | --- | --- |
| Original right | both right | **b**: lost |
| Original wrong | **c**: gained | both wrong |

Difference in accuracy = (c - b) / n. McNemar's exact test asks whether b and c could come from a fair coin.

#### [day4#quality-by-crop] Only the 8-bit file survives; pruning loses 23 points on date palm, the smallest crop

*Measured in Lab 4.3 on the 2,000 held-out leaves*

| Variant | Overall [95%] | Tomato | Cucumber | Citrus | Date palm | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 8-bit | +0.0 [-0.1, +0.3] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -0.2 [-0.6, +0.0] | +1.0 [+0.0, +2.6] | accept |
| Pruned | -4.1 [-5.0, -3.2] | -1.4 [-2.4, -0.6] | -1.0 [-1.8, -0.3] | -4.0 [-5.7, -2.4] | -23.3 [-30.1, -17.1] | reject |
| Waraqa-S | -1.2 [-1.7, -0.6] | -0.4 [-1.0, +0.0] | -0.2 [-0.5, +0.0] | +0.0 [+0.0, +0.0] | -9.8 [-15.0, -5.2] | reject |

#### [day4#uncertain-verdict] An uncertain verdict is a request for evidence, not a pass

*Reading an "uncertain"*

- About 200 date-palm leaves: one leaf is half a point.
- The interval says how many more leaves would settle it.
- Ask Nama for more labelled date-palm leaves before shipping, or keep the larger model for that crop.

- **LO5 evidence.** A comparison table of size, median and tail latency, and quality loss with an interval.

#### [day4#checkpoint-3] Accept, reject or uncertain?

*Checkpoint*

**Question:** A variant loses 0.4 points overall, interval [-1.3, +0.5], and 1.0 point on date palm, interval [-4.0, +2.0]. The floor is 1 point overall and 2 per crop. What is the verdict?

- A. Accept: both losses are within the floor
- B. Reject: the intervals go below the floor
- C. Uncertain: the estimates pass but both intervals reach below the floor
- D. Accept overall; per-crop intervals do not matter

#### [day4#lab-4-3] Quality by crop type

Lab 4.3 · 25 min

Evaluate each compressed Waraqa on the held-out set against the original with a paired comparison, overall and by crop type.

[D4_3_quality_by_crop.ipynb](https://sdaiaacademy.github.io/ae-206/labs/D4_3_quality_by_crop.ipynb)

- [ ] Predictions of each variant (4 min)
- [ ] Paired differences overall (6 min)
- [ ] An interval on each difference (5 min)
- [ ] By crop type (6 min)
- [ ] Accept or reject each variant (4 min)
- [ ] ✱ Stretch: Which leaves flip from right to wrong? Look at twelve of them

### Session 4: Budgets and trade-offs

#### [day4#s4] Budgets and trade-offs

*Session 4*

- Choose the variant that meets a latency and memory budget
- Estimate energy per 1,000 predictions
- Order optimisation changes by measured impact

#### [day4#nama-budget] Five lines, each measured, each with its margin

*Ghadeer's budget for Waraqa on a farm-office laptop*

| Line | Limit |
| --- | --- |
| Latency | at most 15 ms at the 95th percentile, batch 1, four threads |
| Model file | at most 1 MB |
| Memory | at most 150 MB peak for the whole serving process |
| Quality overall | within 1 point of the 32-bit model on the held-out set |
| Quality per crop | within 2 points of the 32-bit model on every crop |

#### [day4#results-table] Only the 8-bit file meets every line; the faster pruned model fails on date palm

*Supplied to Lab 4.4: every variant on one laptop*

| Variant | Runtime | MB | p95 ms | Peak MB | Accuracy | Worst crop | Budget |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Waraqa 32-bit | PyTorch eager | 1.17 | 12.0 | 227 | 99.1% | none | fails: file, memory |
| Waraqa 32-bit | ONNX Runtime | 1.16 | 2.4 | 68 | 99.1% | none | fails: file |
| Waraqa 8-bit | ONNX Runtime | 0.31 | 1.6 | 66 | 99.1% | citrus, -0.2 points | meets every line |
| Waraqa pruned | ONNX Runtime | 0.29 | 1.1 | 65 | 95.0% | date palm, -23.3 points | fails: quality, date palm |
| Waraqa-S | ONNX Runtime | 0.29 | 1.2 | 65 | 97.9% | date palm, -9.8 points | fails: quality, date palm |

#### [day4#energy-estimate] Time per prediction times an assumed power, with the assumption written next to it

*Energy per prediction, as an estimate*

- **Wh per 1,000 = watts x seconds per prediction x 1,000 / 3,600.** The course assumes 15 W for a thin laptop's processor under load; use your laptop's specification if you know it.

- No power meter in the room: this is an estimate, and says so.
- Proportions between variants are meaningful; absolute numbers are rough.
- Social and environmental benefit, one of SDAIA's AI Ethics Principles, enters the trade-off as a number.

#### [day4#order-by-impact] List changes by measured impact, largest first, each with its quality check

*The optimisation log*

| # | Change | Measured effect | Quality |
| --- | --- | --- | --- |
| 1 | The change that moved the binding limit most | before and after, with intervals | paired, per group |
| 2 | The next one, measured after change 1 | before and after | paired, per group |
| … | Changes that did nothing stay in the log | "no measurable effect" |  |

#### [day4#capstone-brief] Your optimisation study: a model, a device, a budget, a repository

*Capstone kickoff*

- A model of your choice: Waraqa, Hasad, or your own that runs on a laptop processor or a free notebook GPU.
- A public repository whose README follows `labs/capstone/README_template.md`.
- Baseline profile, changes ordered by measured impact, final latency, memory and quality against your budget, with intervals and the hardware.

- **The defence.** Eight minutes and four of questions. You re-run your benchmark live, predict the effect of a change the assessor proposes, then measure it.

#### [day4#checkpoint-4] Which variant do you recommend?

*Checkpoint*

**Question:** Variant A is the fastest but fails the quality floor on date palm. Variant B meets every line, with a p95 of 14.9 ms against a 15 ms budget, measured on the course laptop. What do you recommend?

- A. A, because speed matters most in the field
- B. B, with no further condition
- C. Neither: keep the 32-bit model until something is faster
- D. B, with its margin stated and a re-run on a farm-office laptop before it ships

#### [day4#lab-4-4] Choose under the budget

Lab 4.4 · 25 min

From a supplied results table, choose the variant that meets the farm-office budget, justify it in one paragraph, and choose the capstone model.

[D4_4_budget_choice.ipynb](https://sdaiaacademy.github.io/ae-206/labs/D4_4_budget_choice.ipynb)

- [ ] Read the supplied results table (4 min)
- [ ] Filter by the budget and the quality floor (5 min)
- [ ] Energy per 1,000 predictions (5 min)
- [ ] The one-paragraph justification (6 min)
- [ ] Capstone model and budget (5 min)
- [ ] ✱ Stretch: Tighten the latency budget by a third: what is left?

#### [day4#day-4-recap] Three things to keep from the course

*Day 4 recap*

- **Measure first.** Profile, find the binding limit, fix the largest cost, measure again.
- **Honest numbers.** Warm-up, repetitions, fixed inputs, intervals, the machine, a re-run within 10%.
- **Quality must survive.** Every speed-up is checked, paired and per group, against a floor set in advance.

#### [day4#tomorrow-capstone] Capstone presentations

*Tomorrow · Day 5*

Bring your repository, your benchmark ready to re-run on your laptop, and a recorded backup of the run. Eight minutes, then four of questions; the assessor will propose one change for you to predict and measure.
