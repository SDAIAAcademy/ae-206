<!-- academy:title -->
# Machine Learning Systems on One Machine

أنظمة تعلم الآلة على جهاز واحد

AI Engineering Track · SDAIA Academy · AE-206 · Level 200

REPLACE: one line under the title: what participants can do after the course

Participants find out where time and memory go between raw data and a served prediction, and fix the largest cost first. For a fictional agritech company they profile a data pipeline and a training step, estimate memory from weights, activations, optimiser state and the key-value cache, apply mixed precision, batching, quantisation and pruning, export a model to a portable runtime, and benchmark latency, throughput and quality with repeated runs and intervals. Labs run on a laptop CPU, with an optional free notebook GPU, and every speed-up must be measured and checked for loss of quality.

**Format:** Intensive: 4 teaching days of four sessions and a capstone day, with guided pre-work and a tracked follow-up task

**Course page:** https://sdaiaacademy.github.io/ae-206/
<!-- /academy:title -->

## Outcomes

| | Outcomes | Level | Evidence |
|---|---|---|---|
| LO1 | Explain how compute, memory capacity and memory bandwidth limit training and inference, and identify which one binds for a described workload, given four workload descriptions with model size, batch size and hardware figures | Awareness | a worked estimate for each workload that matches the reference within a stated margin |
| LO2 | Measure where time goes in a data pipeline and a training step with profilers and timers, and remove the largest bottleneck, given a slow training script for the leaf-image model | Applied | a before-and-after profile and a measured speed-up with the change explained |
| LO3 | Compute the memory needed for weights, activations, optimiser state and the key-value cache and check each estimate against measured peak memory, given a small vision model and a small language model at three batch sizes and context lengths | Applied | an estimate table next to measured peaks, with each difference explained |
| LO4 | Apply mixed precision, batching and compilation and report their effect on speed, memory and accuracy, given the training and inference scripts and a fixed test set | Applied | a results table with time per step, throughput, peak memory and accuracy for each setting |
| LO5 | Apply post-training quantisation and pruning, export the model to a portable runtime and compare size, latency and quality with the original, given the trained classifier, a small language model and a held-out evaluation set | Applied | a comparison table of size, median and tail latency and quality loss with an interval |
| LO6 | Prepare a benchmark that controls warm-up, repetitions, batch size and input mix, and report percentiles with confidence intervals, given two model variants and a target laptop | Applied | a benchmark script and report that another participant re-runs within 10% of the reported median |
| LO7 | Optimise a model and its serving path to meet a stated latency and memory budget on a target device while keeping quality within an agreed tolerance, given a budget set by the company for its farm-office laptops and a quality floor on the held-out set | Performance | an optimisation log ordered by measured impact and a final benchmark that meets the budget or explains the gap |

<!-- academy:units -->
## Units

| Unit | Title | Sessions | Slides |
|---|---|---|---|
| Day 1 | Measure first | Compute, memory and bandwidth; Timers, profilers and noise; Profiling a training step; Fixing an input-bound pipeline | [`slides/day1.html`](slides/day1.html) |
| Day 2 | Memory and training efficiency | Where memory goes; Mixed precision; Accumulation and checkpointing; Compilation and fusion | [`slides/day2.html`](slides/day2.html) |
| Day 3 | Inference and compression | Latency, throughput and batching; Language model inference on a laptop; Quantisation; Pruning or a smaller model | [`slides/day3.html`](slides/day3.html) |
| Day 4 | Export and benchmark | Export to a portable runtime; A benchmark you can trust; Quality must survive; Budgets and trade-offs | [`slides/day4.html`](slides/day4.html) |
| Capstone | Capstone project presentations |  | [`slides/capstone.html`](slides/capstone.html) |
<!-- /academy:units -->

## Case study

**REPLACE: the name of the fictional organisation every lab works on.** REPLACE: two sentences on the case study and its synthetic data

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Day 1 · Session 1 | Estimate arithmetic intensity for two layers, predict which is memory-bound, then time both | [`D1_1_roofline.ipynb`](labs/D1_1_roofline.ipynb) |
| 1.2 | Day 1 · Session 2 | Time one inference 100 times, plot the distribution and show how warm-up and background load change it | [`D1_2_timing_noise.ipynb`](labs/D1_2_timing_noise.ipynb) |
| 1.3 | Day 1 · Session 3 | Profile one training step and rank data loading, forward pass, backward pass and optimiser time | [`D1_3_profile_step.ipynb`](labs/D1_3_profile_step.ipynb) |
| 1.4 | Day 1 · Session 4 | Add parallel loading, caching and a faster image format and measure the change in step time | [`D1_4_input_pipeline.ipynb`](labs/D1_4_input_pipeline.ipynb) |
| 2.1 | Day 2 · Session 1 | Estimate and measure peak memory for weights, activations and optimiser state at three batch sizes | [`D2_1_memory_budget.ipynb`](labs/D2_1_memory_budget.ipynb) |
| 2.2 | Day 2 · Session 2 | Train with and without reduced precision and compare time per step and validation accuracy | [`D2_2_mixed_precision.ipynb`](labs/D2_2_mixed_precision.ipynb) |
| 2.3 | Day 2 · Session 3 | Fit a larger effective batch in fixed memory with gradient accumulation and activation checkpointing | [`D2_3_accumulate_checkpoint.ipynb`](labs/D2_3_accumulate_checkpoint.ipynb) |
| 2.4 | Day 2 · Session 4 | Compile the model and measure the first-call cost and the steady-state speed-up | [`D2_4_compile_fusion.ipynb`](labs/D2_4_compile_fusion.ipynb) |
| 3.1 | Day 3 · Session 1 | Serve the classifier at batch sizes from 1 to 64 and plot latency against throughput | [`D3_1_batching.ipynb`](labs/D3_1_batching.ipynb) |
| 3.2 | Day 3 · Session 2 | Measure time to first token and tokens per second for a small model at three context lengths and size its key-value cache | [`D3_2_lm_inference.ipynb`](labs/D3_2_lm_inference.ipynb) |
| 3.3 | Day 3 · Session 3 | Quantise the classifier to 8-bit, load supplied 8-bit and 4-bit files of the language model, and measure size, speed and quality loss on a fixed question set | [`D3_3_quantisation.ipynb`](labs/D3_3_quantisation.ipynb) |
| 3.4 | Day 3 · Session 4 | Prune the classifier, fine-tune briefly and compare with a smaller architecture trained from scratch | [`D3_4_pruning.ipynb`](labs/D3_4_pruning.ipynb) |
| 4.1 | Day 4 · Session 1 | Export the classifier to an exchange format, run it in a portable runtime and check that outputs match within tolerance | [`D4_1_export_parity.ipynb`](labs/D4_1_export_parity.ipynb) |
| 4.2 | Day 4 · Session 2 | Write a benchmark with warm-up, repetitions and a fixed input mix and report percentiles with intervals | [`D4_2_benchmark.ipynb`](labs/D4_2_benchmark.ipynb) |
| 4.3 | Day 4 · Session 3 | Evaluate each compressed variant on the held-out set with a paired comparison against the original, by crop type | [`D4_3_quality_by_crop.ipynb`](labs/D4_3_quality_by_crop.ipynb) |
| 4.4 | Day 4 · Session 4 | From a supplied results table, choose the variant that meets the latency and memory budget, justify it in one paragraph and choose the capstone model | [`D4_4_budget_choice.ipynb`](labs/D4_4_budget_choice.ipynb) |
<!-- /academy:labs -->

## Capstone

An optimisation study on a model of the participant's choice, submitted as a public repository whose README reports the baseline profile, each change ordered by measured impact, and the final latency, memory and quality against a stated budget with intervals and the hardware used, followed by an eight-minute defence and four minutes of questions

**Submission:** REPLACE: what participants submit and how: a public repository link, or a shared folder for a course without code

## Prerequisites

- AE-202 Deep Learning and Generative AI Fundamentals
- Trains a small neural network with a hand-written training loop
- Reads a histogram and computes a percentile

<!-- academy:tools -->
## Tools

Each line is a capability; the examples are interchangeable.

- **Deep learning framework with a profiler:** PyTorch and its profiler, JAX and its profiler
- **Python profilers and memory tools:** py-spy, Scalene, memray
- **Portable inference runtime:** ONNX Runtime, ExecuTorch, LiteRT
- **Small language model runtime with quantisation:** llama.cpp, Ollama
- **Benchmark harness:** pytest-benchmark, hyperfine, a supplied timing script
- **Analytical performance model:** MLSys·im from the open Machine Learning Systems textbook, a spreadsheet roofline model
- **Statistics:** SciPy, NumPy
<!-- /academy:tools -->

<!-- academy:sources -->
## Sources

- REPLACE: every source the course cites, with its address, in course.json under "sources"; this list is written from there
<!-- /academy:sources -->

### National references

- SDAIA AI Ethics Principles: REPLACE: the address of the official text
- NDMO Data Classification Policy: REPLACE: the address of the official text
