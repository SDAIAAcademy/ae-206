<!-- academy:title -->
# Machine Learning Systems on One Machine

أنظمة تعلم الآلة على جهاز واحد

AI Engineering Track · SDAIA Academy · AE-206 · Level 200

Find where time and memory go on one machine, fix the largest cost first, and prove each speed-up kept its quality

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
| Capstone | Capstone: an optimisation study and its defence |  | [`slides/capstone.html`](slides/capstone.html) |
<!-- /academy:units -->

## Case study

**Nama Agritech** is a fictional farm-advisory company whose field agents run two models on old farm-office laptops: Waraqa, a leaf-disease classifier for ten disease classes across four crops, and Hasad, a small Arabic language model that answers farmer questions in Modern Standard Arabic, Gulf dialect and mixed Arabic and English. All data is synthetic and generated with a fixed seed by [`labs/prepare_data.py`](labs/prepare_data.py): 20,000 leaf images and 2,000 farmer questions, about one in seven of which carries personal details that the Data Classification Policy places at Restricted. Hasad has 1.27 million parameters and was trained by the course, so nothing is downloaded beyond the Python packages.

<!-- academy:labs -->
## Labs

Every lab is a notebook in `labs/`. Open it in Google Colab from the course page, or run it locally with `uv`. Exercises raise `NotImplementedError` until they are done; the reference versions are in `labs/solutions/`.

| | Session | Lab | File |
|---|---|---|---|
| 1.1 | Day 1 · Session 1 | Two layers on the roofline | [`D1_1_roofline.ipynb`](labs/D1_1_roofline.ipynb) |
| 1.2 | Day 1 · Session 2 | Noise in one inference | [`D1_2_timing_noise.ipynb`](labs/D1_2_timing_noise.ipynb) |
| 1.3 | Day 1 · Session 3 | Profile Lina's training step | [`D1_3_profile_step.ipynb`](labs/D1_3_profile_step.ipynb) |
| 1.4 | Day 1 · Session 4 | Fix the input pipeline | [`D1_4_input_pipeline.ipynb`](labs/D1_4_input_pipeline.ipynb) |
| 2.1 | Day 2 · Session 1 | Memory at three batch sizes | [`D2_1_memory_budget.ipynb`](labs/D2_1_memory_budget.ipynb) |
| 2.2 | Day 2 · Session 2 | Reduced precision | [`D2_2_mixed_precision.ipynb`](labs/D2_2_mixed_precision.ipynb) |
| 2.3 | Day 2 · Session 3 | A large batch in fixed memory | [`D2_3_accumulate_checkpoint.ipynb`](labs/D2_3_accumulate_checkpoint.ipynb) |
| 2.4 | Day 2 · Session 4 | Compile and fuse | [`D2_4_compile_fusion.ipynb`](labs/D2_4_compile_fusion.ipynb) |
| 3.1 | Day 3 · Session 1 | Batch size against latency | [`D3_1_batching.ipynb`](labs/D3_1_batching.ipynb) |
| 3.2 | Day 3 · Session 2 | Hasad on a laptop | [`D3_2_lm_inference.ipynb`](labs/D3_2_lm_inference.ipynb) |
| 3.3 | Day 3 · Session 3 | Quantise Waraqa and Hasad | [`D3_3_quantisation.ipynb`](labs/D3_3_quantisation.ipynb) |
| 3.4 | Day 3 · Session 4 | Prune or shrink | [`D3_4_pruning.ipynb`](labs/D3_4_pruning.ipynb) |
| 4.1 | Day 4 · Session 1 | Export and check parity | [`D4_1_export_parity.ipynb`](labs/D4_1_export_parity.ipynb) |
| 4.2 | Day 4 · Session 2 | A benchmark you can trust | [`D4_2_benchmark.ipynb`](labs/D4_2_benchmark.ipynb) |
| 4.3 | Day 4 · Session 3 | Quality by crop type | [`D4_3_quality_by_crop.ipynb`](labs/D4_3_quality_by_crop.ipynb) |
| 4.4 | Day 4 · Session 4 | Choose under the budget | [`D4_4_budget_choice.ipynb`](labs/D4_4_budget_choice.ipynb) |
<!-- /academy:labs -->

## Capstone

An optimisation study on a model of the participant's choice, submitted as a public repository whose README reports the baseline profile, each change ordered by measured impact, and the final latency, memory and quality against a stated budget with intervals and the hardware used, followed by an eight-minute defence and four minutes of questions

**Submission:** a link to a public GitHub repository built from [`labs/capstone/README_template.md`](labs/capstone/README_template.md), with profiler outputs, benchmark logs and the commit history. In the defence the participant re-runs the benchmark live, predicts the effect of a change the assessor proposes and then measures it.

| Criterion | Weight |
|---|---|
| Baseline profile | 20 |
| Changes ordered by measured impact | 25 |
| Quality re-evaluation | 20 |
| Benchmark method | 20 |
| Reproducibility | 15 |

## Assessment

Each session ends with a checkpoint in the deck. Practice questions for every outcome are in [`assessments/bank.json`](assessments/bank.json) and on the course page; exam items are not kept in this repository.

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

### Performance models and measurement

- [Williams, Waterman and Patterson: Roofline, an insightful visual performance model (UC Berkeley technical report, 2008)](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html)
- [Machine Learning Systems, the open textbook, with MLSys·im](https://mlsysbook.ai/)
- [PyTorch 2.15: torch.profiler](https://docs.pytorch.org/docs/2.15/profiler.html)
- [PyTorch recipe: benchmarking with torch.utils.benchmark](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)
- [py-spy, a sampling profiler for Python](https://github.com/benfred/py-spy)
- [Scalene, a CPU, GPU and memory profiler for Python](https://github.com/plasma-umass/scalene)
- [memray, a memory profiler for Python (Linux and macOS)](https://github.com/bloomberg/memray)
- [JAX documentation: profiling computation](https://docs.jax.dev/en/latest/profiling.html)
- [hyperfine, a command-line benchmarking tool](https://github.com/sharkdp/hyperfine)
- [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/)

### Training efficiently

- [PyTorch 2.15: datasets and data loaders (torch.utils.data)](https://docs.pytorch.org/docs/2.15/data.html)
- [PyTorch 2.15: automatic mixed precision (torch.amp)](https://docs.pytorch.org/docs/2.15/amp.html)
- [PyTorch 2.15: activation checkpointing (torch.utils.checkpoint)](https://docs.pytorch.org/docs/2.15/checkpoint.html)
- [PyTorch 2.15: torch.compile](https://docs.pytorch.org/docs/2.15/generated/torch.compile.html)
- [PyTorch 2.15: torch.backends](https://docs.pytorch.org/docs/2.15/backends.html)
- [Chen, Xu, Zhang and Guestrin: Training deep nets with sublinear memory cost (2016)](https://arxiv.org/abs/1604.06174)

### Inference, compression and export

- [Kwon et al.: Efficient memory management for large language model serving with PagedAttention (2023)](https://arxiv.org/abs/2309.06180)
- [Han, Mao and Dally: Deep compression (2015)](https://arxiv.org/abs/1510.00149)
- [Hinton, Vinyals and Dean: Distilling the knowledge in a neural network (2015)](https://arxiv.org/abs/1503.02531)
- [ONNX Runtime: quantising ONNX models](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [ONNX Runtime: graph optimisations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)
- [PyTorch 2.15: torch.onnx](https://docs.pytorch.org/docs/2.15/onnx.html)
- [ExecuTorch: on-device AI for PyTorch](https://github.com/pytorch/executorch)
- [LiteRT (Google AI Edge)](https://ai.google.dev/edge/litert)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [llama.cpp: quantize tool and quantisation types](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [Ollama](https://ollama.com/)

### National references

- [SDAIA: AI Ethics Principles](https://sdaia.gov.sa/en/SDAIA/about/Documents/ai-principles.pdf)
- [National Data Management Office (SDAIA): Data Classification Policy](https://sdaia.gov.sa/en/SDAIA/about/Documents/DataClassificationPolicy.pdf)
<!-- /academy:sources -->

### National references

- SDAIA AI Ethics Principles: https://sdaia.gov.sa/en/SDAIA/about/Documents/ai-principles.pdf (Fairness, used on Day 4 to report quality by crop; Social and Environmental Benefits, used for the energy estimate)
- NDMO Data Classification Policy: https://sdaia.gov.sa/en/SDAIA/about/Documents/DataClassificationPolicy.pdf (Nama adopts it as its own rule: farmer questions with names, phone or ID numbers are Restricted and never leave the laptop, Day 3)
