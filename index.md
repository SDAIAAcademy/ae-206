---
course: "Machine Learning Systems on One Machine"
code: "AE-206"
version: "2027.1.0"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-206/index.html"
---

# Machine Learning Systems on One Machine

أنظمة تعلم الآلة على جهاز واحد

AI Engineering Track · 4 teaching days + capstone day · 26 hours · Level 200

Find where time and memory go on one machine, fix the largest cost first, and prove each speed-up kept its quality

Participants find out where time and memory go between raw data and a served prediction, and fix the largest cost first. For a fictional agritech company they profile a data pipeline and a training step, estimate memory from weights, activations, optimiser state and the key-value cache, apply mixed precision, batching, quantisation and pruning, export a model to a portable runtime, and benchmark latency, throughput and quality with repeated runs and intervals. Labs run on a laptop CPU, with an optional free notebook GPU, and every speed-up must be measured and checked for loss of quality.

**Who it is for:** Practitioners. Graduates and early-career developers who have completed the FN courses or passed the placement test, and software engineers moving into AI work.

**Format:** Intensive: 4 teaching days of four sessions and a capstone day, with guided pre-work and a tracked follow-up task

## The case study

**Nama Agritech.** Nama Agritech is a fictional farm-advisory company whose agents run two models on old farm-office laptops: Waraqa, a leaf-disease classifier trained on 20,000 synthetic leaf images in ten disease classes and four crops, and Hasad, a small language model the course trained to answer 2,000 synthetic Arabic farmer questions in Modern Standard Arabic, Gulf dialect and mixed Arabic and English. Participants measure, profile, estimate memory, compress, export and benchmark both against the budget Nama set for its laptops; all data is synthetic and generated with a fixed seed.

## The course

### Day 1: Measure first

Slides: https://sdaiaacademy.github.io/ae-206/slides/day1.md

- **Session 1: Compute, memory and bandwidth**
  - Estimate the FLOPs and bytes moved of a layer
  - Place a layer on a roofline from its arithmetic intensity
  - Name the binding limit of a described workload
  - [Lab 1.1: Two layers on the roofline (25 min)](https://sdaiaacademy.github.io/ae-206/labs/D1_1_roofline.ipynb)
- **Session 2: Timers, profilers and noise**
  - Time a call correctly with a monotonic clock and warm-up
  - Report a distribution with its median and tail percentiles
  - Show how background load changes the tail
  - [Lab 1.2: Noise in one inference (25 min)](https://sdaiaacademy.github.io/ae-206/labs/D1_2_timing_noise.ipynb)
- **Session 3: Profiling a training step**
  - Split a training step into loading, forward, backward and optimiser time
  - Read a profiler table sorted by self time
  - Name the largest bottleneck from the measurements
  - [Lab 1.3: Profile Lina's training step (25 min)](https://sdaiaacademy.github.io/ae-206/labs/D1_3_profile_step.ipynb)
- **Session 4: Fixing an input-bound pipeline**
  - Tell an input-bound step from a compute-bound one
  - Add parallel loading and a cache of preprocessed data
  - Measure the speed-up and say which change bought it
  - [Lab 1.4: Fix the input pipeline (25 min)](https://sdaiaacademy.github.io/ae-206/labs/D1_4_input_pipeline.ipynb)

### Day 2: Memory and training efficiency

Slides: https://sdaiaacademy.github.io/ae-206/slides/day2.md

- **Session 1: Where memory goes**
  - Compute the bytes of weights, gradients and optimiser state
  - Estimate activation memory from layer shapes and batch size
  - Explain the gap between an estimate and a measured peak
  - [Lab 2.1: Memory at three batch sizes (25 min)](https://sdaiaacademy.github.io/ae-206/labs/D2_1_memory_budget.ipynb)
- **Session 2: Mixed precision**
  - Choose between float32, float16 and bfloat16 for a step
  - Train with autocast and compare time and accuracy
  - Read whether a processor has native support before expecting a speed-up
  - [Lab 2.2: Reduced precision (25 min)](https://sdaiaacademy.github.io/ae-206/labs/D2_2_mixed_precision.ipynb)
- **Session 3: Accumulation and checkpointing**
  - Reach an effective batch with gradient accumulation
  - Trade recomputation for memory with activation checkpointing
  - Verify that a memory saving did not change the gradient
  - [Lab 2.3: A large batch in fixed memory (25 min)](https://sdaiaacademy.github.io/ae-206/labs/D2_3_accumulate_checkpoint.ipynb)
- **Session 4: Compilation and fusion**
  - Fold a batch norm into the convolution before it
  - Measure a compiler's first-call cost against its steady-state gain
  - Complete a results table of time, memory and accuracy per setting
  - [Lab 2.4: Compile and fuse (25 min)](https://sdaiaacademy.github.io/ae-206/labs/D2_4_compile_fusion.ipynb)

### Day 3: Inference and compression

Slides: https://sdaiaacademy.github.io/ae-206/slides/day3.md

- **Session 1: Latency, throughput and batching**
  - Measure latency and throughput across batch sizes
  - Explain why batching raises throughput and latency together
  - Choose a batch size for a latency budget
  - [Lab 3.1: Batch size against latency (25 min)](https://sdaiaacademy.github.io/ae-206/labs/D3_1_batching.ipynb)
- **Session 2: Language model inference on a laptop**
  - Measure time to first token and tokens per second
  - Size a key-value cache from the model's shape
  - Route a question by its data classification
  - [Lab 3.2: Hasad on a laptop (25 min)](https://sdaiaacademy.github.io/ae-206/labs/D3_2_lm_inference.ipynb)
- **Session 3: Quantisation**
  - Quantise a classifier to 8 bits with calibration data
  - Compare 8-bit and 4-bit language model files on size, speed and answers
  - Re-measure quality by group after every quantisation
  - [Lab 3.3: Quantise Waraqa and Hasad (25 min)](https://sdaiaacademy.github.io/ae-206/labs/D3_3_quantisation.ipynb)
- **Session 4: Pruning or a smaller model**
  - Prune whole channels by an importance score
  - Recover accuracy with brief fine-tuning
  - Compare a pruned model with a smaller one trained from scratch
  - [Lab 3.4: Prune or shrink (25 min)](https://sdaiaacademy.github.io/ae-206/labs/D3_4_pruning.ipynb)

### Day 4: Export and benchmark

Slides: https://sdaiaacademy.github.io/ae-206/slides/day4.md

- **Session 1: Export to a portable runtime**
  - Export a model to an exchange format with a dynamic batch axis
  - Check runtime outputs against the original within a tolerance
  - Compare graph optimisation levels of a runtime
  - [Lab 4.1: Export and check parity (25 min)](https://sdaiaacademy.github.io/ae-206/labs/D4_1_export_parity.ipynb)
- **Session 2: A benchmark you can trust**
  - Write a benchmark that controls warm-up, repetitions and input mix
  - Report percentiles with bootstrap confidence intervals
  - Re-run a benchmark and judge the agreement
  - [Lab 4.2: A benchmark you can trust (25 min)](https://sdaiaacademy.github.io/ae-206/labs/D4_2_benchmark.ipynb)
- **Session 3: Quality must survive**
  - Compare two models on the same items with a paired test
  - Put an interval on an accuracy difference
  - Accept or reject a variant on the quality floor by crop type
  - [Lab 4.3: Quality by crop type (25 min)](https://sdaiaacademy.github.io/ae-206/labs/D4_3_quality_by_crop.ipynb)
- **Session 4: Budgets and trade-offs**
  - Choose the variant that meets a latency and memory budget
  - Estimate energy per 1,000 predictions
  - Order optimisation changes by measured impact
  - [Lab 4.4: Choose under the budget (25 min)](https://sdaiaacademy.github.io/ae-206/labs/D4_4_budget_choice.ipynb)

### Capstone: Capstone: an optimisation study and its defence

Slides: https://sdaiaacademy.github.io/ae-206/slides/capstone.md

## Capstone

An optimisation study on a model of the participant's choice, submitted as a public repository whose README reports the baseline profile, each change ordered by measured impact, and the final latency, memory and quality against a stated budget with intervals and the hardware used, followed by an eight-minute defence and four minutes of questions

**What you submit:** A public GitHub repository whose README reports the baseline profile, each change ordered by measured impact, and the final latency, memory and quality against the stated budget with intervals and the hardware used; profiler outputs, benchmark logs and the commit history are in the repository. Then the eight-minute defence: the participant re-runs the benchmark live, predicts the effect of a change the assessor proposes and then measures it.

**Rubric:**

- Baseline profile: 20%
- Changes ordered by measured impact: 25%
- Quality re-evaluation: 20%
- Benchmark method: 20%
- Reproducibility: 15%

## Outcomes

- **LO1** Explain how compute, memory capacity and memory bandwidth limit training and inference, and identify which one binds for a described workload, given four workload descriptions with model size, batch size and hardware figures (Awareness) Evidence: a worked estimate for each workload that matches the reference within a stated margin
- **LO2** Measure where time goes in a data pipeline and a training step with profilers and timers, and remove the largest bottleneck, given a slow training script for the leaf-image model (Applied) Evidence: a before-and-after profile and a measured speed-up with the change explained
- **LO3** Compute the memory needed for weights, activations, optimiser state and the key-value cache and check each estimate against measured peak memory, given a small vision model and a small language model at three batch sizes and context lengths (Applied) Evidence: an estimate table next to measured peaks, with each difference explained
- **LO4** Apply mixed precision, batching and compilation and report their effect on speed, memory and accuracy, given the training and inference scripts and a fixed test set (Applied) Evidence: a results table with time per step, throughput, peak memory and accuracy for each setting
- **LO5** Apply post-training quantisation and pruning, export the model to a portable runtime and compare size, latency and quality with the original, given the trained classifier, a small language model and a held-out evaluation set (Applied) Evidence: a comparison table of size, median and tail latency and quality loss with an interval
- **LO6** Prepare a benchmark that controls warm-up, repetitions, batch size and input mix, and report percentiles with confidence intervals, given two model variants and a target laptop (Applied) Evidence: a benchmark script and report that another participant re-runs within 10% of the reported median
- **LO7** Optimise a model and its serving path to meet a stated latency and memory budget on a target device while keeping quality within an agreed tolerance, given a budget set by the company for its farm-office laptops and a quality floor on the held-out set (Performance) Evidence: an optimisation log ordered by measured impact and a final benchmark that meets the budget or explains the gap

## Tools

- **Deep learning framework with a profiler:** PyTorch and its profiler, JAX and its profiler
- **Python profilers and memory tools:** py-spy, Scalene, memray
- **Portable inference runtime:** ONNX Runtime, ExecuTorch, LiteRT
- **Small language model runtime with quantisation:** llama.cpp, Ollama
- **Benchmark harness:** pytest-benchmark, hyperfine, a supplied timing script
- **Analytical performance model:** MLSys·im from the open Machine Learning Systems textbook, a spreadsheet roofline model
- **Statistics:** SciPy, NumPy

## Prerequisites

- AE-202 Deep Learning and Generative AI Fundamentals
- Trains a small neural network with a hand-written training loop
- Reads a histogram and computes a percentile
- Setup: a laptop with uv (any recent processor, 8 GB of memory), or Google Colab, where a free GPU runtime is optional. Nothing downloads a model: both models are in the repository. Run 00_setup_check before Day 1; see labs/README.md.

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

---

SDAIA Academy · AI Engineering Track · Machine Learning Systems on One Machine

Course version 2027.1.0 · engine 1.0.0 · built 2026-11-15 · commit 71acc15
