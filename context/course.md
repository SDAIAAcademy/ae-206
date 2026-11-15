---
course: "Machine Learning Systems on One Machine"
code: "AE-206"
version: "2027.1.0"
scope: "course"
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-206/"
---

# Machine Learning Systems on One Machine

Find where time and memory go on one machine, fix the largest cost first, and prove each speed-up kept its quality

Participants find out where time and memory go between raw data and a served prediction, and fix the largest cost first. For a fictional agritech company they profile a data pipeline and a training step, estimate memory from weights, activations, optimiser state and the key-value cache, apply mixed precision, batching, quantisation and pruning, export a model to a portable runtime, and benchmark latency, throughput and quality with repeated runs and intervals. Labs run on a laptop CPU, with an optional free notebook GPU, and every speed-up must be measured and checked for loss of quality.

- **Audience:** Practitioners. Graduates and early-career developers who have completed the FN courses or passed the placement test, and software engineers moving into AI work.
- **Prerequisites:** AE-202 Deep Learning and Generative AI Fundamentals; Trains a small neural network with a hand-written training loop; Reads a histogram and computes a percentile; Setup: a laptop with uv (any recent processor, 8 GB of memory), or Google Colab, where a free GPU runtime is optional. Nothing downloads a model: both models are in the repository. Run 00_setup_check before Day 1; see labs/README.md.
- **Case study:** Nama Agritech: Nama Agritech is a fictional farm-advisory company whose agents run two models on old farm-office laptops: Waraqa, a leaf-disease classifier trained on 20,000 synthetic leaf images in ten disease classes and four crops, and Hasad, a small language model the course trained to answer 2,000 synthetic Arabic farmer questions in Modern Standard Arabic, Gulf dialect and mixed Arabic and English. Participants measure, profile, estimate memory, compress, export and benchmark both against the budget Nama set for its laptops; all data is synthetic and generated with a fixed seed.

## Outcomes

- LO1 (Awareness): Explain how compute, memory capacity and memory bandwidth limit training and inference, and identify which one binds for a described workload, given four workload descriptions with model size, batch size and hardware figures
- LO2 (Applied): Measure where time goes in a data pipeline and a training step with profilers and timers, and remove the largest bottleneck, given a slow training script for the leaf-image model
- LO3 (Applied): Compute the memory needed for weights, activations, optimiser state and the key-value cache and check each estimate against measured peak memory, given a small vision model and a small language model at three batch sizes and context lengths
- LO4 (Applied): Apply mixed precision, batching and compilation and report their effect on speed, memory and accuracy, given the training and inference scripts and a fixed test set
- LO5 (Applied): Apply post-training quantisation and pruning, export the model to a portable runtime and compare size, latency and quality with the original, given the trained classifier, a small language model and a held-out evaluation set
- LO6 (Applied): Prepare a benchmark that controls warm-up, repetitions, batch size and input mix, and report percentiles with confidence intervals, given two model variants and a target laptop
- LO7 (Performance): Optimise a model and its serving path to meet a stated latency and memory budget on a target device while keeping quality within an agreed tolerance, given a budget set by the company for its farm-office laptops and a quality floor on the held-out set

## Day 1: Measure first

Where time goes, measured before anything is changed

#### [day1#title] Measure first

*SDAIA Academy · Machine Learning Systems on One Machine*

Day 1: Where time goes, measured before anything is changed

AI Engineering Track · 4 teaching days + capstone day · 4 sessions · 4 labs · AE-206

#### [day1#four-days-one-machine] Four days on one laptop, from raw data to a served prediction

*The course at a glance*

- **Day 1 · Measure first.** The machine's limits, timers, profilers, an input-bound pipeline.
- **Day 2 · Memory and training.** Where memory goes, mixed precision, accumulation, checkpointing, compilation.
- **Day 3 · Inference and compression.** Batching, a language model on a laptop, quantisation, pruning.
- **Day 4 · Export and benchmark.** A portable runtime, a benchmark you can trust, quality by group, a budget.

> **Day 5 · Capstone.** An optimisation study on a model you choose, in a public repository, defended in eight minutes with a live re-run of your benchmark.

#### [day1#nama-agritech] Nama Agritech: two models on old farm-office laptops

*One case study all week*

A fictional farm-advisory company. Its agents visit farms with a laptop that is several years old, often offline.

- **Waraqa ورقة.** Classifies a photographed leaf into healthy or one of nine diseases, on tomato, cucumber, citrus and date palm.
- **Hasad حصاد.** A small language model that answers farmers' questions in Modern Standard Arabic, Gulf dialect and mixed Arabic and English.
- **The budget.** Fast enough per leaf, small enough to ship, light on memory, and no quality lost where farmers would notice.

#### [day1#setup-check] Run the setup check now: it builds the data and loads both models

*Before Session 1*

1. Colab: open `00_setup_check.ipynb` from the course page and run all cells.
2. Laptop: `git clone` the course, then in `labs/` run `uv sync` and `uv run jupyter lab`.
3. The check prints your processor, threads and memory, builds the held-out leaves and answers one farmer question.
4. Its last line says whether this machine can compile models (Lab 2.4).

- **Quick mode.** Set `NAMA_QUICK=1` for smaller runs. Every solution finishes in minutes on a laptop processor.
- **Write down your machine.** Every number this week is a number on a machine.

#### [day1#agenda] Measure first

*Today*

- **Session 1 · Compute, memory and bandwidth**: Compute, memory capacity and bandwidth as the three limits, and the roofline that says which one binds
- **Session 2 · Timers, profilers and noise**: Timers, warm-up, variance and percentiles: why one timing is never the answer
- **Session 3 · Profiling a training step**: The anatomy of a training step, read with timers and with the framework's profiler
- **Session 4 · Fixing an input-bound pipeline**: Decoding, file formats, parallel workers and caching: removing an input bottleneck

### Session 1: Compute, memory and bandwidth

#### [day1#s1] Compute, memory and bandwidth

*Session 1*

- Estimate the FLOPs and bytes moved of a layer
- Place a layer on a roofline from its arithmetic intensity
- Name the binding limit of a described workload

#### [day1#three-limits] Every step is limited by compute, memory capacity or memory bandwidth

*What can make a step slow*

- **Compute.** Floating-point operations per second the processor can do. A large matrix product runs into this roof.
- **Capacity.** Bytes that fit at all. Too many and the run fails, or spills to slower memory. Day 2 is about this one.
- **Bandwidth.** Bytes per second between memory and the cores. Elementwise layers and decoding one token at a time run into this roof.

> **What goes wrong.** Optimising the limit that does not bind. Fewer FLOPs do not help a step that waits for memory.

#### [day1#count-flops-bytes] Count FLOPs and bytes by hand: it takes two lines

*Before any timer*

| Layer | FLOPs | Bytes moved (float32) | Intensity |
| --- | --- | --- | --- |
| A · fully connected, batch 512, 1,024 to 1,024 | 2 x 512 x 1,024 x 1,024 = 1.07 G | 4 x (input + weight + output) = 8.4 MB | 128 FLOP/byte |
| B · add a bias, then ReLU, on 4,096 x 4,096 | 2 per element = 0.034 G | read and write each element = 134 MB | 0.25 FLOP/byte |

Arithmetic intensity = FLOPs divided by bytes moved. It is a property of the layer, not of the machine.

#### [day1#roofline-rule] Attainable speed is the lower of two roofs

*The roofline idea*

- **attainable = min(peak, intensity x bandwidth).** Below the ridge point a layer waits for memory; above it, for arithmetic.
- **Ridge point = peak / bandwidth.** The intensity at which the two roofs meet.

- A **floor on time**, never a promise: real layers sit under their roof.
- It tells you **which roof** to push: fewer bytes or fewer FLOPs.
- Batching raises intensity: the weights are read once for more rows.

#### [day1#this-laptop] A thin laptop: 138 GFLOP/s, 33.7 GB/s, a ridge at 4.1 FLOP per byte

*Measured in Lab 1.1 on the course's laptop*

- **138** GFLOP/s, a 2,048-square matrix product, 4 threads
- **33.7** GB/s, a 256 MB copy
- **4.1** FLOP per byte at the ridge

Layer A (128 FLOP/byte) is far above the ridge: compute-bound. Layer B (0.25) is far below: memory-bound. Measured: A 6.7 ms against a predicted 7.8 ms; B 6.9 ms against 4.0 ms.

#### [day1#caches-and-reuse] Data reused close to the cores is cheap; data fetched from main memory is not

*Caches and data movement*

1. **Registers** a few hundred bytes per core
2. **L1 and L2 caches** tens of kilobytes to a few megabytes
3. **Last-level cache** megabytes, shared
4. **Main memory** gigabytes, the bandwidth you measured

> **Why the matrix product is fast.** It works on tiles that stay in cache and reuses each number many times. The 256 MB copy cannot: it reads everything once.

#### [day1#cpu-gpu-npu] Each processor trades generality for throughput on dense arithmetic

*CPUs, GPUs and neural accelerators*

- **CPU.** Few strong cores with wide vector units. Good at branches, small batches and everything around the model.
- **GPU.** Thousands of simple lanes and very high memory bandwidth. Needs large, regular work to stay busy.
- **Neural accelerator.** Fixed-function units for matrix products in low precision, on many new laptops and phones. Needs a runtime that supports it.

> **Same roofline, different numbers.** Each has its own peak and bandwidth, so the same layer can bind differently on each.

#### [day1#analytical-model] An analytical model gives the first estimate; the measurement checks it

*Pre-work, used today*

- Pre-work (`labs/prework/`): a reading and a roofline estimate for four workloads, by hand or with a spreadsheet.
- **MLSys·im**, from the open Machine Learning Systems textbook, models training and inference analytically.
- An estimate within a factor of two is useful: it says which limit to look at.

- **LO1 evidence.** A worked estimate for each of four workloads that matches the reference within a stated margin, done without AI in the supervised check.

#### [day1#checkpoint-1] Which layer gets faster with fewer FLOPs?

*Checkpoint*

**Question:** On a laptop with a ridge point of 3 FLOP per byte, you can halve the FLOPs of one layer. Which layer's time falls most?

- A. A matrix product with an intensity of 120 FLOP per byte
- B. An elementwise ReLU with an intensity of 0.25 FLOP per byte
- C. A layer sitting exactly at the ridge point
- D. All three equally, since time is FLOPs over peak

#### [day1#lab-1-1] Two layers on the roofline

Lab 1.1 · 25 min

Measure the laptop's peak compute and bandwidth, estimate the arithmetic intensity of a matrix product and an elementwise layer, predict which one is memory-bound, then time both and place them on a roofline.

[D1_1_roofline.ipynb](https://sdaiaacademy.github.io/ae-206/labs/D1_1_roofline.ipynb)

- [ ] Measure peak compute with a large matrix product (5 min)
- [ ] Measure bandwidth with a large copy (4 min)
- [ ] Estimate FLOPs and bytes of both layers by hand (6 min)
- [ ] Predict, then time both layers (5 min)
- [ ] Place both on the roofline (5 min)
- [ ] ✱ Stretch: Repeat the matrix product at batch 1 and watch it slide under the bandwidth roof

### Session 2: Timers, profilers and noise

#### [day1#s2] Timers, profilers and noise

*Session 2*

- Time a call correctly with a monotonic clock and warm-up
- Report a distribution with its median and tail percentiles
- Show how background load changes the tail

#### [day1#right-clock] Time with a monotonic clock, around the call, many times

*Timers*

```python
# file: Lab 1.2
ms = []
for _ in range(100):
    t0 = time.perf_counter()
    with torch.no_grad():
        model(leaf)
    ms.append((time.perf_counter() - t0) * 1000)
```

- `perf_counter` never jumps; `time.time` follows the wall clock, which can be adjusted.
- On a GPU, call `torch.cuda.synchronize()` before reading the clock: the work is asynchronous.
- One timing is an anecdote. A hundred are a distribution.

#### [day1#latency-distribution] Latency has a floor and a long right tail

*Measured in Lab 1.2: one Waraqa inference, batch 1, 100 calls*

- **5.7 ms** median
- **7.4 ms** 95th percentile
- **8.4 ms** 99th percentile

> **Report the median and the 95th percentile.** The mean is pulled up by the tail; a standard deviation assumes a bell curve that latency never has.

#### [day1#cold-start] The first call is a different event: report it, do not average it

*The first calls*

**1.3 x** the warm median, for the first call of a freshly loaded model (Lab 1.2)

- The first call allocates memory, picks kernels and fills caches.
- A user who opens the app pays it once; a benchmark that includes it in 100 calls hides both numbers.
- So: warm up, then time; and time the cold start on its own.

#### [day1#background-load] Background load moves the tail before the median

*Measured in Lab 1.2 with two busy processes*

| Machine state | Median ms | p95 ms | p99 ms |
| --- | --- | --- | --- |
| Quiet | 5.7 | 7.4 | 8.4 |
| Two processes computing in a loop | 4.3 | 9.0 | 17.6 |

A farm-office laptop runs a browser, a sync client and an antivirus scan. Write what else was running next to every number.

#### [day1#timing-rule] Warm up, repeat, report percentiles and the conditions

*The rule for the week*

1. **1 · Warm up** untimed calls first; the cold start reported apart
2. **2 · Repeat** at least 100 calls on fixed inputs
3. **3 · Summarise** median and 95th percentile, later with intervals
4. **4 · Record** processor, threads, power mode, what else ran

#### [day1#timers-vs-profilers] A timer says how long; a profiler says what ran

*Timers and profilers*

- **Timers.** Cheap, honest about wall time, blind to what happened inside. Put them around phases you name.
- **Profilers.** Record every operator or sample every call stack. Show where time goes inside a phase, with some overhead of their own.

> **Use both.** Timers to find the slow phase, a profiler to find why it is slow, timers again to prove the fix.

#### [day1#checkpoint-2] Which number goes in the budget?

*Checkpoint*

**Question:** 100 warm calls of Waraqa: mean 6.1 ms, median 5.2 ms, 95th percentile 9.8 ms, first call 140 ms. Ghadeer's budget says "per leaf, in the office". Which number do you compare with it?

- A. The mean, 6.1 ms
- B. The median, 5.2 ms
- C. The 95th percentile, 9.8 ms, with the cold start reported apart
- D. The first call, 140 ms

#### [day1#lab-1-2] Noise in one inference

Lab 1.2 · 25 min

Time one Waraqa inference 100 times, plot the distribution and show how the first calls and a busy machine change it.

[D1_2_timing_noise.ipynb](https://sdaiaacademy.github.io/ae-206/labs/D1_2_timing_noise.ipynb)

- [ ] Time one inference 100 times (5 min)
- [ ] Plot the distribution and read its percentiles (5 min)
- [ ] Cold start: the first calls against the rest (5 min)
- [ ] Background load: repeat while another process computes (6 min)
- [ ] Write the timing rule for the course (4 min)
- [ ] ✱ Stretch: Change the thread count and see which percentile moves most

### Session 3: Profiling a training step

#### [day1#s3] Profiling a training step

*Session 3*

- Split a training step into loading, forward, backward and optimiser time
- Read a profiler table sorted by self time
- Name the largest bottleneck from the measurements

#### [day1#step-anatomy] A training step has four phases, and any of them can be the slow one

*A training step*

1. **1 · Load** read, decode, augment, batch
2. **2 · Forward** compute the loss
3. **3 · Backward** compute the gradients
4. **4 · Optimiser** update the weights

> **What goes wrong.** Buying a faster processor for a step that waits for its data.

#### [day1#lina-script] Lina's loader decodes a 224-pixel PNG and rotates it at full size, for every leaf, every epoch

*The supplied slow script*

```python
# file: nama.py · FieldExportSlow
img = Image.open(self.folder / name).convert("RGB")   # decode a PNG
img = img.rotate(angle, resample=Image.BILINEAR)       # augment at 224 px
img = img.resize((64, 64), Image.BILINEAR)             # then shrink
x = torch.tensor(np.array(img), dtype=torch.float32)   # one leaf at a time
# ... in the main process: num_workers=0
```

#### [day1#phase-timers] Four timers split the step; on a GPU, synchronise before each reading

*Timers around phases*

```python
# file: lina_train.py · phase_step
t0 = time.perf_counter(); x, y = next(batches)
t1 = time.perf_counter(); loss = F.cross_entropy(model(x), y)
t2 = time.perf_counter(); loss.backward()
t3 = time.perf_counter(); opt.step(); opt.zero_grad()
t4 = time.perf_counter()
# GPU: torch.cuda.synchronize() before each perf_counter()
```

#### [day1#profiler-table] The profiler table: sort by self time, read the top rows

*Measured in Lab 1.3 with torch.profiler*

| Row | Self CPU ms | What it is |
| --- | --- | --- |
| `enumerate(DataLoader)` | 879.0 | the loader: reading, decoding and rotating the PNG leaves |
| `aten::convolution_backward` | 431.2 | gradients of the six convolutions |
| `aten::mkldnn_convolution` | 180.5 | the six convolutions, forward |
| `aten::max_pool2d_with_indices` | 41.8 | max pooling, forward |
| `aten::native_batch_norm` | 36.1 | batch norm, forward |

**Self** time is spent in the operator itself; **total** includes what it calls.

#### [day1#lina-ranking] Lina's step is input-bound: loading is the largest phase

*Measured in Lab 1.3: median of the timed steps, batch 64*

| Item | Value |
| --- | --- |
| Loading | 1,420 ms, 66% |
| Backward | 385 ms, 18% |
| Forward | 342 ms, 16% |
| Optimiser | 5 ms, 0.2% |

One step: 2,142 ms. One epoch of 16,000 leaves at this rate: about 9 minutes.

#### [day1#amdahl-ceiling] Removing one phase can never beat the sum of the others

*The ceiling of a fix*

**2.9 x** the most Lina's step can gain if loading cost nothing (Lab 1.3)

- Speed-up limit = 1 / (1 - share of the phase you fix).
- Compute this before you start: it says whether the fix is worth a day.
- After the fix, the next phase is the bottleneck: measure again.

#### [day1#flame-graphs] A flame graph shows the whole program, loader included

*Python-level profilers*

- **Sampling profilers** (py-spy) read the call stack many times a second; width of a tower is time.
- **Scalene** separates Python time, native time and memory per line.
- **memray** tracks every allocation (Linux and macOS only).

```bash
# file: from labs/
py-spy record -o artifacts/lina.svg \
  -- python lina_train.py --steps 10
```

#### [day1#checkpoint-3] What do you fix first?

*Checkpoint*

**Question:** A step's phases: loading 60%, backward 25%, forward 13%, optimiser 2%. The profiler's largest operator row is `convolution_backward`. What do you work on first?

- A. The backward convolution, the largest profiler row
- B. Loading, the largest measured phase
- C. The optimiser, by switching to SGD
- D. A faster processor for forward and backward

#### [day1#lab-1-3] Profile Lina's training step

Lab 1.3 · 25 min

Profile one step of the slow training script on the field app's PNG export and rank data loading, forward pass, backward pass and optimiser time.

[D1_3_profile_step.ipynb](https://sdaiaacademy.github.io/ae-206/labs/D1_3_profile_step.ipynb)

- [ ] Run the slow script and time its first steps (4 min)
- [ ] Time each phase with timers (6 min)
- [ ] Profile one step with the framework profiler (6 min)
- [ ] Rank the four phases (5 min)
- [ ] Name the largest bottleneck (4 min)
- [ ] ✱ Stretch: Record a flame graph of the same script with py-spy

### Session 4: Fixing an input-bound pipeline

#### [day1#s4] Fixing an input-bound pipeline

*Session 4*

- Tell an input-bound step from a compute-bound one
- Add parallel loading and a cache of preprocessed data
- Measure the speed-up and say which change bought it

#### [day1#input-bound] A step is input-bound when the model waits for its next batch

*Input-bound against compute-bound*

- **Input-bound.** Loading is the largest phase. A faster model changes nothing. Fix the pipeline.
- **Compute-bound.** Forward and backward dominate. The pipeline is good enough. Days 2 and 3 work here.

> **Signs.** A GPU at low utilisation; one core at 100% while others idle; step time that does not change with model size.

#### [day1#formats-decode] Decoding is compute too, and the file format sets its price

*Measured in Lab 1.4: one leaf, 50 decodes*

| Format of one leaf | Decode ms | Size kB |
| --- | --- | --- |
| PNG, 224 pixels (the field app's export) | 1.85 | 119 |
| JPEG quality 90, 64 pixels | 0.17 | 1.4 |

JPEG is lossy: the two decoded leaves differ by 2.5 of 255 per pixel on average. Whether that matters is a quality question, measured, not seen.

#### [day1#parallel-workers] Workers prepare the next batches while the model trains on this one

*Parallel loading and prefetching*

```python
# file: Lab 1.4
DataLoader(ds, batch_size=64, shuffle=True,
           num_workers=4,
           persistent_workers=True,
           prefetch_factor=2)
```

- Workers **hide** loading behind compute; they do not make it cheaper.
- They take cores the training step also wants.
- On Windows and macOS each worker is a fresh Python process: the first batch is slow.

#### [day1#cache-preprocessed] Decode once, store the 64-pixel bytes, read bytes every epoch

*Caching preprocessed data*

**50 MB** for 4,096 leaves as one uint8 array, built once in 88 s (Lab 1.4)

- The cache is right when preprocessing is the same every epoch.
- It is wrong when augmentation must differ each epoch at full size, or when the data changes faster than the cache.

> **What it changed besides speed.** Lina rotated by any angle; the cache rotates by multiples of 90 degrees. A speed-up that changes what the model learns is checked for quality.

#### [day1#pipeline-before-after] The before-and-after table is the evidence, not the code change

*Measured in Lab 1.4, full mode*

| Pipeline | Step ms | Loading ms | Speed-up |
| --- | --- | --- | --- |
| Baseline (Lina) | 1,059 | 343 | 1.0x |
| 4 workers | 1,001 | 1 | 1.1x |
| Cache, main process | 725 | 11 | 1.5x |
| Cache, 4 workers | 693 | 0 | 1.5x |

When loading is near zero the step is compute-bound: what is left is forward, backward and the optimiser.

#### [day1#checkpoint-4] Loading vanished. Why is the step only a little faster?

*Checkpoint*

**Question:** With four workers, the loading phase drops from 800 ms to almost zero, but the step goes from 2,900 ms to 2,200 ms. What explains it?

- A. The workers are not running
- B. It is timer noise; the real speed-up is larger
- C. Decoding became cheaper, but not by enough
- D. The rest of the step was always about 2,100 ms, and the workers take cores from it

#### [day1#lab-1-4] Fix the input pipeline

Lab 1.4 · 25 min

Add parallel loading, a cache of decoded images and a faster image format to Lina's loader and measure the change in step time.

[D1_4_input_pipeline.ipynb](https://sdaiaacademy.github.io/ae-206/labs/D1_4_input_pipeline.ipynb)

- [ ] Baseline: step time and loader time (4 min)
- [ ] Parallel workers (6 min)
- [ ] Cache the decoded images (6 min)
- [ ] A faster image format (5 min)
- [ ] The before-and-after table (4 min)
- [ ] ✱ Stretch: Find the worker count after which the step stops getting faster

#### [day1#day-1-recap] Three things to keep from Day 1

*Day 1 recap*

- **Know the roof.** Count FLOPs and bytes; the intensity says whether compute or bandwidth binds.
- **Distributions, not numbers.** Warm up, repeat, report the median and the 95th percentile and the machine.
- **Largest phase first.** Time the phases, profile the largest, fix it, measure again.

#### [day1#tomorrow-memory] Memory and training efficiency

*Tomorrow · Day 2*

Tomorrow you estimate what a training step holds in memory and check it against a measured peak, then test mixed precision, gradient accumulation, activation checkpointing and compilation on Waraqa. Bring your Lab 1.1 peak and bandwidth figures.

## Day 2: Memory and training efficiency

Memory first, then the three ways to train faster in it

#### [day2#title] Memory and training efficiency

*SDAIA Academy · Machine Learning Systems on One Machine*

Day 2: Memory first, then the three ways to train faster in it

AI Engineering Track · 4 teaching days + capstone day · 4 sessions · 4 labs · AE-206

#### [day2#agenda] Memory and training efficiency

*Today*

- **Session 1 · Where memory goes**: Weights, gradients, optimiser state and activations: an estimate you can check
- **Session 2 · Mixed precision**: Number formats, autocast and what the hardware actually accelerates
- **Session 3 · Accumulation and checkpointing**: Gradient accumulation and activation checkpointing: a large batch in fixed memory
- **Session 4 · Compilation and fusion**: Graphs, operator fusion and compilation: the first-call cost and the steady state

#### [day2#review-day-1] Which measurement settles it?

*Review of Day 1*

**Question:** Lina says her new loader made training "much faster". She shows one epoch timed before (31 min) and one after (24 min), each run once, on different days. What do you ask for?

- A. Nothing: 7 minutes is too large to be noise
- B. Phase times of both loaders over the same steps on the same machine, start-up excluded
- C. The framework profiler's operator table for the new loader
- D. The roofline estimate of the loader

### Session 1: Where memory goes

#### [day2#s1] Where memory goes

*Session 1*

- Compute the bytes of weights, gradients and optimiser state
- Estimate activation memory from layer shapes and batch size
- Explain the gap between an estimate and a measured peak

#### [day2#four-parts-of-memory] Training memory has four parts, and only one grows with the batch

*What a training step holds*

- **Weights.** 4 bytes per parameter in float32.
- **Gradients.** One per weight: another 4 bytes.
- **Optimiser state.** AdamW keeps two numbers per weight: 8 bytes.
- **Activations.** What the forward pass keeps for the backward pass: grows with batch and image size.

> **Rule of thumb.** 16 bytes per parameter before any activation, for float32 and AdamW.

#### [day2#activation-estimate] Each convolution block keeps about two tensors the size of its output

*Estimate by hand*

| Block output (per leaf) | Numbers |
| --- | --- |
| 32 x 64 x 64, twice | 2 x 131,072 |
| 64 x 32 x 32, twice | 2 x 65,536 |
| 128 x 16 x 16 | 32,768 |
| 128 x 8 x 8 | 8,192 |
| Sum, x 2 tensors x 4 bytes | about 3.5 MB per leaf |

#### [day2#memory-measured] Activations dominate, and grow in proportion to the batch

*Measured in Lab 2.1: one AdamW step of Waraqa*

| Batch | Estimate MB | Saved activations MB | Measured peak MB |
| --- | --- | --- | --- |
| 16 | 60 | 67 | 94 |
| 64 | 227 | 270 | 325 |
| 256 | 894 | 1,078 | 1,165 |

Saved activations are counted exactly by autograd hooks; the peak is the process's resident memory, sampled during the step.

#### [day2#memory-gaps] Each gap between estimate and measurement has a name

*Explain every difference*

- **Estimate < saved.** Pooling indices (8-byte integers) and the input batch were not counted.
- **Saved < peak.** Gradients of activations and convolution work buffers are alive during the backward pass.
- **Small batches.** Fixed costs dominate: weights, optimiser state, the allocator's own reserve.

> **LO3 evidence.** The estimate table next to the measured peaks, with each difference explained in a line.

#### [day2#memory-scaling] For a language model, the fixed part is the problem before any batch

*The same arithmetic, scaled up*

| Model | Weights, float32 | + gradients + AdamW |
| --- | --- | --- |
| Waraqa, 0.29 M parameters | 1.2 MB | 4.6 MB |
| A 1 B parameter model | 4 GB | 16 GB |
| A 7 B parameter model | 28 GB | 112 GB |

That is why language models are trained in reduced precision with sharded optimiser state, and served with quantised weights (Day 3).

#### [day2#checkpoint-1] What happens at batch 512?

*Checkpoint*

**Question:** Waraqa's step peaks at 1,000 MB at batch 256, of which about 10 MB is weights, gradients and optimiser state. Predict the peak at batch 512.

- A. About 1,010 MB: only the batch's input grows
- B. About 4,000 MB: memory grows with the square of the batch
- C. About 3,000 MB: the optimiser state triples
- D. About 2,000 MB: the activations double, the fixed part does not

#### [day2#lab-2-1] Memory at three batch sizes

Lab 2.1 · 25 min

Estimate, then measure, peak training memory of Waraqa for weights, activations and optimiser state at batch 16, 64 and 256.

[D2_1_memory_budget.ipynb](https://sdaiaacademy.github.io/ae-206/labs/D2_1_memory_budget.ipynb)

- [ ] Count weights, gradients and optimiser state (5 min)
- [ ] Estimate activations by hand (6 min)
- [ ] Measure peak memory at three batch sizes (7 min)
- [ ] Explain each difference (4 min)
- [ ] Predict batch 512 before running it (3 min)
- [ ] ✱ Stretch: Swap AdamW for SGD with momentum and predict the change first

### Session 2: Mixed precision

#### [day2#s2] Mixed precision

*Session 2*

- Choose between float32, float16 and bfloat16 for a step
- Train with autocast and compare time and accuracy
- Read whether a processor has native support before expecting a speed-up

#### [day2#number-formats] Sixteen bits buy half the bytes and give up range or precision

*Number formats (recap of AE-202)*

| Format | Bits | Exponent / fraction | Gives up |
| --- | --- | --- | --- |
| float32 | 32 | 8 / 23 | nothing: the reference |
| float16 | 16 | 5 / 10 | range: small gradients underflow to zero |
| bfloat16 | 16 | 8 / 7 | precision: the range of float32, fewer digits |
| int8 | 8 | integer with a scale | both, recovered by calibration (Day 3) |

#### [day2#autocast] Autocast runs the heavy operations in 16 bits and keeps the rest in 32

*Mixed precision*

```python
# file: Lab 2.2
with torch.autocast("cpu", dtype=torch.bfloat16):
    out = model(x)                  # convolutions in bfloat16
    loss = F.cross_entropy(out.float(), y)
loss.backward()                      # gradients flow back
opt.step()                           # float32 master weights
```

- **Mixed**: weights and optimiser stay in float32; small updates would vanish in 16 bits.
- On a CPU, autocast defaults to bfloat16; on a GPU to float16.
- float16 needs a **gradient scaler** to keep small gradients from underflowing.

#### [day2#hardware-support] Reduced precision is faster only where the hardware computes in it

*Ask the hardware first*

- PyTorch reports the vector extensions it uses: `torch.backends.cpu.get_cpu_capability()`.
- The course laptop reports **AVX2**: no native bfloat16 arithmetic.
- Without it, bfloat16 numbers are converted to float32 to compute and back: extra work.

- **Predict before you run.** Write down faster, same or slower. Then measure.

#### [day2#precision-result] On a processor without native bfloat16, autocast made each step 24 times slower, at the same accuracy

*Measured in Lab 2.2: 12 steps of 64 leaves from the warm-start checkpoint*

| Setting | ms per step | Leaves per s | Peak MB | Val accuracy |
| --- | --- | --- | --- | --- |
| float32 | 929 | 69 | 324 | 0.777 |
| bfloat16 autocast | 22,252 | 3 | 280 | 0.776 |

The blog post's "twice as fast" may hold on its hardware. On this one, the measurement decides.

#### [day2#claims-and-hardware] A speed-up without its hardware, settings and quality is not a result

*Reading a speed-up claim*

1. **Hardware** which processor, which extensions
2. **Settings** batch, threads, what was measured
3. **Both sides** the same steps, the same data
4. **Quality** accuracy after, on the same set

#### [day2#checkpoint-2] Why did bfloat16 not help here?

*Checkpoint*

**Question:** On a laptop whose processor reports AVX2 only, bfloat16 autocast made Waraqa's training step slower with the same accuracy. What is the best explanation?

- A. The model is too small to benefit from any optimisation
- B. bfloat16 lost precision, so the step had to do more work
- C. The processor has no native bfloat16 arithmetic, so each operation pays for conversions
- D. The optimiser now runs in bfloat16 and needs more steps

#### [day2#lab-2-2] Reduced precision

Lab 2.2 · 25 min

Train Waraqa from a warm-start checkpoint with and without bfloat16 autocast and compare time per step and validation accuracy.

[D2_2_mixed_precision.ipynb](https://sdaiaacademy.github.io/ae-206/labs/D2_2_mixed_precision.ipynb)

- [ ] What this processor supports (3 min)
- [ ] Train in 32-bit for a fixed number of steps (6 min)
- [ ] Train the same steps with bfloat16 autocast (6 min)
- [ ] Validation accuracy of both (5 min)
- [ ] The results row for each setting (5 min)
- [ ] ✱ Stretch: On a Colab GPU, float16 autocast with a gradient scaler

### Session 3: Accumulation and checkpointing

#### [day2#s3] Accumulation and checkpointing

*Session 3*

- Reach an effective batch with gradient accumulation
- Trade recomputation for memory with activation checkpointing
- Verify that a memory saving did not change the gradient

#### [day2#effective-batch] Lina's recipe needs an effective batch of 256; the budget allows 400 MB

*The problem*

**1,174 MB** measured peak of one batch of 256 (Lab 2.3)

- Mutaib's rule: monthly retraining on a farm-office laptop may use at most 400 MB above the process at rest.
- The learning rate and schedule were tuned for 256: a smaller batch changes training.
- Two ways to keep 256: accumulate, or recompute.

#### [day2#gradient-accumulation] Four micro-batches of 64, one optimiser step

*Gradient accumulation*

```python
# file: Lab 2.3
for xm, ym in zip(x.chunk(4), y.chunk(4)):
    loss = F.cross_entropy(model(xm), ym) / 4
    loss.backward()        # gradients add up in .grad
opt.step()
opt.zero_grad()
```

- Divide each loss by the number of micro-batches: the sum is the mean over 256.
- Activations of one micro-batch at a time: about a quarter of the memory.
- Costs time: four smaller calls do less work in parallel.

#### [day2#same-gradient] Accumulation equals a large batch only for layers that treat each example alone

*Checked in Lab 2.3*

| Batch norm mode | Largest relative gradient difference |
| --- | --- |
| Evaluation (stored statistics) | 3.8e-05 |
| Training (statistics per batch) | 1.2 |

With batch norm, each micro-batch is normalised by its own statistics. It usually trains fine; it is not the same computation.

#### [day2#activation-checkpointing] Keep a few activations, recompute the rest during the backward pass

*Activation checkpointing*

```python
# file: Lab 2.3
from torch.utils.checkpoint import checkpoint_sequential
feats = checkpoint_sequential(model.features, 4, x,
                              use_reentrant=False)
logits = model.head(feats.mean(dim=(2, 3)))
```

- Only the boundaries of 4 segments are kept; each segment's inside is recomputed when its gradient is needed.
- Trades about one extra forward pass for memory.
- The idea is from Chen et al., 2016: memory that grows more slowly than depth.

#### [day2#fit-table] Accumulation fits; checkpointing cuts memory, not enough here

*Measured in Lab 2.3, budget 400 MB*

| Method | Peak MB | Fits | s per step | Time |
| --- | --- | --- | --- | --- |
| One batch of 256 | 1,174 | no | 3.63 | 1.00x |
| Accumulation, 4 x 64 | 319 | yes | 3.67 | 1.01x |
| Checkpointing, 4 segments, 256 | 807 | no | 4.62 | 1.27x |

#### [day2#checkpoint-3] What does activation checkpointing reduce?

*Checkpoint*

**Question:** A language model of 2 B parameters does not fit for training on a 24 GB GPU even at batch 1. Will activation checkpointing make it fit?

- A. Unlikely: weights, gradients and AdamW state already need about 32 GB, which checkpointing does not touch
- B. Yes: checkpointing reduces all training memory
- C. Not alone, but gradient accumulation will
- D. Yes, if the checkpoints are written to disk

#### [day2#lab-2-3] A large batch in fixed memory

Lab 2.3 · 25 min

Fit an effective batch of 256 into Nama's training memory budget with gradient accumulation and activation checkpointing, and measure what each costs in time.

[D2_3_accumulate_checkpoint.ipynb](https://sdaiaacademy.github.io/ae-206/labs/D2_3_accumulate_checkpoint.ipynb)

- [ ] Batch 256 against the budget (4 min)
- [ ] Accumulation: four micro-batches of 64 (6 min)
- [ ] Check: the same gradient as one large batch (5 min)
- [ ] Activation checkpointing (6 min)
- [ ] Memory and time per method (4 min)
- [ ] ✱ Stretch: Checkpoint two segments against four and find the cheapest that fits

### Session 4: Compilation and fusion

#### [day2#s4] Compilation and fusion

*Session 4*

- Fold a batch norm into the convolution before it
- Measure a compiler's first-call cost against its steady-state gain
- Complete a results table of time, memory and accuracy per setting

#### [day2#why-fusion] Separate operations read and write the whole activation each time

*Operator fusion*

**Convolution** writes z → **Batch norm** reads z, writes y → **ReLU** reads y, writes y

> **Fused.** One kernel computes the convolution, the affine map and the ReLU while the data is still in cache: one write instead of three.

#### [day2#fold-batch-norm] At inference, batch norm folds into the convolution's weights

*One fusion by hand*

```python
# file: Lab 2.4
s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
W_new = conv.weight * s.reshape(-1, 1, 1, 1)
b_new = bn.bias - bn.running_mean * s
```

- At inference batch norm is a fixed affine map per channel; a convolution is linear: the two combine.
- Not during training: the statistics change every batch.
- Outputs agree to rounding: largest logit difference 5.7e-06 in Lab 2.4.

#### [day2#compilers] A compiler captures the graph, fuses operations and generates code once

*Compilation*

- `torch.compile(model)`: capture on the first call, then reuse the generated code.
- Its default backend generates C++ for processors: it needs a C++ compiler.
- Colab, Linux and macOS usually have one; the course's Windows laptop did not.

- **Two numbers, always.** The first-call cost (capture and code generation) and the steady-state speed per call.

#### [day2#compile-result] Folding batch norm bought 6%; compiling bought nothing here, after a 13-second first call

*Measured in Lab 2.4: inference, batch 32*

| Setting | ms per batch | Leaves per s | Accuracy |
| --- | --- | --- | --- |
| Eager | 130.3 | 246 | 99.1% |
| Batch norm folded | 122.3 | 262 | 99.1% |
| Folded and compiled (aot_eager) | 135.0 | 237 | 99.1% |

First call with compilation: 13.0 s (backend aot_eager).

#### [day2#lo4-results] One table: time, throughput, peak memory and accuracy per setting

*The day's evidence*

- Training rows: float32, bfloat16 autocast, accumulation, checkpointing.
- Inference rows: eager, batch norm folded, compiled.
- Each row names its machine, batch and threads.

- **LO4 evidence.** A results table with time per step, throughput, peak memory and accuracy for each setting.

#### [day2#checkpoint-4] Is compilation worth it here?

*Checkpoint*

**Question:** Compiling Waraqa costs 20 s on the first call and saves 2 ms per batch. The field app starts, classifies about 50 leaves, and closes. Should it compile at start-up?

- A. Yes: 2 ms per batch is a 20% gain
- B. No: 50 leaves save well under a second, and the agent waits 20 s at every start
- C. No: compilation lowers accuracy
- D. Yes: the gain grows the longer the laptop runs

#### [day2#lab-2-4] Compile and fuse

Lab 2.4 · 25 min

Fold batch norm into Waraqa's convolutions by hand, compile the model where a compiler is available, and measure the first-call cost and the steady-state speed-up.

[D2_4_compile_fusion.ipynb](https://sdaiaacademy.github.io/ae-206/labs/D2_4_compile_fusion.ipynb)

- [ ] Eager baseline (4 min)
- [ ] Fold batch norm into the convolution (7 min)
- [ ] Check that the outputs match (4 min)
- [ ] Compile: first call and steady state (6 min)
- [ ] Complete the results table (4 min)
- [ ] ✱ Stretch: In Colab, compare the default compile mode with max-autotune

#### [day2#day-2-recap] Three things to keep from Day 2

*Day 2 recap*

- **Estimate, then measure.** 16 bytes per parameter plus activations that scale with the batch; name every gap.
- **Hardware decides.** Reduced precision and compilers help where the machine supports them. Measure on yours.
- **Memory for time.** Accumulation and checkpointing fit a batch into a budget and pay in time.

#### [day2#tomorrow-inference] Inference and compression

*Tomorrow · Day 3*

Tomorrow you serve Waraqa at batch sizes from 1 to 64, run Hasad on the laptop and size its key-value cache, decide which farmer questions may leave the device, then quantise and prune both models and measure what each change costs in quality.

## Day 3: Inference and compression

Serving one prediction fast, then making the model smaller

#### [day3#title] Inference and compression

*SDAIA Academy · Machine Learning Systems on One Machine*

Day 3: Serving one prediction fast, then making the model smaller

AI Engineering Track · 4 teaching days + capstone day · 4 sessions · 4 labs · AE-206

#### [day3#agenda] Inference and compression

*Today*

- **Session 1 · Latency, throughput and batching**: Latency against throughput, batching and the queue that batching creates
- **Session 2 · Language model inference on a laptop**: Prefill and decode, the key-value cache and its arithmetic, and which questions may leave the device
- **Session 3 · Quantisation**: Quantisation formats, scales and calibration, for a classifier and a language model
- **Session 4 · Pruning or a smaller model**: Unstructured and structured pruning, a smaller architecture, and distillation as an idea

#### [day3#review-day-2] Where did the memory go?

*Review of Day 2*

**Question:** Waraqa's training peak at batch 256 is about 40 times its weights, gradients and optimiser state together. Which change cuts that peak most, at the same effective batch?

- A. Switching AdamW for SGD to drop the optimiser state
- B. Accumulating four micro-batches of 64
- C. Storing the weights in 16 bits
- D. Folding batch norm into the convolutions

### Session 1: Latency, throughput and batching

#### [day3#s1] Latency, throughput and batching

*Session 1*

- Measure latency and throughput across batch sizes
- Explain why batching raises throughput and latency together
- Choose a batch size for a latency budget

#### [day3#latency-vs-throughput] Latency is how long one request waits; throughput is how many finish per second

*Two kinds of work at Nama*

- **In the office: latency.** An agent photographs one leaf and waits. Budget: 15 ms at the 95th percentile.
- **At night: throughput.** The office re-scores the day's archive. Nobody waits for one leaf; the job must finish.

#### [day3#batch-sweep] Batching raises throughput until the cores are busy, and raises latency all the way

*Measured in Lab 3.1: Waraqa in PyTorch, four threads*

| Batch | Median ms per batch | ms per leaf | Leaves per s |
| --- | --- | --- | --- |
| 1 | 7.9 | 7.9 | 126 |
| 2 | 12.5 | 6.3 | 159 |
| 4 | 19.7 | 4.9 | 203 |
| 8 | 37.8 | 4.7 | 212 |
| 16 | 81.0 | 5.1 | 198 |
| 32 | 212.2 | 6.6 | 151 |
| 64 | 318.2 | 5.0 | 201 |

#### [day3#batch-wait] A batch does not exist until its requests arrive

*The hidden cost*

- **average wait = (b - 1) / (2 x rate).** plus the batch's own time. At 2 leaves per second a batch of 8 makes the average leaf wait about 1.75 s before computing starts.

- Office, 2 leaves per second: batch 1 is the only batch that meets the budget.
- Night archive, 200 per second: larger batches fill fast and serve more.
- Servers batch **dynamically**: up to b requests or t milliseconds, whichever comes first.

#### [day3#goodput] Count only the requests served within the objective

*Goodput against a service objective*

**500 x 50% = 250** leaves per second of goodput, for a server that answers 500 per second with half of them late

- Throughput counts everything; goodput counts what the user accepts.
- A larger batch can raise throughput and lower goodput.
- Load tests of a real service measure it: AE-205 does that under load.

#### [day3#checkpoint-1] Which batch size for the office?

*Checkpoint*

**Question:** Waraqa serves more leaves per second at batch 16 than at batch 1 (Lab 3.1). In the office, an agent sends one leaf about every two seconds. Which batch size should the office app use?

- A. 16, for the throughput
- B. 4, a compromise
- C. 1: requests are rare, so any larger batch adds seconds of waiting
- D. 64, the cheapest per leaf

#### [day3#lab-3-1] Batch size against latency

Lab 3.1 · 25 min

Serve Waraqa at batch sizes from 1 to 64, plot latency against throughput and add the time a request waits for its batch.

[D3_1_batching.ipynb](https://sdaiaacademy.github.io/ae-206/labs/D3_1_batching.ipynb)

- [ ] Time batch sizes from 1 to 64 (6 min)
- [ ] Throughput per batch size (4 min)
- [ ] Plot latency against throughput (5 min)
- [ ] Add the wait for the batch to fill (6 min)
- [ ] Pick the batch size for the farm office (4 min)
- [ ] ✱ Stretch: Simulate dynamic batching with a timeout and find its tail latency

### Session 2: Language model inference on a laptop

#### [day3#s2] Language model inference on a laptop

*Session 2*

- Measure time to first token and tokens per second
- Size a key-value cache from the model's shape
- Route a question by its data classification

#### [day3#meet-hasad] Hasad: a small open-weight model the course trained, shipped in the repository

*The language model on the laptop*

- A decoder-only transformer: 6 layers, width 128, 4 heads, a context of 512.
- 1.27 million parameters; one token is one character.
- Trained on Nama's synthetic advisory facts; it answers the farmer questions of the case.

- **Why so small.** Every lab runs on a laptop processor and nothing is downloaded. The arithmetic of prefill, decode and the cache is the same at any size.

#### [day3#prefill-decode] Reading the prompt is one big pass; writing the answer is one pass per token

*Prefill and decode*

1. **Prefill** all prompt tokens at once: many rows, compute-bound; ends with the first token
2. **Decode** one token per pass, reading every weight and the cache: memory-bound at batch 1

- **Time to first token.** What the farmer feels before anything appears. Grows with the prompt.
- **Tokens per second.** How fast the answer streams. Depends on bytes read per token.

#### [day3#hasad-measured] Time to first token grows with the context; decoding slows far less

*Measured in Lab 3.2: farm notes before the question*

| Notes (characters) | Prompt tokens | TTFT ms | Decode tokens per s |
| --- | --- | --- | --- |
| 64 | 136 | 13.9 | 233 |
| 256 | 328 | 30.3 | 220 |
| 448 | 432 | 45.1 | 163 |

#### [day3#kv-cache] The cache keeps each token's key and value in every layer, so decode never recomputes them

*The key-value cache*

- **bytes = 2 x layers x (heads x head size) x tokens x bytes per number.** Hasad: 2 x 6 x 128 x tokens x 4 = 6,144 bytes per token.

| Tokens | Estimate | Cache tensors |
| --- | --- | --- |
| 157 | 964.6 kB | 964.6 kB |
| 349 | 2,144.3 kB | 2,144.3 kB |
| 472 | 2,900.0 kB | 2,900.0 kB |

#### [day3#kv-scale] For a real assistant the cache is no longer small: it grows with every token and every conversation

*The same formula at a real model's shape*

| Model shape (16-bit numbers) | Tokens | Cache per conversation |
| --- | --- | --- |
| 16 layers, 8 key-value heads of 64 | 4,096 | 134 MB |
| 32 layers, 8 key-value heads of 128 | 8,000 | 1.05 GB |

Serving systems page the cache into blocks so that conversations share memory without fragmenting it (PagedAttention).

#### [day3#route-by-classification] Nama routes farmer questions by data classification before by speed

*Which questions may leave the laptop*

- Nama applies the four levels of the **NDMO Data Classification Policy**: Top Secret, Secret, Restricted, Public.
- The policy lists personally identifiable information (names, addresses, phone numbers, identity numbers) among its examples of Restricted data.
- Reham's rule: Restricted questions are answered **on the device only**; Public ones may go to a hosted model.

- Restricted: a question with a mobile number or an identity number
- Public: "How often do I water tomatoes in Qassim in summer?"

#### [day3#routing-result] A router must never leak; keeping a Public question local only costs speed

*Measured in Lab 3.2 on the 2,000 questions*

- **307** Restricted questions by the answer key
- **0** Restricted questions the rule would send out
- **0** Public questions kept on the device

> **Patterns catch the formats they know.** A phone number written in words passes. A router is a control to test, not a guarantee.

#### [day3#checkpoint-2] What grows with the context?

*Checkpoint*

**Question:** Farm agents start pasting a week of farm notes (2,000 characters) before each question. On a model with a context of 4,096 tokens, what changes most?

- A. The memory of the model's weights
- B. Tokens per second, which fall in proportion to the context
- C. The number of tokens in the answer
- D. The time to first token and the size of the key-value cache

#### [day3#lab-3-2] Hasad on a laptop

Lab 3.2 · 25 min

Measure Hasad's time to first token and tokens per second at three context lengths, size its key-value cache, and route farmer questions to the device or a hosted model by data classification.

[D3_2_lm_inference.ipynb](https://sdaiaacademy.github.io/ae-206/labs/D3_2_lm_inference.ipynb)

- [ ] Time to first token at three context lengths (6 min)
- [ ] Tokens per second while decoding (4 min)
- [ ] Size the key-value cache by hand (5 min)
- [ ] Check the estimate against the cache tensors (4 min)
- [ ] Route questions by classification (6 min)
- [ ] ✱ Stretch: Decode without the cache and measure what it costs

### Session 3: Quantisation

#### [day3#s3] Quantisation

*Session 3*

- Quantise a classifier to 8 bits with calibration data
- Compare 8-bit and 4-bit language model files on size, speed and answers
- Re-measure quality by group after every quantisation

#### [day3#quantisation-idea] Store numbers as small integers with a scale, and map them back when needed

*Quantisation*

- **real value = scale x (integer - zero point).** 8 bits give 256 levels; 4 bits give 16. The scale is chosen per tensor, per channel or per block of weights.

- **Smaller files**: a quarter (8-bit) or an eighth (4-bit) of float32.
- **Faster** only where kernels compute on integers or blocks.
- **Lossy**: quality is measured again, by group.

#### [day3#static-dynamic] Static quantisation calibrates activation ranges on real data; dynamic computes them on the fly

*For a classifier*

|  | Weights | Activations | Needs |
| --- | --- | --- | --- |
| Static | 8-bit, ahead of time | 8-bit, ranges from calibration | a few hundred representative inputs |
| Dynamic | 8-bit, ahead of time | ranges computed at run time | nothing; less speed on convolutions |
| Quantisation-aware training | training sees the rounding and adapts to it | a training run |  |

#### [day3#calibration-reader] The calibration set must look like the field: all four crops, faint and clear cases

*Lab 3.3, step 1*

```python
# file: Lab 3.3
class Reader(CalibrationDataReader):
    def get_next(self):
        if self.i >= len(self.x):
            return None
        out = {"image": U.to_input(self.x[self.i:self.i + 16]).numpy()}
        self.i += 16
        return out

quantize_static("waraqa_pre.onnx", "waraqa_int8.onnx", Reader(x_cal),
                quant_format=QuantFormat.QDQ, per_channel=True)
```

#### [day3#waraqa-int8] The 8-bit file is a quarter of the size and slightly faster, with the same accuracy overall

*Measured in Lab 3.3: ONNX Runtime, batch 1, four threads*

| File | MB | Median ms | p95 ms | Accuracy | Date palm |
| --- | --- | --- | --- | --- | --- |
| Waraqa 32-bit | 1.16 | 4.64 | 9.84 | 99.1% | 92.7% |
| Waraqa 8-bit | 0.31 | 3.98 | 7.86 | 99.1% | 93.8% |

#### [day3#hasad-quantised] Hasad in 8 and 4 bits: smaller files, and quality checked by language variety

*Measured in Lab 3.3 on 200 farmer questions*

| File | MB | tokens/s | Right | MSA | Gulf | Mixed |
| --- | --- | --- | --- | --- | --- | --- |
| 32-bit | 5.08 | 159 | 100% | 100% | 100% | 100% |
| 8-bit, expanded once | 1.45 | 187 | 100% | 100% | 100% | 100% |
| 4-bit, expanded once | 0.86 | 221 | 100% | 100% | 100% | 100% |
| 4-bit, expanded per call | 0.86 | 51 | 100% | 100% | 100% | 100% |

#### [day3#quant-speed] A smaller file is not a faster model: the runtime decides

*Read the table twice*

- **Expand once.** Small download and disk; same memory and speed as float32.
- **Expand per call.** Small memory; slower in a general framework that expands before multiplying.
- **Compute on blocks.** Small memory and faster decoding, because fewer bytes are read per token. Needs a runtime built for it.

#### [day3#checkpoint-3] Is the 8-bit file ready to ship?

*Checkpoint*

**Question:** Another team's 8-bit leaf classifier is a quarter of the size, 30% faster, and 0.6 points less accurate overall. Date palm leaves, one in ten of their held-out set, lost 3 points. What do you do next?

- A. Put intervals on the overall and per-crop differences, and compare them with the floor per crop
- B. Ship it: 0.6 points is within a 1-point floor
- C. Reject it: any crop that loses is unacceptable
- D. Recalibrate with twice as many leaves and ship that

#### [day3#lab-3-3] Quantise Waraqa and Hasad

Lab 3.3 · 25 min

Quantise Waraqa to 8 bits, load the supplied 8-bit and 4-bit files of Hasad, and measure size, speed and quality loss on the fixed question set.

[D3_3_quantisation.ipynb](https://sdaiaacademy.github.io/ae-206/labs/D3_3_quantisation.ipynb)

- [ ] Calibrate and quantise Waraqa to 8 bits (6 min)
- [ ] Size, latency and accuracy of both Waraqa files (5 min)
- [ ] Load Hasad's 8-bit and 4-bit files (4 min)
- [ ] Size, speed and right answers by language variety (6 min)
- [ ] One comparison table (4 min)
- [ ] ✱ Stretch: Quantise Waraqa dynamically instead and compare with the static file

### Session 4: Pruning or a smaller model

#### [day3#s4] Pruning or a smaller model

*Session 4*

- Prune whole channels by an importance score
- Recover accuracy with brief fine-tuning
- Compare a pruned model with a smaller one trained from scratch

#### [day3#pruning-kinds] Zeroing weights saves nothing on dense kernels; removing channels does

*Pruning*

- **Unstructured.** Set the smallest weights to zero anywhere. The matrices keep their shape: a dense kernel multiplies zeros as fast as other numbers.
- **Structured.** Remove whole filters (channels). Every matrix gets smaller, so every layer does less work.

#### [day3#channel-scores] Score each filter by its L1 norm, keep the top half, copy the weights

*Lab 3.4*

```python
# file: Lab 3.4
scores = U.channel_scores(base)           # one tensor per convolution
keep = [torch.sort(torch.topk(s, len(s) // 2).indices).values
        for s in scores]
pruned = U.prune_channels(base, keep)     # a smaller Waraqa, weights copied
U.train_steps(pruned, x_tr, y_tr, steps, lr=5e-4)   # a short fine-tune
```

#### [day3#pruning-results] Pruning breaks the model until it is fine-tuned; Waraqa-S, trained small, keeps the most accuracy

*Measured in Lab 3.4: batch 1, PyTorch, four threads*

| Model | Parameters | MB | Median ms | Accuracy |
| --- | --- | --- | --- | --- |
| Waraqa | 288,746 | 1.17 | 12.8 | 99.1% |
| Pruned, no fine-tune | 72,954 | 0.31 | 7.5 | 8.4% |
| Pruned, 250 fine-tune steps | 72,954 | 0.31 | 7.8 | 95.6% |
| Waraqa-S, trained small | 72,954 | 0.31 | 8.4 | 97.9% |
| Pruned, one epoch (supplied) | 72,954 | 0.31 | 7.7 | 95.0% |

#### [day3#distillation-concept] Distillation trains the small model to imitate the large one's probabilities

*A third route, as a concept*

- The teacher's output says not only "rust" but "rust, a little like leaf spot".
- The student learns from those soft targets as well as the labels.
- Often beats the same small model trained on labels alone.

- **When to choose which.** Prune when you own a trained large model and want a cheap result; shrink or distil when you train anyway and want the simplest artefact.

#### [day3#checkpoint-4] Why is the masked model not faster?

*Checkpoint*

**Question:** You set half of Waraqa's convolution weights to zero by magnitude. Accuracy barely changes and latency does not change at all. Why?

- A. The pruning did not take effect
- B. The dense kernels still multiply every weight, zero or not
- C. Batch norm undoes the pruning at inference
- D. Only the classifier head determines latency

#### [day3#lab-3-4] Prune or shrink

Lab 3.4 · 25 min

Prune half of Waraqa's channels, fine-tune briefly and compare with Waraqa-S, the smaller architecture trained from scratch.

[D3_4_pruning.ipynb](https://sdaiaacademy.github.io/ae-206/labs/D3_4_pruning.ipynb)

- [ ] Score channels by their L1 norm (4 min)
- [ ] Prune half and measure accuracy (5 min)
- [ ] Fine-tune briefly (6 min)
- [ ] Compare with Waraqa-S (5 min)
- [ ] Size, latency and accuracy table (5 min)
- [ ] ✱ Stretch: Mask half the weights without removing channels: does it run faster?

#### [day3#day-3-recap] Three things to keep from Day 3

*Day 3 recap*

- **Batching is a trade.** Throughput for latency, and only when requests arrive fast enough.
- **Cache arithmetic.** 2 x layers x width x tokens x bytes, per conversation, on top of the weights.
- **Compression is lossy.** Smaller is not faster without the right kernels, and quality moves unevenly.

#### [day3#tomorrow-export] Export and benchmark

*Tomorrow · Day 4*

Tomorrow you export Waraqa to a portable runtime and prove it matches, write a benchmark someone else can re-run within 10%, put intervals on the quality of every compressed variant by crop, and choose the one that meets Nama's budget. Then you choose your capstone model.

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

## Capstone: Capstone: an optimisation study and its defence

Eight minutes of results, a live re-run, four minutes of questions

#### [capstone#title] Capstone: an optimisation study and its defence

*SDAIA Academy · Machine Learning Systems on One Machine*

Capstone: Eight minutes of results, a live re-run, four minutes of questions

AI Engineering Track · 4 teaching days + capstone day · 8 min talk and demo · 4 min Q&A · AE-206

#### [capstone#agenda] How the presentations run

*Capstone*

- **Expectations and final prep**: the rubric, a last rehearsal, then the first presentations
- **Presentations**: each one timed, with questions after it
- **Review and closing**: peer feedback, the trainer's review, what next

#### [capstone#rubric] The rubric

*How presentations are scored*

| Item | Value |
| --- | --- |
| Baseline profile | 20% |
| Changes ordered by measured impact | 25% |
| Quality re-evaluation | 20% |
| Benchmark method | 20% |
| Reproducibility | 15% |

> **Show results from at least two measured runs.** A claim without a number is a plan, not a result.

#### [capstone#format] 8 minutes to present, 4 to answer

*Format*

- **1 · Model, device, budget.** What you optimised, on which machine, against which limits and quality floor.
- **2 · The log.** The baseline profile, then each change by measured impact, with quality checked.
- **3 · The live re-run.** Run the final benchmark now, on this laptop, and compare with your README.

> **Peer feedback.** After each talk, everyone writes one strength, one question and one suggestion.

#### [capstone#predict-then-measure] The assessor proposes a change; you predict its effect, then measure it

*The four minutes of questions*

1. **1 · Propose** "Use one thread", "batch 8", "the 8-bit file", "a longer prompt"
2. **2 · Predict** direction and rough size, with the reason: which limit binds
3. **3 · Measure** re-run the benchmark with the change, live
4. **4 · Explain** why the number matched the prediction, or why not

#### [capstone#common-gaps] Five gaps to close before you present

*What reviewers flag most often*

- A speed-up without the **machine, threads and batch** it was measured with.
- A median from one run, without **repetitions or an interval**.
- Quality checked overall but **not per group**.
- Changes listed in the order they were tried, not by **measured impact**.
- A README whose commands do not **reproduce** the table from a fresh clone.

#### [capstone#running-order] Who presents when

*Running order*

| # | Presenter | Project |
| --- | --- | --- |
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |
| 4 |  |  |
| 5 |  |  |
| 6 |  |  |
| 7 |  |  |
| 8 |  |  |

#### [capstone#final-prep] Final prep · 20 minutes

*Before the first presentation*

Rehearse once, record a backup, try the demo on this screen.

#### [capstone#now-presenting] Talk and demo

*Now presenting*

##### Questions

#### [capstone#review] What the presentations showed

*Trainer review*

- **Strongest patterns.** What several projects got right.
- **Common gaps.** What most projects would fix next.
- **One surprise.** Something nobody expected.

#### [capstone#what-next] Keep going

*After the course*

- **Your project stays yours:** keep it reproducible and add the next improvement you named today
- **The follow-up task:** re-run your benchmark on a second device and write a short comparison note: what moved, and why
- **Next Academy courses:** AE-205 for serving and monitoring a model under load; AE-401 for inference performance and cost at scale; AE-310 for fine-tuning, distillation and small models in practice

#### [capstone#thanks] Thank you

*SDAIA Academy*

Machine Learning Systems on One Machine
