---
course: "Machine Learning Systems on One Machine"
code: "AE-206"
version: "2027.2.0"
deck: "day1"
day: 1
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-206/slides/day1.html"
---

# Machine Learning Systems on One Machine

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
