---
course: "Machine Learning Systems on One Machine"
code: "AE-206"
version: "2027.1.0"
deck: "day2"
day: 2
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-206/slides/day2.html"
---

# Machine Learning Systems on One Machine

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
