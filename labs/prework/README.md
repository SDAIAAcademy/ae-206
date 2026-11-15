# Pre-work: four workloads on one machine

About two hours before Day 1: a reading, then four estimates by hand. Bring the estimates to
Session 1, where they are discussed; the supervised entry quiz asks for estimates of the same
kind. **AI-off:** the first attempt at every estimate is yours, with a calculator or a
spreadsheet and nothing else.

## 1. Reading (about 90 minutes)

- The roofline model: Williams, Waterman and Patterson, *Roofline: an insightful visual
  performance model* (UC Berkeley technical report EECS-2008-134),
  https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html. Read for three ideas:
  peak compute, memory bandwidth, and the arithmetic intensity at which the two roofs meet.
- The open *Machine Learning Systems* textbook, https://mlsysbook.ai/: the parts on hardware
  and on benchmarking that your trainer assigns. Its analytical model, MLSys·im, does the same
  arithmetic as the exercise below; a spreadsheet does too.

## 2. The machine

Nama Agritech's farm-office laptop, with round figures for the exercise (Lab 1.1 measures the
real ones on your own laptop):

| | |
|---|---|
| Peak compute | 100 GFLOP/s |
| Memory bandwidth | 20 GB/s |
| Memory capacity | 8 GB, of which 6 GB is free for the model |

## 3. The exercise (about 30 minutes)

For each workload, estimate the FLOPs and the bytes moved (or held), say which limit binds
(compute, memory bandwidth or memory capacity), and give the number the question asks for.
An estimate within 20% of the reference counts as matching.

Three rules are enough:

- attainable rate = the lower of peak compute and (arithmetic intensity x bandwidth), where
  arithmetic intensity = FLOPs / bytes moved; the ridge point is peak / bandwidth;
- a fully connected layer does one multiply and one add per weight per row of the batch, and
  moves its input, its weights and its output once;
- training in float32 with AdamW holds 16 bytes per parameter (weight, gradient and two
  optimiser numbers) before any activation.

| # | Workload | Asked |
|---|---|---|
| 1 | One fully connected layer, 4,096 inputs to 4,096 outputs, float32, **batch 1** | intensity, binding limit, least time per call |
| 2 | The same layer at **batch 256** | intensity, binding limit, least time per call |
| 3 | A language model of 1 billion parameters stored in 16 bits decodes for one user: each new token reads every weight once and does about 2 FLOPs per weight | binding limit, most tokens per second |
| 4 | Training a convolutional network of 50 million parameters in float32 with AdamW; the forward pass keeps 40 MB of activations per image for the backward pass | memory at batch 64; binding limit at batch 256; largest batch that fits in the 6 GB |

<details><summary>Reference estimates (open after your own attempt)</summary>

The ridge point of this machine is 100 / 20 = 5 FLOPs per byte.

1. FLOPs: 2 x 4,096 x 4,096 = 33.6 million. Bytes: 4 x (4,096 + 4,096 x 4,096 + 4,096), about
   67 MB. Intensity: 0.5 FLOPs per byte, below the ridge: **memory bandwidth** binds. Least
   time: 67 MB / 20 GB/s, about **3.4 ms** (compute alone would allow 0.34 ms).
2. FLOPs: 256 x 33.6 million = 8.6 GFLOP. Bytes: 4 x (2 x 256 x 4,096 + 4,096 x 4,096), about
   75 MB: the weights are read once for all 256 rows. Intensity: about 114 FLOPs per byte,
   above the ridge: **compute** binds. Least time: 8.6 / 100, about **86 ms** per batch, or
   0.34 ms per row, ten times less per row than at batch 1.
3. Bytes: 1 billion x 2 = 2 GB per token. FLOPs: 2 GFLOP per token. Intensity: 1 FLOP per
   byte, below the ridge: **memory bandwidth** binds. Least time per token: 2 GB / 20 GB/s =
   100 ms, so at most **10 tokens per second** (compute alone would allow 50). The 2 GB of
   weights fit in the 6 GB, so capacity does not bind.
4. Fixed part: 50 million x 16 bytes = 0.8 GB. Batch 64: 0.8 + 64 x 0.04 = about **3.4 GB**,
   which fits. Batch 256: 0.8 + 256 x 0.04 = about 11 GB, more than the machine holds:
   **memory capacity** binds, before speed is even a question. Largest batch in 6 GB:
   (6 - 0.8) / 0.04 = **130 images**.

</details>

## After the course: the follow-up task

Re-run your capstone benchmark on a second device (another laptop, or a free Colab runtime)
with the same input file and settings, and write a short comparison note: both machines, both
medians and 95th percentiles with their intervals, what moved, and why. The trainer tracks the
note.
