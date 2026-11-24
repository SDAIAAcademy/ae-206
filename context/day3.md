---
course: "Machine Learning Systems on One Machine"
code: "AE-206"
version: "2027.2.0"
deck: "day3"
day: 3
language: "en"
view: "trainee"
source: "https://sdaiaacademy.github.io/ae-206/slides/day3.html"
---

# Machine Learning Systems on One Machine

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
