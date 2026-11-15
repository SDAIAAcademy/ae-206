# Machine Learning Systems on One Machine: labs

Hands-on notebooks for the SDAIA Academy course **AE-206 Machine Learning Systems on One Machine (أنظمة تعلم الآلة على جهاز واحد)**. Every lab works on **Nama Agritech (نماء للتقنيات الزراعية)**, a fictional farm-advisory company whose agents run two models on old farm-office laptops: **Waraqa (ورقة)**, a leaf-disease classifier, and **Hasad (حصاد)**, a small language model that answers farmers' questions. Each lab starts from a supplied checkpoint or model file, so none depends on another lab of the same day.

Everything runs on a laptop processor. A free Colab runtime works too; a Colab GPU is optional (Lab 2.2's stretch and Lab 2.4's compiler are where it differs). **Nothing downloads a model**: both models were trained by the course and are in `data/models/`.

## Quick start

**Google Colab**
1. Open a notebook from the [course page](https://sdaiaacademy.github.io/ae-206/), or directly: `https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/<notebook>.ipynb`.
2. Run the first cell. It clones this repository and installs `requirements-colab.txt` (Colab already has PyTorch).

**Local**
```bash
git clone https://github.com/SDAIAAcademy/ae-206
cd ae-206/labs
uv sync                      # creates the environment from pyproject.toml and uv.lock (PyTorch for processors)
uv run jupyter lab
```
Run `00_setup_check.ipynb` once before Day 1. It checks every library, builds the synthetic data (about a minute) and says whether this machine can compile models (Lab 2.4).

**Quick mode.** Set `NAMA_QUICK=1` before starting Jupyter for smaller runs (fewer leaves, steps and repetitions); every solution finishes in a few minutes on a laptop.

**Windows and long folder paths.** If an import fails with "The filename or extension is too long", keep the environment in a short folder: set `UV_PROJECT_ENVIRONMENT` to a short path (for example a folder at the root of your drive) before `uv sync`. Run the labs as notebooks: Lab 1.4 starts DataLoader worker processes, which on Windows and macOS need the notebook (or an `if __name__ == "__main__":` guard in a script).

## Notebooks

| Day | Session | Notebook |
|---|---|---|
| 0 | Setup | [`00_setup_check.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/00_setup_check.ipynb) |
| 1 | Compute, memory and bandwidth | [`D1_1_roofline.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D1_1_roofline.ipynb) |
| 1 | Timers, profilers and noise | [`D1_2_timing_noise.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D1_2_timing_noise.ipynb) |
| 1 | Profiling a training step | [`D1_3_profile_step.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D1_3_profile_step.ipynb) |
| 1 | Fixing an input-bound pipeline | [`D1_4_input_pipeline.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D1_4_input_pipeline.ipynb) |
| 2 | Where memory goes | [`D2_1_memory_budget.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D2_1_memory_budget.ipynb) |
| 2 | Mixed precision | [`D2_2_mixed_precision.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D2_2_mixed_precision.ipynb) |
| 2 | Accumulation and checkpointing | [`D2_3_accumulate_checkpoint.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D2_3_accumulate_checkpoint.ipynb) |
| 2 | Compilation and fusion | [`D2_4_compile_fusion.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D2_4_compile_fusion.ipynb) |
| 3 | Latency, throughput and batching | [`D3_1_batching.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D3_1_batching.ipynb) |
| 3 | Language model inference on a laptop | [`D3_2_lm_inference.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D3_2_lm_inference.ipynb) |
| 3 | Quantisation | [`D3_3_quantisation.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D3_3_quantisation.ipynb) |
| 3 | Pruning or a smaller model | [`D3_4_pruning.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D3_4_pruning.ipynb) |
| 4 | Export to a portable runtime | [`D4_1_export_parity.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D4_1_export_parity.ipynb) |
| 4 | A benchmark you can trust | [`D4_2_benchmark.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D4_2_benchmark.ipynb) |
| 4 | Quality must survive | [`D4_3_quality_by_crop.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D4_3_quality_by_crop.ipynb) |
| 4 | Budgets and trade-offs | [`D4_4_budget_choice.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D4_4_budget_choice.ipynb) |

Exercises are marked **✏️ Exercise** and raise `NotImplementedError` until you fill them in; the cell after each one has a safety net, so the rest of the notebook still runs. Complete versions are in `solutions/`. Cells marked **✱ Stretch** are optional. Timings depend on your machine and on what else it is running: compare your numbers with your own earlier numbers, and write down the machine with every result.

The notebooks are generated from the percent-format sources in `_src/` by `_tools/py2nb.py` (`uv run python _tools/py2nb.py` from `labs/`), which also writes the solutions. Edit the sources, not the notebooks. Lab outputs go to `artifacts/` and the generated data to `cache/`; git ignores both.

## The case study: Nama Agritech

All data is synthetic and deterministic (seed 206); no real person, farm or company is described.

| Data | Where | Contents |
|---|---|---|
| Leaf images | built by `prepare_data.py` into `cache/` | 20,000 leaves in ten classes (healthy and nine diseases) and four crops (tomato 35%, cucumber 30%, citrus 25%, date palm 10%); 16,000 training, 2,000 validation, 2,000 held out |
| Field export | built on first use by Labs 1.3 and 1.4 | the same leaves as 224-pixel PNG files, the format the field app uploads |
| Farmer questions | built by `prepare_data.py` | 2,000 Arabic questions in Modern Standard Arabic, Gulf dialect and mixed Arabic and English, with answer, key fact, variety and a data classification (Public or Restricted) |
| Models | `data/models/` | `waraqa.pt` and `.onnx` (the classifier), `waraqa_warm.pt` (after one epoch), `waraqa_int8.onnx`, `waraqa_pruned.*`, `waraqa_s.*` (half the channels, trained from scratch), `hasad_32.pt`, `hasad_8.pt`, `hasad_4.pt` |
| Supplied results | `data/supplied/` | Nama's budget and the measured results table of Lab 4.4 |
| Reference values | `data/reference/` | the course's results that later labs start from |

Waraqa is a six-layer convolutional network for 64-pixel leaves. Hasad is a character-level transformer (six layers, width 128, a context of 512 characters) trained by `_tools/train_models.py` on the synthetic advisory facts; its 8-bit and 4-bit files store the linear layers in groups of 32 weights with one 16-bit scale per group.

## Standalone programs

| Script | What it is | Run |
|---|---|---|
| `lina_train.py` | Lina's slow training script (Labs 1.3 and 1.4 import it) | `uv run lina_train.py --steps 20` |
| `bench.py` | The course's timing script: one ONNX model, warm-up, repetitions, fixed inputs, percentiles with bootstrap intervals, the machine, a JSON report | `uv run bench.py data/models/waraqa.onnx --inputs artifacts/mix.npy --out artifacts/b.json` |
| `prepare_data.py` | The data generator | `uv run python prepare_data.py --export` |
| `capstone/README_template.md` | The capstone repository's README, section by section | |

## Helpers: `nama.py`

| Helper | What it does |
|---|---|
| `leaves(split, n)`, `to_input(x)` | leaf images as uint8 arrays with class and crop; the model's input tensor |
| `load_model(name)`, `accuracy(model, x, y)`, `predict(...)` | a supplied Waraqa checkpoint in evaluation mode, and its quality |
| `load_hasad(bits, mode)`, `prompt_for(q, notes)`, `grade(answer, key)` | Hasad from its 32-, 8- or 4-bit file; its prompt; whether an answer has the key fact |
| `time_ms(fn, n, warmup)`, `summarise(ms)`, `bootstrap_ci(values, stat)` | timing with warm-up; median and percentiles; an interval |
| `peak_rss_mb(fn)`, `saved_activation_mb(model, x, y)` | peak memory of a call (sampled resident memory, or the GPU's counter); the bytes autograd keeps |
| `train_steps(model, x, y, steps, ...)` | training steps with autocast, accumulation or checkpointing; seconds per step |
| `export_onnx(model, path)`, `onnx_session(path, threads, level)`, `onnx_predict(...)` | export with a dynamic batch axis; an ONNX Runtime session |
| `channel_scores(model)`, `prune_channels(model, keep)` | L1 scores of filters; a smaller Waraqa with the kept channels |
| `machine()` | the processor, threads, memory and versions a report must name |
