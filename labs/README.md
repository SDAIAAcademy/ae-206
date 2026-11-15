# Machine Learning Systems on One Machine: labs

## Quick start

- Google Colab: open a notebook from the course page; its first cell clones this repository.
- Locally: clone the repository, then open the notebooks in `labs/` with Jupyter.

| Unit | Session | Notebook |
|---|---|---|
| Day 1 | Session 1 · Compute, memory and bandwidth | [`D1_1_roofline.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D1_1_roofline.ipynb) |
| Day 1 | Session 2 · Timers, profilers and noise | [`D1_2_timing_noise.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D1_2_timing_noise.ipynb) |
| Day 1 | Session 3 · Profiling a training step | [`D1_3_profile_step.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D1_3_profile_step.ipynb) |
| Day 1 | Session 4 · Fixing an input-bound pipeline | [`D1_4_input_pipeline.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D1_4_input_pipeline.ipynb) |
| Day 2 | Session 1 · Where memory goes | [`D2_1_memory_budget.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D2_1_memory_budget.ipynb) |
| Day 2 | Session 2 · Mixed precision | [`D2_2_mixed_precision.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D2_2_mixed_precision.ipynb) |
| Day 2 | Session 3 · Accumulation and checkpointing | [`D2_3_accumulate_checkpoint.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D2_3_accumulate_checkpoint.ipynb) |
| Day 2 | Session 4 · Compilation and fusion | [`D2_4_compile_fusion.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D2_4_compile_fusion.ipynb) |
| Day 3 | Session 1 · Latency, throughput and batching | [`D3_1_batching.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D3_1_batching.ipynb) |
| Day 3 | Session 2 · Language model inference on a laptop | [`D3_2_lm_inference.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D3_2_lm_inference.ipynb) |
| Day 3 | Session 3 · Quantisation | [`D3_3_quantisation.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D3_3_quantisation.ipynb) |
| Day 3 | Session 4 · Pruning or a smaller model | [`D3_4_pruning.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D3_4_pruning.ipynb) |
| Day 4 | Session 1 · Export to a portable runtime | [`D4_1_export_parity.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D4_1_export_parity.ipynb) |
| Day 4 | Session 2 · A benchmark you can trust | [`D4_2_benchmark.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D4_2_benchmark.ipynb) |
| Day 4 | Session 3 · Quality must survive | [`D4_3_quality_by_crop.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D4_3_quality_by_crop.ipynb) |
| Day 4 | Session 4 · Budgets and trade-offs | [`D4_4_budget_choice.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D4_4_budget_choice.ipynb) |

The notebooks are generated from the percent-format sources in `labs/_src/` by `labs/_tools/py2nb.py` (`uv run python _tools/py2nb.py` from `labs/`), which also writes the solutions to `labs/solutions/`. Edit the sources, not the notebooks.
