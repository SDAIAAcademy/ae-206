# Machine Learning Systems on One Machine: labs

## Quick start

- Google Colab: open a notebook from the course page; its first cell clones this repository.
- Locally: clone the repository, then open the notebooks in `labs/` with Jupyter.

| Unit | Session | Notebook |
|---|---|---|
| Day 1 | Session 1 · Compute, memory and bandwidth | [`D1_1_estimate_arithmetic_intensity.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D1_1_estimate_arithmetic_intensity.ipynb) |
| Day 1 | Session 2 · Timers, profilers and noise | [`D1_2_time_one_inference.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D1_2_time_one_inference.ipynb) |
| Day 1 | Session 3 · Profiling a training step | [`D1_3_profile_one_training.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D1_3_profile_one_training.ipynb) |
| Day 1 | Session 4 · Fixing an input-bound pipeline | [`D1_4_add_parallel_loading.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D1_4_add_parallel_loading.ipynb) |
| Day 2 | Session 1 · Where memory goes | [`D2_1_estimate_measure_peak.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D2_1_estimate_measure_peak.ipynb) |
| Day 2 | Session 2 · Mixed precision | [`D2_2_train_without_reduced.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D2_2_train_without_reduced.ipynb) |
| Day 2 | Session 3 · Accumulation and checkpointing | [`D2_3_fit_larger_effective.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D2_3_fit_larger_effective.ipynb) |
| Day 2 | Session 4 · Compilation and fusion | [`D2_4_compile_model_measure.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D2_4_compile_model_measure.ipynb) |
| Day 3 | Session 1 · Latency, throughput and batching | [`D3_1_serve_classifier_batch.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D3_1_serve_classifier_batch.ipynb) |
| Day 3 | Session 2 · Language model inference on a laptop | [`D3_2_measure_time_first.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D3_2_measure_time_first.ipynb) |
| Day 3 | Session 3 · Quantisation | [`D3_3_quantise_classifier_8.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D3_3_quantise_classifier_8.ipynb) |
| Day 3 | Session 4 · Pruning or a smaller model | [`D3_4_prune_classifier_fine.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D3_4_prune_classifier_fine.ipynb) |
| Day 4 | Session 1 · Export to a portable runtime | [`D4_1_export_classifier_exchange.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D4_1_export_classifier_exchange.ipynb) |
| Day 4 | Session 2 · A benchmark you can trust | [`D4_2_write_benchmark_warm.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D4_2_write_benchmark_warm.ipynb) |
| Day 4 | Session 3 · Quality must survive | [`D4_3_evaluate_each_compressed.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D4_3_evaluate_each_compressed.ipynb) |
| Day 4 | Session 4 · Budgets and trade-offs | [`D4_4_from_supplied_results.ipynb`](https://colab.research.google.com/github/SDAIAAcademy/ae-206/blob/main/labs/D4_4_from_supplied_results.ipynb) |

The notebooks are generated from the percent-format sources in `labs/_src/` by `labs/_tools/py2nb.py` (`uv run python _tools/py2nb.py` from `labs/`), which also writes the solutions to `labs/solutions/`. Edit the sources, not the notebooks.
