# Changes

## 2027.2.0 (2026-11-24)

- Added: `fp16-underflow-signs` and `loss-scaling` (day 2, session 2): reading float16 gradient underflow from a measured table and fixing it with a loss scale, with the figures computed by `labs/_tools/fp16_underflow.py`.
- Added: a stretch in Lab 2.2 that measures the gradients float16 loses on Waraqa and picks a loss scale, on any machine.
- Added: a practice question on float16 underflow (day 2, session 2).
- Added: the PyTorch AMP examples page to the sources.
- Tightened: `hardware-support` and the notes of `claims-and-hardware` (day 2).
