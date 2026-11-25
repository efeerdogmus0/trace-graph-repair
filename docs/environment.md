# Environment

Observed on 2026-11-09 / 2026-11-10 on the development machine. No secrets recorded.

| Item | Observed |
|---|---|
| OS / architecture | macOS 15.6 (24G84), arm64 |
| CPU / RAM | Apple M1, 8 cores, 16 GiB |
| Free disk | about 39 GiB of 228 GiB at project start |
| NVIDIA GPU / CUDA | none (`nvidia-smi` not present) |
| Python | 3.9.6 (system, `/usr/bin/python3`); project uses `.venv` |
| Git | 2.39.5 (Apple Git-154) |
| Time zone | UTC+03; logs and state files use UTC |
| Python env managers | venv only; no conda, no uv |

Installed in `.venv` (exact versions in `requirements-lock.txt`): pillow 11.3.0, PyYAML
6.0.3, pytest 8.4.2, networkx 3.2.1, numpy 2.0.2, matplotlib 3.9.4. No torch, no
transformers, nothing from the Hugging Face Hub has been downloaded.

Fonts: DejaVu TTF files copied from matplotlib's bundled data into `assets/fonts/`.

## Access (state only, as observed 2026-11-09/10; current state below the table)

| Service | State |
|---|---|
| Hugging Face Hub (read, anonymous) | model metadata reachable through the public API |
| Hugging Face account / token | not checked |
| GitHub | not checked for this project; repository is local only |
| Colab | not opened; needs the owner's Google sign-in |
| Kaggle | not opened; GPU documentation page could not be loaded during the source check |
| Free GPU quota | unknown |

Current state (2026-11-22): Colab was used through the owner's signed-in browser for the early pilot runs (2026-11-10/12);
from 2026-11-12 all GPU work ran as private Kaggle script kernels through the Kaggle CLI (`RUNBOOK.md`, "Kaggle"). Free
weekly Kaggle GPU quota, tracked in `runs/budget.jsonl` / `reports/budget.md`. GitHub and Hugging Face: nothing published.

## Measured on this machine

| Operation | Time |
|---|---|
| Build smoke dataset (100 root graphs, 427 images, 988 samples) | about 21 s |
| Validate smoke dataset (34,922 checks incl. pixel read-back) | about 4 s |
| Full test suite (43 tests) | about 16 s |
| Build pilot dataset (712 root graphs, 3,076 images, 7,104 samples) | 124 s |
| Validate pilot dataset (249,139 checks) | 27 s |

Image files average about 44 kB each (512 x 512 PNG).

## GPU environment (Kaggle), added 2026-11-16

The tables above describe the laptop at project start (2026-11-09/10) and are kept as observed then; training and
evaluation ran on Kaggle script kernels, not on this machine.

| Item | Observed | Source |
|---|---|---|
| GPUs | 2 x Tesla T4, 15360 MiB each; the kernel uses GPU 0 only | `nvidia-smi` line in `runs/kaggle/<job>/trace-<job>.log` (e.g. job22) |
| peft / transformers | 0.21.2 / 5.22.2 (pinned by the kernel) | second line of `<job>.log` |
| Fonts | six DejaVu files, sha256 equal to the laptop's | `fonts ok 6` |
| Dataset | content and image digests equal to the tracked manifest | `content digest identical` |
| torch, CUDA, cuDNN, Python, Kaggle image | **not recorded** for kernels built before 2026-11-16 (all dev kernels; jobF01-03, jobF09-12) | no version line was printed |

Final kernels built from 2026-11-16 on (jobF04-08, jobF13) print one `ENV-SUMMARY` line with these versions into the
kernel log; `python scripts/kaggle/kernel_env.py <jobs>` reads it (that line only) and exits 1 if the recorded
environments differ. The earlier kernels cannot be completed after the fact; the paper may quote torch/CUDA versions
only for kernels that recorded them.

Recorded result (2026-11-22, `reports/final/kernel_env.json`, written by
`scripts/kaggle/kernel_env.py jobF01 ... jobF13 jobR1 --out ...`): the six kernels that printed the line (jobF04-08, jobF13)
report identical environments, no differing field:

| Item | Value |
|---|---|
| Kaggle image | `gcr.io/kaggle-gpu-images/python@sha256:fc720c15...` (full digest in the JSON) |
| Python / platform | 3.13.16 / Linux 6.18.55+ x86_64, glibc 2.39 |
| torch / CUDA / cuDNN | 2.11.0+cu130 / 13.0 / 92700 |
| transformers / peft / accelerate | 5.22.2 / 0.21.2 / 1.15.0 |
| tokenizers / safetensors | 0.23.3 / 0.8.0 |
| numpy / Pillow / matplotlib | 2.1.3 / 13.0.0 / 3.10.0 |
| GPUs | 2 x Tesla T4 (kernel uses GPU 0) |

Not recorded: jobF01-03, jobF09-12 (built before the probe) and the clean-clone rerun jobR1 (built with `make_kernel.py`,
which has no probe; its predictions are byte-identical to the original D run, `reports/repro.md`).
