# Anonymous Code Package Manifest

This folder is a lightweight copy prepared for anonymous GitHub upload.

Included:

- `src/`: CalibPrune package source code.
- `scripts/`: experiment, summarization, calibration, plotting, and table-generation scripts.
- `configs/`: dataset and run configuration files.
- `tests/`: focused unit/smoke tests.
- `paper/`: LaTeX sources, bibliography/style files, paper figures, and compiled PDFs.
- `results/tables/`: lightweight CSV/Markdown tables supporting the manuscript.
- Root metadata: `README.md`, `pyproject.toml`, `.gitignore`, and experiment plan.

Excluded:

- Local model weights under `models/`.
- Dataset caches under `data/`, `.hf-cache/`, and related cache folders.
- Raw logits and raw run payloads under `results/raw/`.
- Runtime logs and local virtual environments.

Before uploading, review README and manuscript author fields for the target anonymous submission policy.
