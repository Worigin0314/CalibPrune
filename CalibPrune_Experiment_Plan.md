# CalibPrune Experiment Plan — Agent-Executable Master Document

> **Project**: CalibPrune — Revealing and Mitigating Calibration Drift in Visual Token Pruning for Small Vision-Language Models
> **Target venue**: PRICAI 2026 (long paper)
> **Hardware**: RTX 4070 Laptop 8 GB VRAM, 32 GB RAM, Windows 11 + WSL2 / Linux
> **Owner**: Worigin
> **Doc version**: v1.0 (2026-06-16)
> **Doc audience**: An autonomous coding agent (Claude Code / Codex / Cursor agent) that executes experiments end-to-end without further clarification.

---

## 0. How to Use This Document (for the Agent)

This document is the single source of truth. You (the agent) MUST:

1. Treat each `## Step N` as an atomic milestone with an explicit **DONE criterion**.
2. Before claiming a step is done, run its **verification command** and paste the actual output in the run log.
3. Maintain `runs/run_log.md` in the project repo with one section per step: timestamp, commands run, key outputs, anomalies.
4. If any **STOP condition** in §11 triggers, halt and request human input — do not silently work around it.
5. All paths assume `PROJECT_ROOT=C:/Users/Worigin/projects/calibprune` (Windows) or `~/projects/calibprune` (WSL/Linux).
6. Use the Python venv at `PROJECT_ROOT/.venv` exclusively. Do not install packages globally.
7. Never download a model checkpoint larger than 8 GB without first checking free disk space and the §3 model size table.

The document is organized as:

| Part | Section | Purpose |
|------|---------|---------|
| **Setup** | §1 – §4 | Environment, models, datasets |
| **Phase A (Empirical Study)** | §5 – §7 | Reproduce baselines, discover calibration drift |
| **Phase B (Method)** | §8 – §9 | Implement and tune CalibPrune |
| **Phase C (Evaluation)** | §10 | Main results, ablations, visualizations |
| **Risk & Writing** | §11 – §14 | Stop rules, fallback, paper deliverables |

---

## 1. Project Overview

### 1.1 Research Question
Does visual token pruning systematically degrade the **confidence calibration** of small VLMs (≤7B), even when accuracy is preserved? If so, can a training-free post-hoc method restore calibration without retraining the model?

### 1.2 Hypotheses
- **H1 (Phenomenon)**: Across mainstream training-free pruners (FastV, SparseVLM, VisionZip, PyramidDrop, VTW), Expected Calibration Error (ECE) increases by ≥1.5× at 50% retention versus the unpruned VLM, on at least 4 of 6 benchmarks.
- **H2 (Mechanism)**: The increase is driven by **logit sharpening** — the post-pruning maximum softmax probability distribution shifts rightward (mean Δ ≥ 0.05).
- **H3 (Remedy)**: A one-parameter pruning-coupled temperature \(T(r) = T_0 + \beta(1-r)\) tuned on a 500-sample calibration set reduces ECE by ≥30% while keeping accuracy drop ≤0.3 absolute points.
- **H4 (Selective benefit)**: CalibPrune improves Area Under Risk-Coverage (AURC) by ≥15% over uncorrected pruning, demonstrating downstream reliability value.

### 1.3 Success / Failure Criteria
- **GO** (proceed to writing): H1 holds AND (H3 OR H4) holds.
- **PIVOT** (switch to fallback): H1 fails after Step 6.
- **SOFT STOP**: H1 holds but H3 fails — pivot the framing to "CalibPrune as diagnostic + open problem" (short paper).

### 1.4 Out-of-scope (do not implement unless explicitly asked)
- Fine-tuning any VLM backbone
- Training new pruning heads
- Models > 8B parameters at any precision
- Multi-GPU experiments
- Reinforcement learning, RLHF, DPO

---

## 2. Environment Setup

### 2.1 Hardware assumptions
- GPU: NVIDIA RTX 4070 Laptop, 8 GB VRAM (compute capability 8.9)
- Driver: ≥ 555.x
- CUDA runtime: 12.1 (we will pin to this)
- RAM: ≥ 32 GB
- Disk: ≥ 200 GB free for datasets + models

### 2.2 OS choice
Prefer **WSL2 Ubuntu 22.04** on the user's Windows machine for cleaner CUDA tooling. Fallback: native Windows + Anaconda. All commands below assume bash. Translate to PowerShell only if WSL is unavailable.

### 2.3 One-shot bootstrap
```bash
# Run inside WSL or Git Bash
export PROJECT_ROOT="$HOME/projects/calibprune"
mkdir -p "$PROJECT_ROOT" && cd "$PROJECT_ROOT"

# Python 3.10 via uv (preferred) or conda
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.10 .venv
source .venv/bin/activate

# Core scientific stack — pinned versions for reproducibility
uv pip install \
  "torch==2.3.1+cu121" "torchvision==0.18.1+cu121" \
  --extra-index-url https://download.pytorch.org/whl/cu121

uv pip install \
  "transformers==4.44.2" \
  "accelerate==0.33.0" \
  "bitsandbytes==0.43.3" \
  "peft==0.12.0" \
  "datasets==2.20.0" \
  "evaluate==0.4.2" \
  "scikit-learn==1.5.1" \
  "numpy==1.26.4" \
  "pandas==2.2.2" \
  "matplotlib==3.9.1" \
  "seaborn==0.13.2" \
  "Pillow==10.4.0" \
  "tqdm==4.66.5" \
  "einops==0.8.0" \
  "sentencepiece==0.2.0" \
  "tiktoken==0.7.0" \
  "pyyaml==6.0.2" \
  "hydra-core==1.3.2" \
  "wandb==0.17.7" \
  "rich==13.7.1" \
  "netcal==1.3.5" \
  "torchmetrics==1.4.1"

# Sanity check
python -c "import torch; print('CUDA OK:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**DONE criterion**: `python -c "import torch; print(torch.cuda.is_available())"` prints `True` AND reported GPU name contains `RTX 4070`.

### 2.4 Repository skeleton
```
calibprune/
├── README.md
├── pyproject.toml
├── configs/
│   ├── models/
│   │   ├── llava15_7b_4bit.yaml
│   │   ├── minicpmv26_2b.yaml
│   │   └── qwen2vl_2b.yaml
│   ├── datasets/
│   │   ├── vqav2.yaml
│   │   ├── gqa.yaml
│   │   ├── scienceqa.yaml
│   │   ├── textvqa.yaml
│   │   ├── mmbench_cn.yaml
│   │   └── pope.yaml
│   ├── pruners/
│   │   ├── none.yaml
│   │   ├── fastv.yaml
│   │   ├── sparsevlm.yaml
│   │   ├── visionzip.yaml
│   │   ├── pyramiddrop.yaml
│   │   └── vtw.yaml
│   └── calibrators/
│       ├── none.yaml
│       ├── temperature_scaling.yaml
│       ├── platt.yaml
│       ├── histogram_binning.yaml
│       └── calibprune.yaml
├── src/calibprune/
│   ├── __init__.py
│   ├── models/
│   │   ├── loader.py            # 4-bit loaders for each VLM
│   │   └── vlm_wrapper.py       # unified forward(image, query) → logits
│   ├── pruners/
│   │   ├── base.py
│   │   ├── fastv.py
│   │   ├── sparsevlm.py
│   │   ├── visionzip.py
│   │   ├── pyramiddrop.py
│   │   └── vtw.py
│   ├── calibrators/
│   │   ├── base.py
│   │   ├── temperature.py
│   │   ├── platt.py
│   │   ├── histogram.py
│   │   └── calibprune.py        # core contribution
│   ├── metrics/
│   │   ├── calibration.py       # ECE, MCE, Brier, NLL, reliability
│   │   ├── selective.py         # AURC, risk-coverage curve
│   │   └── statistics.py        # bootstrap CI, paired tests
│   ├── data/
│   │   └── loaders.py           # unified VQA loader
│   ├── pipelines/
│   │   ├── run_eval.py          # single (model, pruner, dataset, ratio)
│   │   ├── run_phenomenon.py    # Phase A grid
│   │   └── run_method.py        # Phase B/C grid
│   └── viz/
│       └── plots.py             # all paper figures
├── scripts/
│   ├── download_datasets.sh
│   ├── download_models.py
│   ├── prepare_calibration_split.py
│   ├── verify_env.py
│   └── reproduce_table1.sh
├── tests/
│   ├── test_pruners.py
│   ├── test_calibrators.py
│   ├── test_metrics.py
│   └── test_smoke.py
├── runs/
│   └── run_log.md               # agent maintains this
├── results/
│   ├── tables/
│   ├── figures/
│   └── raw/                     # per-run JSON dumps
└── paper/
    ├── outline.md
    ├── related_work.md
    └── claims.md
```

### 2.5 Verification (`scripts/verify_env.py`)
Implement this script to print: torch version, CUDA version, GPU name, free VRAM (GB), free disk (GB on PROJECT_ROOT), versions of transformers/bitsandbytes. The agent runs this at the start of every session.

**DONE criterion for §2**: Project skeleton exists; verify_env.py prints "ALL CHECKS PASSED".

---

## 3. Models

We use 3 backbones to ensure cross-architecture generalization of the claimed phenomenon. **All loaded in 4-bit via bitsandbytes NF4** unless noted.

### 3.1 Model registry

| Tag | HF repo | Params | Precision | VRAM (load) | VRAM (inference peak) | Use |
|-----|---------|--------|-----------|-------------|----------------------|-----|
| `llava15_7b_4bit` | `llava-hf/llava-1.5-7b-hf` | 7.06 B | NF4 + bf16 compute | ~4.2 GB | ~6.5 GB | **Primary** |
| `minicpmv26_2b` | `openbmb/MiniCPM-V-2_6` | 2.43 B | NF4 + bf16 compute | ~1.8 GB | ~3.8 GB | Cross-arch |
| `qwen2vl_2b` | `Qwen/Qwen2-VL-2B-Instruct` | 2.21 B | NF4 + bf16 compute | ~1.6 GB | ~3.5 GB | Cross-arch |

> If 4-bit LLaVA-1.5-7B + dataset preprocessing peaks above 7.5 GB on the user's laptop (driver overhead), reduce `max_new_tokens` to 64 and disable any KV-cache padding. Document in run_log.

### 3.2 Loader contract (`src/calibprune/models/loader.py`)
Each loader returns:
```python
class VLM(NamedTuple):
    name: str
    model: torch.nn.Module          # in eval mode, 4-bit
    processor: Any                  # HF processor
    visual_token_layer_idx: int     # where visual tokens enter LLM
    num_visual_tokens: int          # default budget
    forward_with_logits: Callable[[PIL.Image, str], LogitsBundle]
```

Where `LogitsBundle` exposes:
```python
@dataclass
class LogitsBundle:
    logits: torch.Tensor            # [V] over vocab at first generated token
    pred_token: int
    pred_text: str
    answer_set_logits: Optional[Dict[str, float]]  # for MCQ benchmarks
```

This unified API is what every pruner / calibrator / metric consumes.

### 3.3 Download script (`scripts/download_models.py`)
- Uses `huggingface_hub.snapshot_download` with `local_dir=$PROJECT_ROOT/models/<tag>`.
- Skips if already present.
- Asserts free disk ≥ 30 GB before each download.

**DONE criterion for §3**: All three models loadable; smoke test `python -m calibprune.pipelines.run_eval --model llava15_7b_4bit --dataset pope --split val[:8] --pruner none --calibrator none` returns predictions without OOM.

---

## 4. Datasets

### 4.1 Dataset registry

| Tag | Source | Eval split | # samples used | Task type | Answer extraction | Notes |
|-----|--------|------------|----------------|-----------|--------------------|-------|
| `vqav2` | HF `HuggingFaceM4/VQAv2` | val | 5 000 (stratified) | open-ended VQA | first generated token + softmax over answer vocab top-3000 | use VQA accuracy |
| `gqa` | HF `lmms-lab/GQA` | testdev_balanced | 5 000 | open-ended VQA | same as VQAv2 | |
| `scienceqa` | HF `derek-thomas/ScienceQA` | test (image subset) | 4 241 | MCQ | logit over answer letters A–E | image questions only |
| `textvqa` | HF `lmms-lab/textvqa` | val | 5 000 | open-ended VQA | open-text matching | OCR-heavy |
| `mmbench_cn` | HF `lmms-lab/MMBench` (zh) | dev | 4 377 | MCQ | logit over A–D | **PRICAI 加分项** |
| `pope` | HF `lmms-lab/POPE` | popular + random + adversarial | 9 000 | binary yes/no | logit(yes) vs logit(no) | hallucination |

### 4.2 Download (`scripts/download_datasets.sh`)
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$PROJECT_ROOT"
mkdir -p data && cd data
python -c "
from datasets import load_dataset
import os
target = os.environ['PROJECT_ROOT'] + '/data/hf_cache'
os.environ['HF_DATASETS_CACHE'] = target
for ds, conf, split in [
    ('lmms-lab/POPE', None, 'test'),
    ('derek-thomas/ScienceQA', None, 'test'),
    ('lmms-lab/GQA', 'testdev_balanced_instructions', 'testdev'),
    ('lmms-lab/textvqa', None, 'validation'),
    ('lmms-lab/MMBench', 'cc', 'dev'),
    ('HuggingFaceM4/VQAv2', None, 'validation[:6000]'),
]:
    print('Pulling', ds, conf, split)
    load_dataset(ds, conf, split=split, cache_dir=target)
"
```

### 4.3 Calibration split (`scripts/prepare_calibration_split.py`)
For each eval dataset, **deterministically** carve out:
- `cal500`: 500 samples (used to fit calibrators) — seed = 20260616
- `test`: remaining samples (used to report ALL numbers in the paper)

Persist split indices to `data/splits/<dataset>_cal500.json` so every run reads identical sets.

**Critical**: NEVER fit any calibrator on the test split. Every script must assert this.

### 4.4 Answer-set logit extraction
Open-ended VQA must be converted to closed-form scoring for calibration measurement:
- For VQAv2/GQA/TextVQA: build the top-3000 answer vocabulary from training set, score with sum of token log-probs at first generation step (use processor tokenizer offsets).
- For ScienceQA/MMBench: score over single answer-letter tokens.
- For POPE: score yes vs no first-token logit.

This logit-extraction module is in `src/calibprune/models/vlm_wrapper.py::extract_answer_logits()` and is shared by all pruners.

**DONE criterion for §4**: All six dataset configs load; `python -c "from calibprune.data.loaders import build; b = build('pope', 'test', n=4); print(b[0])"` prints a valid sample with image + question + gold answer.

---

## 5. Baseline Pruners (Phase A — Step 1)

We reproduce 5 mainstream training-free visual token pruners + 2 sanity baselines.

### 5.1 Pruner contract (`src/calibprune/pruners/base.py`)
```python
class Pruner(Protocol):
    name: str
    requires_attentions: bool

    def __call__(
        self,
        visual_tokens: torch.Tensor,      # [N, D]
        attentions: Optional[List[torch.Tensor]],  # per-layer attn weights so far
        retention_ratio: float,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # (pruned_tokens, kept_indices)
        ...
```

All pruners are hooked into the VLM via a forward hook on the LLM's input embedding layer (per backbone, the exact module differs — document each in `pruners/<name>.py` docstring).

### 5.2 Pruners

| Pruner | Ref | Trigger layer | Score | Notes for impl |
|--------|-----|---------------|-------|----------------|
| `none` | — | — | — | identity |
| `random` | sanity | input | uniform | seed-fixed |
| `uniform` | sanity | input | spatial-uniform | every K-th token |
| `fastv` | Chen et al. ECCV 2024 | layer K (default K=2) | text→image avg attention | drop bottom (1-r) |
| `sparsevlm` | Zhang et al. ICML 2025 | layer 2 | text-guided + clustering | re-cluster after drop |
| `visionzip` | Yang et al. CVPR 2025 | input (encoder side) | CLIP cls-token attn + merge | dominant + contextual tokens |
| `pyramiddrop` | Xing et al. CVPR 2025 | layers {8,16,24} | per-layer attn, progressive | pyramid schedule |
| `vtw` | Lin et al. AAAI 2025 | beyond layer K* | withdraw all visual tokens after K* | k* tuned per model |

### 5.3 Reference repos to mine for correctness
- FastV: github.com/pkunlp-icler/FastV
- SparseVLM: github.com/Gumpest/SparseVLMs
- VisionZip: github.com/dvlab-research/VisionZip
- PyramidDrop: github.com/Cooperx521/PyramidDrop
- VTW: github.com/lzhxmu/VTW

Do not vendor their code; reimplement in our unified API. Validate by matching their published accuracy at 50% retention on POPE within 1.0 absolute points.

### 5.4 Unit tests (`tests/test_pruners.py`)
- Token-count invariant: kept tokens count = round(r · N)
- Order-preservation flag respected
- Random pruner determinism with seed
- FLOPs estimator returns monotone decreasing in r

**DONE criterion for §5**: All 7 pruners pass unit tests; FastV on POPE@retention 0.5 with LLaVA-1.5-7B is within 1.0 accuracy points of the FastV paper's reported number; runtime under 12 min for 1 000 samples on RTX 4070.

---

## 6. Metrics (Phase A — Step 2)

### 6.1 Calibration metrics (`src/calibprune/metrics/calibration.py`)
- **ECE** (15-bin equal-width) — primary
- **Adaptive ECE** (15 equal-mass bins) — robustness
- **MCE**
- **Brier score** (multi-class one-hot)
- **NLL**
- **Reliability diagram** data: confidence bins, accuracy per bin, sample count per bin
- All implementations cross-checked against `netcal` library on a synthetic logit set in `tests/test_metrics.py`.

### 6.2 Selective metrics (`src/calibprune/metrics/selective.py`)
- Risk-coverage curve (sweep coverage from 0.1 to 1.0)
- AURC (Area Under Risk-Coverage Curve) — lower is better
- E-AURC (excess AURC over Bayes-optimal)
- Selective accuracy at coverage ∈ {0.8, 0.9, 0.95}

### 6.3 Statistical tests (`src/calibprune/metrics/statistics.py`)
- 1 000-sample paired bootstrap CI (BCa) for every reported number
- Paired Wilcoxon signed-rank for "CalibPrune vs baseline" at α = 0.05
- Holm correction across 6 datasets

### 6.4 Output schema (every run dumps one JSON to `results/raw/`)
```json
{
  "run_id": "20260619-1547-llava15_7b_4bit-pope-fastv-r0.5-none",
  "model": "llava15_7b_4bit",
  "dataset": "pope",
  "pruner": "fastv",
  "retention": 0.5,
  "calibrator": "none",
  "n_samples": 9000,
  "accuracy": 0.8612,
  "ece": 0.0934,
  "adaptive_ece": 0.0876,
  "mce": 0.241,
  "brier": 0.0871,
  "nll": 0.382,
  "aurc": 0.0421,
  "selective_acc": {"0.80": 0.912, "0.90": 0.881, "0.95": 0.862},
  "reliability": {"bin_conf": [...], "bin_acc": [...], "bin_count": [...]},
  "max_softmax_mean": 0.872,
  "max_softmax_p95": 0.991,
  "wall_clock_min": 8.4,
  "peak_vram_gb": 6.3,
  "git_commit": "abc1234",
  "config_hash": "..."
}
```

**DONE criterion for §6**: ECE/Brier/AURC implementations match `netcal` to 1e-4; output schema validated by `tests/test_metrics.py::test_schema_round_trip`.

---

## 7. Phenomenon Discovery Grid (Phase A — Step 3, **the critical week**)

This is the make-or-break experiment for H1/H2. Run this **before** writing any method code.

### 7.1 Grid
- Models: {llava15_7b_4bit, minicpmv26_2b, qwen2vl_2b}
- Datasets: {vqav2, gqa, scienceqa, textvqa, mmbench_cn, pope}
- Pruners: {none, fastv, sparsevlm, visionzip, pyramiddrop, vtw}
- Retention: {0.25, 0.50, 0.75} (none uses 1.0)
- Calibrator: none
- Seeds: {20260616, 20260617, 20260618}

Total: 3 × 6 × (1 + 5 × 3) × 3 = **864 cells**.

### 7.2 Compute budget reality check
On RTX 4070 Laptop 8 GB:
- LLaVA-1.5-7B 4-bit POPE 1 000 samples ≈ 12 min → 9 000 ≈ 108 min
- Total Phase A wall time ≈ 864 × ~30 min mean = ~430 hours sequential

This is too long. **Mitigations to apply before launching**:

1. Subsample test split per dataset to a fixed `n=2000` (after carving cal500). All paper numbers will state this clearly.
2. Run 3 seeds only on POPE (hallucination — most sensitive) and on the primary model LLaVA. Other cells use single seed.
3. After §7.4 first pass, optionally drop retention=0.75 if drift is monotone in r (typical case).

Revised budget: ~80 wall-hours → about **5 nights of overnight runs**.

### 7.3 Launcher (`src/calibprune/pipelines/run_phenomenon.py`)
- Hydra config with `+sweep=phenomenon`
- Atomic per-cell writes; resumable (skip if `results/raw/<run_id>.json` exists)
- Logs to wandb project `calibprune-phaseA`
- Per-cell timeout 60 min; on OOM, retry once with `batch_size=1, max_new_tokens=32`

### 7.4 Analysis (`notebooks/01_phenomenon.ipynb` or `scripts/analyze_phenomenon.py`)
Produce:
- **Table P1**: rows = pruner × retention; cols = ECE (mean ± 95% CI) per dataset; final col = average ΔECE vs unpruned.
- **Table P2**: same shape but for accuracy.
- **Figure F1 (teaser)**: x = retention ratio, y = ECE; one line per pruner; subplots per dataset.
- **Figure F2**: max-softmax probability histograms before/after pruning at r=0.5, four panels.

### 7.5 Decision gate
After running, check H1 + H2.

**GO**: continue to §8. Record evidence in `paper/claims.md`.
**PIVOT**: trigger §13 fallback to CLCP.
**SOFT STOP**: H1 yes / H2 no → continue but adjust framing.

**DONE criterion for §7**: results/raw/ contains ≥ 95% of expected JSON files; Tables P1/P2 + Figures F1/F2 generated; decision recorded in `runs/run_log.md`.

---

## 8. CalibPrune Method (Phase B)

### 8.1 Method recap (formal)
Given a frozen VLM \(f\), a pruner \(\pi\) with retention ratio \(r \in (0, 1]\), and a calibration set \(D_{\text{cal}}\):

**Pruning-coupled temperature**:
\[ T(r; T_0, \beta) = T_0 + \beta \cdot (1 - r) \]

**Prediction**:
\[ \hat p(y \mid I, Q; \pi, r) = \mathrm{softmax}\!\left( \frac{f(\pi(I, r), Q)}{T(r; T_0^\star, \beta^\star)} \right) \]

**Parameter fit** on `cal500`:
\[ (T_0^\star, \beta^\star) = \arg\min_{T_0, \beta \in \mathbb{R}_+} \mathrm{ECE}_{15}\!\left( D_{\text{cal}}, T(r; T_0, \beta) \right) \]
solved by grid search: \(T_0 \in \{0.8, 0.9, ..., 1.5\}\) × \(\beta \in \{0.0, 0.1, ..., 2.0\}\) (default), with a fine local refinement (Nelder-Mead) around the grid minimum.

**Optional Pruning-Mask Ensemble (PME)** for stochastic pruners (random, FastV with attention-noise injection, SparseVLM with clustering randomness):
\[ \hat p_{\text{PME}}(y) = \frac{1}{K} \sum_{k=1}^K \hat p(y; \pi^{(k)}, r), \quad K \in \{3, 5\} \]

### 8.2 Calibrator implementation (`src/calibprune/calibrators/calibprune.py`)
```python
class CalibPrune(Calibrator):
    name = "calibprune"

    def __init__(self, t0_grid, beta_grid, use_pme=False, K=3):
        ...

    def fit(self, logits_at_ratios: Dict[float, np.ndarray], labels: np.ndarray):
        # logits_at_ratios[r] -> [N, V] logits collected on cal500 at retention r
        # iterate (T0, beta) grid, evaluate combined ECE across retentions, pick argmin
        # store T0_star, beta_star
        ...

    def transform(self, logits: np.ndarray, retention: float) -> np.ndarray:
        return logits / (self.T0_star + self.beta_star * (1.0 - retention))
```

### 8.3 Calibration baselines to beat
- `none` (raw)
- `temperature_scaling` (one global T fit on cal500, ignoring r)
- `platt` (logistic on max-softmax — adapted from binary, use top-1 vs top-2 margin)
- `histogram_binning` (15 bins fit on cal500)
- Optional: `Dirichlet calibration` (advanced; only if time)

### 8.4 Hyperparameter sensitivity (results in §10.3 ablation)
- T0 grid: 0.6 ↔ 2.0 (step 0.1)
- β grid: 0.0 ↔ 3.0 (step 0.1)
- Calibration set size N ∈ {100, 250, 500, 1000}
- PME K ∈ {1, 3, 5}
- Robustness to cross-dataset calibration (fit on GQA, eval on VQAv2)

### 8.5 Theory note (§Method 3.4 of paper)
Include one **Proposition** with proof sketch:
> Under a logit-sharpening model where pruning multiplies pre-softmax logits by a factor \(\rho(r) > 1\) (verified empirically in F2), the calibration-optimal temperature is exactly \(\rho(r)\). A first-order Taylor expansion gives \(\rho(r) \approx 1 + \beta(1-r)\) for some \(\beta\) depending on the model.

Proof goes in appendix.

**DONE criterion for §8**: `tests/test_calibrators.py::test_calibprune_recovers_temperature_on_synth` passes (synthetic logits with known sharpening; CalibPrune recovers β within 10%).

---

## 9. Main Experiment Grid (Phase C — main results)

### 9.1 Grid
- Models: {llava15_7b_4bit (primary), minicpmv26_2b, qwen2vl_2b}
- Datasets: all 6 (test split, 2 000 samples each)
- Pruners: {fastv, sparsevlm, visionzip, pyramiddrop, vtw}
- Retention: {0.25, 0.50, 0.75}
- Calibrators: {none, temperature_scaling, platt, histogram_binning, calibprune, calibprune+PME}
- Seeds: 3 on primary model + POPE; 1 elsewhere

For each cell record full schema in §6.4.

### 9.2 Main Table 1 — Calibration on LLaVA-1.5-7B
Rows: pruner × retention (15 rows). Cols: 6 datasets × {Acc, ECE, AURC}. Bold = best per (dataset × metric × retention). CalibPrune is the proposed method row.

### 9.3 Main Table 2 — Cross-backbone generalization
Avg-of-datasets summary per backbone for CalibPrune vs strongest baseline (Temperature Scaling) at retention 0.5.

### 9.4 Main Table 3 — Calibrator comparison
Pin model=LLaVA, retention=0.5; rows=calibrators; cols=datasets×{ECE, AURC}; show CalibPrune dominates.

---

## 10. Ablation & Visualization

### 10.1 Ablation tables

| Ablation | Vary | Hold | Output |
|----------|------|------|--------|
| A1 | use_pme ∈ {off, K=3, K=5} | model=LLaVA, pruner=FastV, r=0.5 | shows PME marginal value |
| A2 | calibration size N ∈ {100, 250, 500, 1000} | model=LLaVA, pruner=FastV, r=0.5 | data-efficiency curve |
| A3 | β grid step ∈ {0.05, 0.1, 0.2, 0.5} | model=LLaVA, pruner=FastV, r=0.5 | hyperparam robustness |
| A4 | cross-dataset fit: train on D_i, eval on D_j | model=LLaVA, pruner=FastV, r=0.5 | 6×6 matrix |
| A5 | retention granularity: {0.25, 0.5, 0.75} vs single-r-fit | | shows multi-r fit necessity |
| A6 | T0 only (β=0) vs full | | β is load-bearing |
| A7 | Cross-backbone calibrator transfer | fit on LLaVA, apply to MiniCPM | shows backbone-dependence |

### 10.2 Figures (publication-ready, code in `src/calibprune/viz/plots.py`)

1. **Figure 1 (teaser)**: ECE vs retention, 6 subplots per dataset, 5 pruner lines + raw VLM dashed.
2. **Figure 2**: Reliability diagrams — 3 panels (raw, FastV @ r=0.5, FastV @ r=0.5 + CalibPrune).
3. **Figure 3**: Max-softmax histogram pre/post pruning, supports H2.
4. **Figure 4**: Risk-coverage curves on POPE, three lines (raw, FastV, FastV+CalibPrune).
5. **Figure 5 (appendix)**: Per-benchmark ECE bar chart.
6. **Figure 6 (appendix)**: β★ vs pruner (one bar per pruner), shows pruner-specific sharpening.

All figures use:
- Colorblind-safe palette (matplotlib `tab10` or `viridis`)
- Vector PDF + 600-dpi PNG fallback
- Font size ≥ 9 pt in final size

### 10.3 Compute budget for §9–§10
Main grid: 3 × 6 × 5 × 3 × 6 ≈ 1 620 cells, but **calibrators reuse pruner outputs**: pruner+inference is run **once** per (model, dataset, pruner, retention, seed) and stored in `results/raw/logits/`. Then calibrators are CPU post-processing. Effective new VLM forward passes ≈ Phase A volume → another ~80 wall-hours.

**DONE criterion for §10**: All 6 tables + 6 figures present in `results/tables/` and `results/figures/`; each figure has a sibling `.tex` `\includegraphics` snippet.

---

## 11. STOP Conditions & Risk Register

Immediately halt and message the user if any of these fire:

| Code | Trigger | Action |
|------|---------|--------|
| S1 | Phase A finds ΔECE < 1.3× at r=0.5 averaged across pruners (H1 fails) | Halt; propose §13 pivot |
| S2 | Sustained VRAM OOM despite batch_size=1, max_new_tokens=32 | Halt; report; consider further quantization or smaller model fallback |
| S3 | Any baseline pruner cannot reproduce within 2.0 absolute accuracy points of published numbers at r=0.5 | Halt; do not silently report mismatched results |
| S4 | Disk free < 10 GB at any moment | Halt; do not delete data; ask user |
| S5 | Single experiment cell exceeds 90 min wall time | Halt that cell; report; consider further dataset subsampling |
| S6 | More than 5% of grid cells fail with same error | Halt; treat as systematic bug |
| S7 | CalibPrune fails on synthetic recovery test | Halt; method is bugged |

Risks NOT requiring halt (proceed but log):
- Per-cell variance > 2 ECE points → add more seeds for that cell
- Cross-backbone transfer of calibrator fails → make this a finding, not a silent failure

---

## 12. Schedule (4–8 Week Plan)

| Wk | Phase | Deliverables | DONE criterion |
|----|-------|--------------|----------------|
| W1 | Setup | §2–§4 complete; smoke test on LLaVA-POPE | verify_env passes; 8-sample POPE eval prints accuracy |
| W2 | Baselines | §5 all 7 pruners; match published POPE@r=0.5 within 1.0 abs pt | unit tests pass; reproduction log committed |
| W3 | **Phenomenon** | §7 grid on LLaVA across all 6 datasets; Tables P1/P2; Figure F1 | **GO / PIVOT decision recorded** |
| W4 | Method | §8 CalibPrune impl + synthetic test + first run on FastV@LLaVA@POPE | ECE on POPE@r=0.5 drops ≥ 30% |
| W5 | Main grid | §9 grid completes; Main Tables 1–3 | results/tables/ has 3 tables; bootstrap CIs computed |
| W6 | Ablation | §10 ablations A1–A7 + all figures | all 6 figures pdf-rendered |
| W7 | Paper draft | Intro, Method, Experiments, Related Work | full draft compiled, ≤ 8 pages main + refs |
| W8 | Polish + submit | Internal review, rebuttal-prep notes, camera prep | submission package zipped |

Slack: W4 can absorb 1 week of W3 overrun. If phenomenon finding is unambiguous in W3, advance schedule by 1 week.

---

## 13. Fallback Plan: Switch to CLCP

If §7 decision gate triggers PIVOT, switch to **CLCP (Cross-Layer Consistency Pruning)** as the paper:

- Reuse §1–§6 and §5 infrastructure verbatim.
- New method (`src/calibprune/pruners/clcp.py`): compute a per-token rank in attention at layers L* = {2, 8, 16, 24}, define
  \[ \mathrm{CLCS}(t) = \overline{\mathrm{rank}}_{l \in L^*}(t) - \lambda \cdot \mathrm{Var}_{l \in L^*}\,\mathrm{rank}_l(t) \]
  and prune by lowest CLCS.
- New target metric: accuracy-FLOPs trade-off; **also report ECE drift** to leverage Phase A data (this becomes a strength).
- The Phase A phenomenon data is NOT wasted — it appears in CLCP paper's §3.1 motivation ("pruning is fragile across layers, hence cross-layer voting").

Estimated additional work: 5–7 days. Schedule still fits 8 weeks.

---

## 14. Paper Deliverables Checklist

For PRICAI long-paper submission (typically 14 pages incl. refs):

- [ ] Title + abstract (200 words English)
- [ ] §1 Intro with 1 teaser figure + 3 contributions
- [ ] §2 Related Work in 4 categories (see paper §5 outline above)
- [ ] §3 Method: notation, empirical study summary, CalibPrune formal definition, theory proposition + proof
- [ ] §4 Experiments: setup table, Tables 1–3, Figures 1–4
- [ ] §5 Ablations: Tables A1–A7
- [ ] §6 Discussion: limitations, broader impact (incl. PRICAI-style discussion of resource-constrained AI in APAC)
- [ ] §7 Conclusion
- [ ] Appendix: theory proof, additional figures (5–6), per-benchmark breakdown, reproducibility checklist
- [ ] Reproducibility checklist (PRICAI requires)
- [ ] Code release pointer (anonymized for double-blind)

### 14.1 Anti-pattern guardrails (from academic-paper skill writing rules)
- Do NOT use em-dash overuse, "delve into", "crucial", "it is important to note".
- Do NOT fabricate citations; every reference verified via DOI lookup before submission.
- Every plot must have axis labels, units, legend, and a self-contained caption.
- Every claim in abstract must map to a Table/Figure in the paper (build claim-evidence map in `paper/claims.md`).

---

## 15. Reproducibility Package (for submission)

- All seeds fixed (20260616, 20260617, 20260618)
- `pyproject.toml` pins every dep
- `scripts/reproduce_table1.sh` reproduces Main Table 1 end-to-end on a single 8 GB GPU within 24 wall-hours
- Pre-computed `results/raw/` JSONs released alongside code
- README explains the directory layout and how to extend

---

## 16. Open Questions for the Human

Before launching the long-running Phase A grid, the agent SHOULD pause and confirm:

1. Is the dataset subsampling (n=2000 test per benchmark) acceptable? (Default: yes.)
2. Do we need a 4th backbone (e.g., InternVL2-2B) for additional generalization? (Default: no; can add in W6 as bonus.)
3. Should we add a medical extension (Idea 9 MedSelect-VLM) as Section 6.1 application study? (Default: no; risk additional time.)
4. wandb project name and entity to use? (Agent should set up but not commit credentials.)

The agent's default behavior is to proceed with the defaults above and surface anything blocking.

---

## 17. Glossary

- **VLM**: Vision-Language Model
- **ECE**: Expected Calibration Error
- **AURC**: Area Under Risk-Coverage Curve
- **PME**: Pruning-Mask Ensemble
- **Retention ratio r**: fraction of visual tokens kept after pruning
- **Logit sharpening**: phenomenon where post-softmax max prob distribution shifts toward 1.0
- **cal500**: a 500-sample held-out split exclusively for fitting calibrators

---

*End of master document. The agent should now create the repository skeleton (§2.4), run `verify_env.py` (§2.5), and proceed to §3 model download. Maintain `runs/run_log.md` from the very first action.*
