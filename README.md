# CalibPrune

CalibPrune studies calibration drift caused by visual token pruning in small
vision-language models, then tests a pruning-coupled temperature correction.

This repository was initialized from `CalibPrune_Experiment_Plan.md` for the
PRICAI 2026 experiment track. The current codebase contains the local experiment
scaffold, metrics, calibrators, pruner contracts, fixture smoke pipeline, and
run logging. Real VLM runs require local model and dataset caches; the current
workspace has LLaVA-1.5-7B and Qwen2-VL-2B smoke paths documented below.

## Quick Start

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\verify_env.py --core-only
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m calibprune.pipelines.run_eval `
  --model toy_vlm --dataset pope --split test --n 8 `
  --pruner none --retention 1.0 --calibrator none --offline-fixture
```

To reproduce the latest real-model engineering gate, run the three deterministic
POPE 128/512 seeds and then aggregate only the unpruned/FastV literature-hook
rows:

```powershell
$env:PYTHONPATH = "src"
foreach ($seed in 20260616,20260617,20260618) {
  $suffix = if ($seed -eq 20260616) { "" } else { "_seed$seed" }
  .\.venv\Scripts\python.exe scripts\run_pope_lite_inprocess.py `
    --model llava15_7b_4bit --dataset pope --seed $seed `
    --n-cal 128 --n-test 512 --pruners none,fastv `
    --retentions 0.25,0.5,0.75 `
    --output-dir "results\raw\pope_lite_llava_128_512$suffix"
}

.\.venv\Scripts\python.exe scripts\aggregate_results.py `
  --input-glob "results/raw/pope_lite_llava_128_512*/*.json" `
  --pruners none,fastv --exclude-sanity `
  --output-csv results\tables\pope_lite_llava_128_512_fastv_multiseed_summary.csv
```

To reproduce the latest single-seed two-pruner engineering gate, reuse the
same POPE 128/512 split with both FastV and VisionZip, then batch-calibrate the
saved logits:

```powershell
.\.venv\Scripts\python.exe scripts\run_pope_lite_inprocess.py `
  --model llava15_7b_4bit --dataset pope --seed 20260616 `
  --n-cal 128 --n-test 512 --pruners none,fastv,visionzip `
  --retentions 0.25,0.5,0.75 `
  --output-dir results\raw\pope_lite_llava_128_512

.\.venv\Scripts\python.exe scripts\calibrate_lite_grid.py `
  --model llava15_7b_4bit --dataset pope --seed 20260616 `
  --n-cal 128 --n-test 512 --pruners fastv,visionzip `
  --retentions 0.25,0.5,0.75 `
  --output-dir results\raw\pope_lite_llava_128_512_visionzip_calibrated `
  --output-prefix llava_pope_128_512

.\.venv\Scripts\python.exe scripts\aggregate_results.py `
  --input-glob "results/raw/pope_lite_llava_128_512/llava15_7b_4bit-pope-*.json" `
  --input-glob "results/raw/pope_lite_llava_128_512_visionzip_calibrated/*.json" `
  --subset-name test --pruners none,fastv,visionzip --exclude-sanity `
  --output-csv results\tables\pope_lite_llava_128_512_fastv_visionzip_test_comparison.csv
```

## Robust POPE-Lite Runner

Create deterministic cal/test splits and run a resumable small grid. For real
LLaVA runs, prefer `run_pope_lite_inprocess.py` because it loads the model once
and evaluates all cal/test cells in the same process. The runner started as a
POPE utility but now accepts `--dataset pope`, `scienceqa`, `mmbench_cn`, or
`aokvqa`.

```powershell
.\.venv\Scripts\python.exe scripts\prepare_calibration_split.py `
  --dataset pope --n-total 9000 --n-cal 500 --output-dir data\splits

.\.venv\Scripts\python.exe scripts\run_pope_lite_inprocess.py `
  --model llava15_7b_4bit --n-cal 128 --n-test 512 `
  --pruners none,fastv,uniform,random --retentions 0.5 `
  --output-dir results\raw\pope_lite_llava_128_512

.\.venv\Scripts\python.exe scripts\analyze_results.py `
  --input-glob "results/raw/pope_lite_llava_128_512/*.json" `
  --pruners none,fastv --exclude-sanity `
  --output-csv results\tables\pope_lite_llava_128_512_summary.csv
```

The POPE-lite runners write stable output names, save answer-level logits, skip
completed cells only when the split hash and sample count match, and rerun
stale cells automatically. Pass `--force` to `run_pope_lite_inprocess.py` when
a code change requires recomputing matching cells. `run_pope_lite.py` remains
useful for subprocess isolation; `run_pope_lite_inprocess.py` is faster for
longer real-model grids. Non-default seeds use seed-suffixed split files such as
`data/splits/pope_cal500_seed20260617.json`; explicitly mixing a split file
with the wrong seed now fails fast. Manifest files include a grid fingerprint so
sub-grid runs in the same output directory do not overwrite each other.

Current real POPE-lite engineering gate, using LLaVA-1.5-7B 4-bit with
`n_cal=128`, `n_test=512`, and `retention=0.5`:

| Condition | Accuracy | ECE | Delta ECE vs matching none |
|---|---:|---:|---:|
| none, raw | 0.8340 | 0.0458 | 0.0000 |
| fastv, raw | 0.8242 | 0.0556 | +0.0099 |
| random, raw | 0.8223 | 0.0646 | +0.0188 |
| uniform, raw | 0.8340 | 0.0455 | -0.0003 |
| feature_norm, raw | 0.8379 | 0.0306 | -0.0152 |
| fastv, temperature scaling | 0.8242 | 0.0484 | -0.0081 |
| feature_norm, temperature scaling | 0.8379 | 0.0283 | -0.0282 |

FastV multi-retention gate, two-seed mean using `n_cal=64`, `n_test=256`:

| FastV retention | Calibrator | Accuracy | ECE | Delta ECE vs raw |
|---:|---|---:|---:|---:|
| 0.25 | none | 0.7852 | 0.0841 | 0.0000 |
| 0.25 | calibprune | 0.7852 | 0.0544 | -0.0297 |
| 0.50 | none | 0.8262 | 0.0683 | 0.0000 |
| 0.50 | calibprune | 0.8262 | 0.0541 | -0.0142 |
| 0.75 | none | 0.8418 | 0.0430 | 0.0000 |
| 0.75 | calibprune | 0.8418 | 0.0485 | +0.0055 |

FastV multi-retention scale-up, three-seed mean using `n_cal=128`,
`n_test=512`:

| FastV retention | Calibrator | Accuracy | ECE | Delta ECE vs raw |
|---:|---|---:|---:|---:|
| 0.25 | none | 0.7995 | 0.0860 | 0.0000 |
| 0.25 | temperature_scaling | 0.7995 | 0.0567 | -0.0293 |
| 0.25 | calibprune | 0.7995 | 0.0585 | -0.0275 |
| 0.50 | none | 0.8359 | 0.0564 | 0.0000 |
| 0.50 | temperature_scaling | 0.8359 | 0.0520 | -0.0044 |
| 0.50 | calibprune | 0.8359 | 0.0500 | -0.0065 |
| 0.75 | none | 0.8457 | 0.0423 | 0.0000 |
| 0.75 | temperature_scaling | 0.8457 | 0.0469 | +0.0046 |
| 0.75 | calibprune | 0.8457 | 0.0468 | +0.0045 |

FastV vs VisionZip single-seed engineering comparison using `n_cal=128`,
`n_test=512`, seed `20260616`:

| Pruner | Retention | Calibrator | Accuracy | ECE | Delta ECE vs unpruned raw |
|---|---:|---|---:|---:|---:|
| none | 1.00 | none | 0.8340 | 0.0458 | 0.0000 |
| FastV | 0.25 | none | 0.7988 | 0.0788 | +0.0330 |
| FastV | 0.50 | none | 0.8242 | 0.0556 | +0.0099 |
| FastV | 0.75 | none | 0.8379 | 0.0345 | -0.0113 |
| VisionZip | 0.25 | none | 0.8242 | 0.0492 | +0.0034 |
| VisionZip | 0.50 | none | 0.8340 | 0.0499 | +0.0041 |
| VisionZip | 0.75 | none | 0.8359 | 0.0415 | -0.0043 |

Reproducible gate figures for this POPE setting can be regenerated with:

```powershell
.\.venv\Scripts\python.exe scripts\generate_gate_figures.py
```

The script writes ECE-retention, confidence-histogram, reliability, and
risk-coverage figures under `results\figures\`, with matching `.tex` snippets.
Its pooled r=0.5 table is
`results\tables\pope_lite_llava_128_512_pooled_figure_metrics.csv`.
Paired bootstrap CIs for the same saved-logits gate can be regenerated with:

```powershell
.\.venv\Scripts\python.exe scripts\generate_gate_statistics.py
```

This writes `results\tables\pope_lite_llava_128_512_paired_bootstrap_ci.csv`;
for r=0.5, the ECE drift from raw FastV over unpruned has a positive 95% CI,
and CalibPrune's ECE reduction versus raw FastV has a negative 95% CI, while
the AURC change is not supported as an improvement.

H2 logit-shift diagnostics can be regenerated directly from saved answer-level
logits. These tables should be used for mechanism claims instead of inferring
logit sharpening from ECE alone:

```powershell
.\.venv\Scripts\python.exe scripts\generate_logit_diagnostics.py `
  --model llava15_7b_4bit --dataset pope --n-test 512 `
  --seeds 20260616,20260617,20260618 `
  --pruners fastv --retentions 0.25,0.5,0.75 `
  --output-csv results\tables\logit_sharpening_llava_pope_fastv_512_3seed.csv

.\.venv\Scripts\python.exe scripts\generate_logit_diagnostics.py `
  --model qwen2vl_2b --dataset pope --n-test 512 `
  --seeds 20260616,20260617,20260618,20260619 `
  --pruners fastv --retentions 0.5 `
  --output-csv results\tables\logit_sharpening_qwen2vl_pope_fastv_512_4seed.csv

.\.venv\Scripts\python.exe scripts\generate_logit_diagnostics.py `
  --model llava15_7b_4bit --dataset pope --n-test 705 `
  --seeds 20260616,20260617,20260618 `
  --pruners sparsevlm,vtw --retentions 0.5 `
  --output-csv results\tables\logit_sharpening_llava_pope_sparse_vtw_layer8_705_3seed.csv
```

The canonical reliability triptych for the AURC caution can be regenerated with:

```powershell
.\.venv\Scripts\python.exe scripts\generate_reliability_triptych.py `
  --result-glob "results/raw/pope_lite_llava_128_512*/*.json" `
  --model llava15_7b_4bit --dataset pope --n-test 512 `
  --seeds 20260616,20260617,20260618 `
  --pruner fastv --retention 0.5 `
  --calibrator temperature_scaling `
  --stem pope_llava_fastv_r05_ts_triptych `
  --summary-csv results\tables\pope_llava_fastv_r05_ts_triptych_summary.csv
```

The summary table shows the key trade-off: temperature scaling lowers ECE from
0.049206 to 0.027529 on FastV r=0.5, while AURC worsens from 0.070002 to
0.076679.

Saved-logit lite grids can be batch-calibrated without hand-writing every
retention path:

```powershell
.\.venv\Scripts\python.exe scripts\calibrate_lite_grid.py `
  --model llava15_7b_4bit --dataset pope --seed 20260616 `
  --n-cal 8 --n-test 16 --pruners fastv,visionzip `
  --retentions 0.25,0.5,0.75 `
  --output-dir results\raw\pope_lite_llava_visionzip_8_16
```

The helper writes unpruned temperature scaling plus per-pruner
temperature-scaling and CalibPrune JSONs from the logits saved by
`run_pope_lite_inprocess.py`. Pass `--include-adaptive-calibprune` to also fit
the optional sample-confidence adaptive CalibPrune ablation.

The same figure pipeline can now be applied to ScienceQA:

```powershell
.\.venv\Scripts\python.exe scripts\generate_gate_figures.py `
  --dataset scienceqa --n-cal 128 --n-test 256 `
  --aggregate-csv results\tables\scienceqa_lite_llava_128_256_multiseed_summary.csv `
  --result-glob "results/raw/scienceqa_lite_llava_128_256*/*.json" `
  --summary-csv results\tables\scienceqa_lite_llava_128_256_pooled_figure_metrics.csv
```

It also supports the three-seed MMBench-CN 128/256 FastV/VisionZip gate;
generated captions automatically state the seed count:

```powershell
.\.venv\Scripts\python.exe scripts\generate_gate_figures.py `
  --dataset mmbench_cn --n-cal 128 --n-test 256 `
  --seeds 20260616,20260617,20260618 `
  --aggregate-csv results\tables\mmbench_cn_lite_llava_128_256_fastv_visionzip_3seed_test_comparison.csv `
  --result-glob "results/raw/mmbench_cn_lite_llava_128_256/*.json" `
  --result-glob "results/raw/mmbench_cn_lite_llava_128_256_seed20260617/*.json" `
  --result-glob "results/raw/mmbench_cn_lite_llava_128_256_seed20260618/*.json" `
  --result-glob "results/raw/mmbench_cn_lite_llava_128_256_calibrated/*.json" `
  --result-glob "results/raw/mmbench_cn_lite_llava_128_256_seed20260617_calibrated/*.json" `
  --result-glob "results/raw/mmbench_cn_lite_llava_128_256_seed20260618_calibrated/*.json" `
  --summary-csv results\tables\mmbench_cn_lite_llava_128_256_3seed_pooled_figure_metrics.csv
```

Feature-norm sanity baseline at retention 0.5, two-seed mean using
`n_cal=128`, `n_test=512`:

| Condition | Seeds | Accuracy | ECE | Delta ECE |
|---|---:|---:|---:|---:|
| none, raw | 2 | 0.8486 | 0.0328 | baseline |
| feature_norm, raw | 2 | 0.8477 | 0.0264 | -0.0064 vs none raw |
| feature_norm, temperature scaling | 2 | 0.8477 | 0.0386 | +0.0122 vs feature_norm raw |

ScienceQA image-subset gate, three-seed mean using streaming data access with
`n_total=2017`, `n_cal=128`, `n_test=256`, and FastV retentions
`{0.25, 0.5, 0.75}`:

| Condition | Seeds | Accuracy | ECE | Delta ECE |
|---|---:|---:|---:|---:|
| none, raw | 3 | 0.6237 | 0.1609 | baseline |
| none, temperature scaling | 3 | 0.6237 | 0.0962 | -0.0647 vs none raw |
| FastV r=0.25, raw | 3 | 0.6354 | 0.1494 | -0.0115 vs none raw |
| FastV r=0.25, temperature scaling | 3 | 0.6354 | 0.0845 | -0.0649 vs FastV raw |
| FastV r=0.25, CalibPrune | 3 | 0.6354 | 0.1093 | -0.0401 vs FastV raw |
| FastV r=0.50, raw | 3 | 0.6146 | 0.1608 | -0.0001 vs none raw |
| FastV r=0.50, temperature scaling | 3 | 0.6146 | 0.1032 | -0.0575 vs FastV raw |
| FastV r=0.50, CalibPrune | 3 | 0.6146 | 0.0980 | -0.0628 vs FastV raw |
| FastV r=0.75, raw | 3 | 0.6146 | 0.1614 | +0.0005 vs none raw |
| FastV r=0.75, temperature scaling | 3 | 0.6146 | 0.0891 | -0.0723 vs FastV raw |
| FastV r=0.75, CalibPrune | 3 | 0.6146 | 0.0898 | -0.0715 vs FastV raw |

MMBench-CN now has a real LLaVA multi-retention, multi-seed engineering gate
through the same runner. The loader dynamically counts the `lmms-lab/MMBench`
`cc`/`test` split and preserves the shuffled split order for streaming rows.
The latest scaled gate uses `n_cal=128`, `n_test=256`, three deterministic
seeds, FastV, and VisionZip:

```powershell
foreach ($seed in 20260616,20260617,20260618) {
  $suffix = if ($seed -eq 20260616) { "" } else { "_seed$seed" }
  .\.venv\Scripts\python.exe scripts\run_pope_lite_inprocess.py `
    --model llava15_7b_4bit --dataset mmbench_cn --split test `
    --seed $seed --n-cal 128 --n-test 256 `
    --pruners none,fastv,visionzip --retentions 0.25,0.5,0.75 `
    --output-dir "results\raw\mmbench_cn_lite_llava_128_256$suffix"

  .\.venv\Scripts\python.exe scripts\calibrate_lite_grid.py `
    --model llava15_7b_4bit --dataset mmbench_cn --seed $seed `
    --n-cal 128 --n-test 256 --pruners fastv,visionzip `
    --retentions 0.25,0.5,0.75 `
    --output-dir "results\raw\mmbench_cn_lite_llava_128_256${suffix}_calibrated" `
    --output-prefix llava_mmbench_cn_128_256
}
```

MMBench-CN engineering gate (`n_cal=128`, `n_test=256`, seeds
`20260616,20260617,20260618`):

| Condition | Seeds | Accuracy | ECE | Delta ECE |
|---|---:|---:|---:|---:|
| none, raw | 3 | 0.3997 | 0.2522 | baseline |
| none, temperature scaling | 3 | 0.3997 | 0.0740 | -0.1782 vs none raw |
| FastV r=0.25, raw | 3 | 0.3893 | 0.2649 | +0.0128 vs none raw |
| FastV r=0.25, temperature scaling | 3 | 0.3893 | 0.0572 | -0.2078 vs FastV raw |
| FastV r=0.25, CalibPrune | 3 | 0.3893 | 0.0608 | -0.2041 vs FastV raw |
| FastV r=0.50, raw | 3 | 0.3919 | 0.2675 | +0.0153 vs none raw |
| FastV r=0.50, temperature scaling | 3 | 0.3919 | 0.0507 | -0.2168 vs FastV raw |
| FastV r=0.50, CalibPrune | 3 | 0.3919 | 0.0808 | -0.1867 vs FastV raw |
| VisionZip r=0.25, raw | 3 | 0.4063 | 0.2569 | +0.0047 vs none raw |
| VisionZip r=0.25, temperature scaling | 3 | 0.4063 | 0.0596 | -0.1973 vs VisionZip raw |
| VisionZip r=0.25, CalibPrune | 3 | 0.4063 | 0.0593 | -0.1976 vs VisionZip raw |
| VisionZip r=0.50, raw | 3 | 0.4049 | 0.2561 | +0.0039 vs none raw |
| VisionZip r=0.50, temperature scaling | 3 | 0.4049 | 0.0596 | -0.1965 vs VisionZip raw |
| VisionZip r=0.50, CalibPrune | 3 | 0.4049 | 0.0720 | -0.1841 vs VisionZip raw |

The same saved logits can run the adaptive CalibPrune ablation without new VLM
forwards:

```powershell
.\.venv\Scripts\python.exe scripts\calibrate_lite_grid.py `
  --model llava15_7b_4bit --dataset mmbench_cn --seed 20260616 `
  --n-cal 128 --n-test 256 --pruners fastv,visionzip `
  --retentions 0.25,0.5,0.75 `
  --output-dir results\raw\mmbench_cn_lite_llava_128_256_adaptive_log_margin `
  --output-prefix llava_mmbench_cn_128_256 `
  --include-adaptive-calibprune
```

By default, `adaptive_calibprune` now uses a probability-margin feature, a
log-temperature link, and `gamma_l2=0.05`. MMBench-CN adaptive CalibPrune
ablation, three-seed mean:

| Condition | ECE | Delta ECE vs raw |
|---|---:|---:|
| FastV r=0.25, adaptive CalibPrune | 0.0511 | -0.2138 |
| FastV r=0.50, adaptive CalibPrune | 0.0549 | -0.2126 |
| FastV r=0.75, adaptive CalibPrune | 0.0687 | -0.1953 |
| VisionZip r=0.25, adaptive CalibPrune | 0.0473 | -0.2096 |
| VisionZip r=0.50, adaptive CalibPrune | 0.0497 | -0.2064 |
| VisionZip r=0.75, adaptive CalibPrune | 0.0524 | -0.2042 |

These runs are still marked `paper_eligible=false`: they validate the real
pipeline and expose a candidate pruning-calibration signal, but they are not a
complete PRICAI evidence package. The FastV gates show calibration is useful
under stronger pruning, but the best calibrator is not uniform across seeds:
at 128/512, temperature scaling has the better three-seed mean at retention 0.25,
CalibPrune only slightly helps at retention 0.5, and neither method should be
claimed as an improvement at retention 0.75. `feature_norm` is only a sanity
baseline for projected-feature ranking; it is not a literature-pruner result,
and temperature scaling is not supported for it on the two-seed mean. The
ScienceQA result now exercises CalibPrune on a second real dataset with three
deterministic seeds, but the current 256-sample-per-seed gate is still
engineering evidence rather than a complete PRICAI claim. On this gate,
temperature scaling is best at FastV retentions 0.25 and 0.75, while
CalibPrune is slightly better at retention 0.5. The MMBench-CN 128/256 gate now
proves that the third real dataset path can produce multi-retention JSON/logits
and CalibPrune summaries for both FastV and VisionZip at a meaningful pilot
scale across three seeds. Both temperature scaling and CalibPrune reduce ECE
versus raw pruned logits on all six pruned MMBench-CN cells, but CalibPrune is
not uniformly best: it only edges temperature scaling on VisionZip r=0.25 in
the three-seed mean. Adaptive CalibPrune improves over the original CalibPrune
on all six pruned MMBench-CN cells and beats ordinary temperature scaling on
five of six ECE cells in that dataset-level mean table. The replication gate is
now more conservative: POPE 128/512 is mixed, ScienceQA 128/256 favors ordinary
temperature scaling, and paired bootstrap CIs for AdaptiveCalibPrune versus
temperature scaling cross zero after Holm-tracked Wilcoxon diagnostics. Treat
log-margin AdaptiveCalibPrune as a benchmark-dependent adaptive calibration
ablation, not a stable headline replacement for temperature scaling. The compact
cross-dataset evidence table is
`results\tables\adaptive_log_margin_cross_dataset_ts_comparison.md`. It still
worsens max-softmax AURC, so selective-reliability improvement remains
unsupported. FastV r=0.5 MMBench-CN diagnostic figures are available under
`results\figures\mmbench_cn_128_256_*`.

Additional robustness and reproduction paths are available:

```powershell
.\.venv\Scripts\python.exe scripts\run_pope_lite_inprocess.py `
  --model qwen2vl_2b --dataset pope --split test --seed 20260616 `
  --n-cal 128 --n-test 512 --pruners none,fastv --retentions 0.5 `
  --output-dir results\raw\pope_lite_qwen2vl_2b_fastv_128_512 --force

.\.venv\Scripts\python.exe scripts\apply_family_holm.py `
  --metric ece `
  --input-csv results\tables\pope_lite_llava_128_512_adaptive_vs_temperature_paired_stats.csv --label POPE `
  --input-csv results\tables\scienceqa_lite_llava_128_256_adaptive_vs_temperature_paired_stats.csv --label ScienceQA `
  --input-csv results\tables\mmbench_cn_lite_llava_128_256_adaptive_vs_temperature_paired_stats.csv --label MMBench-CN-fastv `
  --input-csv results\tables\mmbench_cn_lite_llava_128_256_visionzip_adaptive_vs_temperature_paired_stats.csv --label MMBench-CN-visionzip `
  --input-csv results\tables\aokvqa_full_validation_llava_fastv_r05_adaptive_vs_temperature_paired_stats.csv --label A-OKVQA `
  --output-csv results\tables\adaptive_vs_temperature_ece_family_holm.csv

.\.venv\Scripts\python.exe scripts\summarize_adaptive_vs_temperature.py
```

`qwen2vl_2b` now has a real FastV-style pruning hook: it computes Qwen2-VL
image embeddings, uses decoder-layer attention to rank `<|image_pad|>` tokens,
and reruns the decoder after slicing `inputs_embeds`, `attention_mask`, and
`position_ids`. The current non-smoke evidence is a POPE 128/512 four-seed
FastV gate at `results\tables\pope_lite_qwen2vl_2b_fastv_128_512_4seed_test_comparison.md`:
mean ECE changes from 0.163495 unpruned to 0.129510 at FastV r=0.5, while mean
accuracy changes from 0.497070 to 0.509277. The paired raw pruning stats are in
`results\tables\pope_lite_qwen2vl_2b_fastv_128_512_4seed_pruned_vs_unpruned_stats.md`.
The paper-gate report is
`results\tables\pope_lite_qwen2vl_2b_fastv_128_512_4seed_paper_gate.md`; pooled
n=2048 now passes the current n>=2000/min-seed gate. The earlier n=1
accuracy-zero smoke was one wrong POPE sample, not an answer-decoding failure.

A-OKVQA full validation reproduction is complete at
`results\tables\aokvqa_full_validation_llava_fastv_r05_test_comparison.md`, and
its calibrated companion table is
`results\tables\aokvqa_full_validation_llava_fastv_r05_calibrated_test_comparison.md`.
On FastV r=0.5, temperature scaling improves ECE from 0.060480 to 0.051043,
base CalibPrune to 0.051770, while adaptive log-margin worsens ECE to 0.138961.
The guarded adaptive selector chooses among TS/base/adaptive on a held-out
calibration split. On A-OKVQA FastV r=0.5 it reaches ECE 0.031833; see
`results\tables\aokvqa_full_validation_llava_fastv_r05_guarded_adaptive_result.md`.
The conservative selector with `--selection-margin 0.03 --refit-selected` was
also replicated on POPE/ScienceQA/MMBench-CN FastV three-seed grids; it is more
stable than default adaptive on average but still does not beat TS on average,
so it remains an overfit guard rather than a headline calibrator. See
`results\tables\guarded_adaptive_lite_grid_margin003_refit_fastv_vs_baselines.md`.
The cross-dataset AdaptiveCalibPrune-vs-temperature table now includes A-OKVQA
and uses the family-level Holm artifact at
`results\tables\adaptive_vs_temperature_ece_family_holm.md`.

## Current Experiment Boundary

- `toy_vlm` plus `--offline-fixture` is only a pipeline smoke test.
- Paper claims must use JSON outputs from real model and real dataset runs.
- `fastv` and `visionzip` are currently wired literature-style real-model
  pruning hooks. `visionzip` has passed LLaVA/POPE and MMBench-CN engineering
  comparisons against FastV, including the three-seed MMBench-CN 128/256 gate,
  but it remains `paper_eligible=false` until the planned full-scale repeated
  grids and official-number checks are complete.
- `results\tables\fastv_official_reproduction_check.md` records the current
  FastV official-number audit. The original FastV paper's LLaVA-1.5-7B table
  reports Nocaps, Flickr30k, A-OKVQA, and MMMU, so the local POPE gate cannot be
  claimed as an official FastV reproduction.
- `sparsevlm`, `pyramiddrop`, and `vtw` now execute real mid-layer sequence
  surgery inside the LLaVA decoder and are tagged `literature-midlayer-hook` when
  that path runs. SparseVLM uses text-guided recycling; PyramidDrop progressively
  prunes after layers 8/16/24; VTW withdraws visual tokens while retaining
  boundary and attention-relevant tokens. SparseVLM and VTW default to
  `max(2, n_layers // 4)` (layer 8 on LLaVA-1.5-7B) and can be overridden with
  `CALIBPRUNE_SPARSEVLM_LAYER` and `CALIBPRUNE_VTW_LAYER`.
- The LLaVA/POPE all-comparator table is complete at
  `results\tables\pope_lite_llava_midlayer_sota_1_704_3seed_test_comparison.md`;
  its gate at
  `results\tables\pope_lite_llava_midlayer_sota_1_704_3seed_paper_gate.md`
  passes for all six rows with 3 seeds and pooled n=2112. This table used the
  earlier layer-2 SparseVLM/VTW setting and is useful as a stress result:
  PyramidDrop is near unpruned and VisionZip improves AURC, while layer-2
  SparseVLM/VTW increased calibration drift.
- The improved SparseVLM/VTW default is the layer-8 task-scale gate at
  `results\tables\pope_lite_llava_sparse_vtw_layer8_1_705_3seed_test_comparison.md`
  and
  `results\tables\pope_lite_llava_sparse_vtw_layer8_1_705_3seed_paper_gate.md`.
  It passes with 3 seeds / pooled n=2115. Mean accuracy changes from 0.840662
  unpruned to 0.843026 SparseVLM and 0.843972 VTW; AURC improves from 0.063860
  to 0.061180 and 0.061054. ECE is statistically inconclusive rather than a win,
  so present this as a near-lossless/default-improved comparator result.
- `scienceqa` currently uses the image subset of the Hugging Face
  `derek-thomas/ScienceQA` test split via streaming access. Its deterministic
  split is based on 2017 image-bearing test samples.
- `mmbench_cn` now has an executable multiple-choice loader path for
  `lmms-lab/MMBench` config `cc` and real LLaVA FastV/VisionZip gates at
  32/64 and three-seed 128/256. The latest table is
  `results\tables\mmbench_cn_lite_llava_128_256_fastv_visionzip_3seed_test_comparison.md`.
  It is not paper evidence until the full planned scale and official-number
  checks are complete.
- `aokvqa` now has an executable multiple-choice loader path and a full
  validation LLaVA/FastV r=0.5 official-task reproduction: 1145/1145 validation
  samples, local 4-bit accuracy 0.787773 unpruned and 0.791266 FastV.
- `qwen2vl_2b` now has a Qwen-specific FastV-style visual-token hook. It is
  verified by a real-model POPE 128/512 four-seed paper-gate pass plus the n=1
  token-slicing smoke; broader official-task second-backbone evidence is still
  required before making a broad cross-backbone claim.
- `vqav2`, `gqa`, and `textvqa` are deferred from paper-facing tables until
  open-ended answer extraction, normalization, and scoring are implemented. Do not use
  them as evidence in the current manuscript scope.
- `feature_norm` is an executable sanity baseline, not a PRICAI method claim or
  a substitute for a published pruning baseline.
- `CalibPrune` now requires at least two retention ratios during fitting; a
  single-retention run should use ordinary temperature scaling or remain raw.
- `adaptive_calibprune` is an optional ablation that adds a standardized
  margin/entropy/confidence sharpness term to `T(r)`. The default uses
  margin-log temperature with `gamma_l2=0.05`: ECE improves on the MMBench-CN
  mean table, but POPE/ScienceQA replication and paired TS-vs-adaptive CIs show
  the advantage is benchmark-dependent. AURC still does not support a
  selective-reliability claim. A selective-aware objective can be enabled with
  `--adaptive-selective-weight`, but current MMBench-CN runs expose an ECE/AURC
  trade-off rather than a free improvement.
- `paper/claims.md` is the claim-evidence map; keep unsupported claims marked
  as `needs evidence`.
- Existing POPE-lite outputs are engineering/pipeline validation unless their
  JSON explicitly sets `paper_eligible=true`.










