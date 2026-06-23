# FastV Official Reproduction Check

Source audited: FastV arXiv v3, "An Image is Worth 1/2 Tokens After Layer 2:
Plug-and-Play Inference Acceleration for Large Vision-Language Models"
(`arXiv:2403.06764`).

## Official FastV Reference Values

The FastV paper's LLaVA-1.5-7B main table reports Nocaps, Flickr30k,
A-OKVQA, and MMMU. It does not provide an official POPE score in the paper
text audited here.

| Source | Model | Setting | Benchmark | Official score |
|---|---|---|---|---:|
| FastV arXiv v3 Table 1 | LLaVA-1.5-7B | Baseline | Nocaps CIDEr | 99.8 |
| FastV arXiv v3 Table 1 | LLaVA-1.5-7B | Baseline | Flickr30k CIDEr | 67.9 |
| FastV arXiv v3 Table 1 | LLaVA-1.5-7B | Baseline | A-OKVQA accuracy | 76.7 |
| FastV arXiv v3 Table 1 | LLaVA-1.5-7B | Baseline | MMMU accuracy | 34.8 |
| FastV arXiv v3 Table 1 | LLaVA-1.5-7B | FastV K=2 R=50% | Nocaps CIDEr | 99.7 |
| FastV arXiv v3 Table 1 | LLaVA-1.5-7B | FastV K=2 R=50% | Flickr30k CIDEr | 67.5 |
| FastV arXiv v3 Table 1 | LLaVA-1.5-7B | FastV K=2 R=50% | A-OKVQA accuracy | 77.0 |
| FastV arXiv v3 Table 1 | LLaVA-1.5-7B | FastV K=2 R=50% | MMMU accuracy | 34.4 |

## Local POPE Gate Is Not The Official Check

| Source | Model | Setting | Benchmark | Local score | Status |
|---|---|---|---|---:|---|
| CalibPrune local POPE lite | LLaVA-1.5-7B 4-bit | none raw | POPE 128/512 accuracy | 0.833984 | not comparable |
| CalibPrune local POPE lite | LLaVA-1.5-7B 4-bit | FastV r=0.50 raw | POPE 128/512 accuracy | 0.824219 | not official reproduction |
| CalibPrune local A-OKVQA lite | LLaVA-1.5-7B 4-bit | none raw | A-OKVQA validation 32/64 accuracy | 0.765625 | superseded by full validation |
| CalibPrune local A-OKVQA lite | LLaVA-1.5-7B 4-bit | FastV r=0.50 raw | A-OKVQA validation 32/64 accuracy | 0.734375 | superseded by full validation |
| CalibPrune local A-OKVQA full validation | LLaVA-1.5-7B 4-bit | none raw | A-OKVQA validation 1145/1145 accuracy | 0.787773 | full official-task reproduction, high side vs 0.767 |
| CalibPrune local A-OKVQA full validation | LLaVA-1.5-7B 4-bit | FastV r=0.50 raw | A-OKVQA validation 1145/1145 accuracy | 0.791266 | full official-task reproduction, high side vs 0.770 |

## Verdict

- The current POPE/ScienceQA/MMBench-CN gates validate this repository's
  calibration pipeline and FastV-style hook behavior, but they are not official
  FastV benchmark reproductions.
- A-OKVQA full validation is now complete for LLaVA-1.5-7B 4-bit with the local
  FastV r=0.50 hook: `results/tables/aokvqa_full_validation_llava_fastv_r05_test_comparison.md`.
- Local accuracy is 78.78% for the unpruned model and 79.13% for FastV r=0.50,
  versus FastV Table 1's 76.7% and 77.0%. The absolute values are on the high
  side by about 2.1 percentage points, but the FastV-vs-baseline direction is
  consistent with the official table (+0.35 local points versus +0.3 official
  points).
- This should be described as a completed full-validation A-OKVQA reproduction
  under the local 4-bit/Hugging Face prompt-and-answer protocol, not as an exact
  byte-for-byte reproduction of the FastV authors' evaluation harness.


