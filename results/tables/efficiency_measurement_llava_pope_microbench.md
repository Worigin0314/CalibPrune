Runner-level microbenchmark table. Reductions are relative to the matched unpruned run with the same model, dataset, seed, and sample count; positive values mean lower token count, latency, or peak allocated CUDA memory.

| Model | Dataset | Method | r | n | Seeds | Token_down | Prefill_latency_down | End_to_end_latency_down | Peak_memory_down | Prefill_s_per_sample | E2E_s_per_sample | Peak_MB | Acc | ECE | Evidence |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| llava15_7b_4bit | pope | fastv | 0.50 | 16 | 20260621 | +50.0% | -34.0% | -34.0% | -17.5% | 0.579 | 0.579 | 4793.9 | 0.9375 | 0.1571 | runner-hook |
| llava15_7b_4bit | pope | none | 1.00 | 16 | 20260621 | +0.0% | +0.0% | +0.0% | +0.0% | 0.432 | 0.432 | 4081.0 | 0.9375 | 0.1509 | runner-hook |
| llava15_7b_4bit | pope | sparsevlm | 0.50 | 16 | 20260621 | +50.0% | +34.2% | +34.2% | -0.5% | 0.284 | 0.285 | 4103.3 | 0.9375 | 0.1424 | literature-midlayer-hook |
| llava15_7b_4bit | pope | vtw | 0.50 | 16 | 20260621 | +50.0% | +32.3% | +32.3% | -0.5% | 0.292 | 0.293 | 4103.3 | 0.9375 | 0.1500 | literature-midlayer-hook |
