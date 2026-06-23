Wall-clock speedup is a coarse runner-level proxy computed against the matching unpruned test run with the same model, dataset, seed, and sample count. Positive means faster. Peak GPU memory is not recorded by current JSON outputs.

| model | dataset | pruner | retention | n_samples_per_seed | n_seeds | seeds | token_reduction | wall_clock_speedup_proxy | seconds_per_sample | accuracy | ece | aurc |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| llava15_7b_4bit | aokvqa | fastv | 0.50 | 1145 | 1 | 20260616 | 50.0% | -58.2% | 0.730 | 0.7913 | 0.0605 | 0.0687 |
| llava15_7b_4bit | aokvqa | none | 1.00 | 1145 | 1 | 20260616 | 0.0% | 0.0% | 0.462 | 0.7878 | 0.0644 | 0.0703 |
| llava15_7b_4bit | pope | none | 1.00 | 705 | 3 | 20260616,20260617,20260618 | 0.0% +/- 0.0% | 0.0% +/- 0.0% | 0.419 +/- 0.016 | 0.8407 +/- 0.0208 | 0.0357 +/- 0.0117 | 0.0639 +/- 0.0106 |
| llava15_7b_4bit | pope | sparsevlm | 0.50 | 705 | 3 | 20260616,20260617,20260618 | 50.0% +/- 0.0% | 5.7% +/- 24.4% | 0.394 +/- 0.090 | 0.8430 +/- 0.0201 | 0.0407 +/- 0.0102 | 0.0612 +/- 0.0103 |
| llava15_7b_4bit | pope | vtw | 0.50 | 705 | 3 | 20260616,20260617,20260618 | 50.0% +/- 0.0% | 4.4% +/- 24.7% | 0.399 +/- 0.091 | 0.8440 +/- 0.0209 | 0.0372 +/- 0.0140 | 0.0611 +/- 0.0103 |
| qwen2vl_2b | pope | fastv | 0.50 | 512 | 4 | 20260616,20260617,20260618,20260619 | 50.0% +/- 0.1% | -21.1% +/- 9.2% | 0.584 +/- 0.048 | 0.5093 +/- 0.0125 | 0.1295 +/- 0.0143 | 0.4851 +/- 0.0225 |
| qwen2vl_2b | pope | none | 1.00 | 512 | 4 | 20260616,20260617,20260618,20260619 | 0.0% +/- 0.0% | 0.0% +/- 0.0% | 0.484 +/- 0.051 | 0.4971 +/- 0.0163 | 0.1635 +/- 0.0219 | 0.4899 +/- 0.0341 |
