Qwen2-VL closed-set sanity gate. Acc-minus-chance uses a paired bootstrap over per-sample correctness minus the uniform random chance for that sample's answer set.

| Model | Dataset | Method | r | n | Accuracy | Mean_random_chance | Acc_minus_chance_95CI | ECE | AURC | Token_down | Verbalizer | GPU |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| qwen2vl_2b | scienceqa | fastv | 0.25 | 512 | 0.4258 | 0.3570 | 0.0688 [0.0264, 0.1107] | 0.1881 | 0.4608 | 75.0% | as_written | NVIDIA GeForce RTX 4070 Laptop GPU |
| qwen2vl_2b | scienceqa | fastv | 0.50 | 512 | 0.4238 | 0.3570 | 0.0668 [0.0242, 0.1116] | 0.2036 | 0.4673 | 49.9% | as_written | NVIDIA GeForce RTX 4070 Laptop GPU |
| qwen2vl_2b | scienceqa | fastv | 0.75 | 512 | 0.4238 | 0.3570 | 0.0668 [0.0236, 0.1100] | 0.2057 | 0.4652 | 25.0% | as_written | NVIDIA GeForce RTX 4070 Laptop GPU |
| qwen2vl_2b | scienceqa | none | 1.00 | 512 | 0.4629 | 0.3570 | 0.1059 [0.0637, 0.1498] | 0.1318 | 0.4413 | 0.0% | as_written | NVIDIA GeForce RTX 4070 Laptop GPU |
