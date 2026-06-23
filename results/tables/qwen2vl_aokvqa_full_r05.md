Qwen2-VL closed-set sanity gate. Acc-minus-chance uses a paired bootstrap over per-sample correctness minus the uniform random chance for that sample's answer set.

| Model | Dataset | Method | r | n | Accuracy | Mean_random_chance | Acc_minus_chance_95CI | ECE | AURC | Token_down | Verbalizer | GPU |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| qwen2vl_2b | aokvqa | fastv | 0.50 | 1145 | 0.3083 | 0.2500 | 0.0583 [0.0312, 0.0836] | 0.2653 | 0.6452 | 50.1% | as_written | NVIDIA GeForce RTX 4070 Laptop GPU |
| qwen2vl_2b | aokvqa | none | 1.00 | 1145 | 0.2681 | 0.2500 | 0.0181 [-0.0072, 0.0434] | 0.2023 | 0.7058 | 0.0% | as_written | NVIDIA GeForce RTX 4070 Laptop GPU |
