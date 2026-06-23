Qwen2-VL closed-set sanity gate. Acc-minus-chance uses a paired bootstrap over per-sample correctness minus the uniform random chance for that sample's answer set.

| Model | Dataset | Method | r | n | Accuracy | Mean_random_chance | Acc_minus_chance_95CI | ECE | AURC | Token_down | Verbalizer | GPU |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| qwen2vl_2b | aokvqa | fastv | 0.25 | 512 | 0.3027 | 0.2500 | 0.0527 [0.0117, 0.0938] | 0.2588 | 0.6481 | 75.0% | as_written | NVIDIA GeForce RTX 4070 Laptop GPU |
| qwen2vl_2b | aokvqa | fastv | 0.50 | 512 | 0.2910 | 0.2500 | 0.0410 [0.0000, 0.0801] | 0.2825 | 0.6708 | 50.0% | as_written | NVIDIA GeForce RTX 4070 Laptop GPU |
| qwen2vl_2b | aokvqa | fastv | 0.75 | 512 | 0.2832 | 0.2500 | 0.0332 [-0.0078, 0.0723] | 0.2890 | 0.6672 | 25.0% | as_written | NVIDIA GeForce RTX 4070 Laptop GPU |
| qwen2vl_2b | aokvqa | none | 1.00 | 512 | 0.2559 | 0.2500 | 0.0059 [-0.0332, 0.0449] | 0.2183 | 0.7288 | 0.0% | as_written | NVIDIA GeForce RTX 4070 Laptop GPU |
