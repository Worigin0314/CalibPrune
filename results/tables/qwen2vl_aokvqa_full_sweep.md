Qwen2-VL A-OKVQA full-validation FastV retention sweep. Acc-minus-chance uses 0.25 for the four-choice option-letter setting.

| Method | r | n | Accuracy | Acc_minus_chance | ECE | AURC | Conf | Residual | Token_down | E2E_s | Peak_alloc_MB | Peak_reserved_MB |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 1.00 | 1145 | 0.2681 | +0.0181 | 0.2023 | 0.7058 | 0.4703 |  | 0.0% | 0.376 | 2210.8 | 2782.0 |
| fastv | 0.25 | 1145 | 0.3109 | +0.0609 | 0.2477 | 0.6490 | 0.5561 | 0.5092 | 75.1% | 0.438 | 2105.6 | 2734.0 |
| fastv | 0.50 | 1145 | 0.3083 | +0.0583 | 0.2653 | 0.6452 | 0.5722 | 0.5604 | 50.1% | 0.426 | 2140.7 | 2808.0 |
| fastv | 0.75 | 1145 | 0.2987 | +0.0487 | 0.2736 | 0.6424 | 0.5719 | 0.5808 | 24.9% | 0.377 | 2175.7 | 2808.0 |
