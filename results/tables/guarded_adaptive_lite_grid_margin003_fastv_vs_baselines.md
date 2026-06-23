| dataset | retention | raw_ece | ts_ece | default_adaptive_ece | conservative_guarded_ece | conservative_minus_ts_ece | conservative_minus_default_adaptive_ece | conservative_guarded_aurc | selected_calibrators |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mmbench_cn | 0.25 | 0.264949 | 0.057191 | 0.051101 | 0.062591 | 0.0054 | 0.011489 | 0.504041 | adaptive_calibprune:1;temperature_scaling:2 |
| mmbench_cn | 0.5 | 0.267475 | 0.050664 | 0.054851 | 0.055086 | 0.004421 | 0.000235 | 0.514169 | adaptive_calibprune:2;temperature_scaling:1 |
| mmbench_cn | 0.75 | 0.263999 | 0.074543 | 0.068705 | 0.070693 | -0.003849 | 0.001988 | 0.491733 | temperature_scaling:3 |
| pope | 0.25 | 0.08597 | 0.056702 | 0.053474 | 0.065166 | 0.008464 | 0.011692 | 0.094886 | adaptive_calibprune:2;temperature_scaling:1 |
| pope | 0.5 | 0.056446 | 0.052006 | 0.046905 | 0.057712 | 0.005706 | 0.010807 | 0.069502 | adaptive_calibprune:1;temperature_scaling:2 |
| pope | 0.75 | 0.042258 | 0.046876 | 0.051349 | 0.052653 | 0.005776 | 0.001304 | 0.059721 | adaptive_calibprune:1;temperature_scaling:2 |
| scienceqa | 0.25 | 0.149425 | 0.084486 | 0.103374 | 0.118868 | 0.034382 | 0.015494 | 0.189971 | adaptive_calibprune:1;temperature_scaling:2 |
| scienceqa | 0.5 | 0.160774 | 0.103231 | 0.109586 | 0.09899 | -0.004241 | -0.010596 | 0.180077 | adaptive_calibprune:1;temperature_scaling:2 |
| scienceqa | 0.75 | 0.161368 | 0.089106 | 0.126782 | 0.092169 | 0.003064 | -0.034613 | 0.186347 | temperature_scaling:3 |
