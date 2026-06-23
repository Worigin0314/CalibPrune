| comparison                | metric           |    estimate |      ci_low |     ci_high |   n_samples |   n_resamples |   confidence | interpretation                    |
|:--------------------------|:-----------------|------------:|------------:|------------:|------------:|--------------:|-------------:|:----------------------------------|
| FastV raw - unpruned raw  | accuracy         | -0.00585938 | -0.0175781  |  0.00585938 |        1536 |          1000 |         0.95 | positive_ece_means_drift          |
| FastV raw - unpruned raw  | ece              |  0.0186227  |  0.00286108 |  0.0339741  |        1536 |          1000 |         0.95 | positive_ece_means_drift          |
| FastV raw - unpruned raw  | adaptive_ece     |  0.0148805  | -0.00481445 |  0.029133   |        1536 |          1000 |         0.95 | positive_ece_means_drift          |
| FastV raw - unpruned raw  | aurc             |  0.00914178 |  0.00256963 |  0.0170594  |        1536 |          1000 |         0.95 | positive_ece_means_drift          |
| FastV raw - unpruned raw  | max_softmax_mean |  0.00811772 |  0.00482821 |  0.0115091  |        1536 |          1000 |         0.95 | positive_ece_means_drift          |
| CalibPrune - FastV raw    | accuracy         |  0          |  0          |  0          |        1536 |          1000 |         0.95 | negative_ece_means_improvement    |
| CalibPrune - FastV raw    | ece              | -0.0283022  | -0.0403035  | -0.00320486 |        1536 |          1000 |         0.95 | negative_ece_means_improvement    |
| CalibPrune - FastV raw    | adaptive_ece     | -0.0159624  | -0.0297237  |  0.00142338 |        1536 |          1000 |         0.95 | negative_ece_means_improvement    |
| CalibPrune - FastV raw    | aurc             |  0.00567903 | -0.00107086 |  0.0122657  |        1536 |          1000 |         0.95 | negative_ece_means_improvement    |
| CalibPrune - FastV raw    | max_softmax_mean | -0.0380328  | -0.0396832  | -0.0363882  |        1536 |          1000 |         0.95 | negative_ece_means_improvement    |
| CalibPrune - unpruned raw | accuracy         | -0.00585938 | -0.0175781  |  0.00585938 |        1536 |          1000 |         0.95 | negative_ece_means_below_unpruned |
| CalibPrune - unpruned raw | ece              | -0.00967955 | -0.0225831  |  0.0148943  |        1536 |          1000 |         0.95 | negative_ece_means_below_unpruned |
| CalibPrune - unpruned raw | adaptive_ece     | -0.00108194 | -0.0210378  |  0.0155399  |        1536 |          1000 |         0.95 | negative_ece_means_below_unpruned |
| CalibPrune - unpruned raw | aurc             |  0.0148208  |  0.0057457  |  0.0260473  |        1536 |          1000 |         0.95 | negative_ece_means_below_unpruned |
| CalibPrune - unpruned raw | max_softmax_mean | -0.029915   | -0.033313   | -0.0263049  |        1536 |          1000 |         0.95 | negative_ece_means_below_unpruned |