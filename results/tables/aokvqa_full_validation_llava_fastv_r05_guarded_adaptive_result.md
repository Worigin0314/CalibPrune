| setting | selected_calibrator | accuracy | ece | adaptive_ece | aurc | notes |
|---|---|---:|---:|---:|---:|---|
| fastv_raw | none | 0.791266 | 0.060480 | 0.056170 | 0.068675 | raw FastV r=0.5 |
| fastv_temperature_scaling | temperature_scaling | 0.791266 | 0.051043 | 0.048215 | 0.069122 | full n_cal=32 fit |
| fastv_calibprune | calibprune | 0.791266 | 0.051770 | 0.056658 | 0.069161 | full n_cal=32 fit |
| fastv_adaptive_log_margin_default | adaptive_calibprune | 0.791266 | 0.138961 | 0.138184 | 0.071425 | default gamma_l2=0.05 overfits this gate |
| fastv_guarded_adaptive_v025 | temperature_scaling | 0.791266 | 0.031833 | 0.035786 | 0.068998 | guard selects TS on 8-sample validation split |
