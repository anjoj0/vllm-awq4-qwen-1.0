# Nowcast3D Long-Document Sanity Comparison

Baseline label: `bf16_tp8_auto_cross_precision`

| Config | Profile | Suite | Mode | QA | Retention | Needle EM | Citation | Evidence | Source | Abstention | JSON | Mean wall (s) | Mean TTFT (s) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| awq4_rdna3_tp4_cross_precision | 64k | core | evidence | 88.33% | 91.38% | 75.00% | 88.46% | 88.46% | 84.62% | 100.00% | 93.33% | 63.304 | 56.440 |
| bf16_tp8_auto_cross_precision | 64k | core | evidence | 96.67% | 100.00% | 100.00% | 96.15% | 96.15% | 92.31% | 100.00% | 100.00% | 22.674 | 19.253 |
