# FreezeShift same-allocation latency amendment

This corrective measurement was declared after the original gate anomaly. 
The original negative verdict remains preserved; this report repairs the 
cross-session comparison rather than silently replacing it.

| Model | mean Q3 (ms) | paired Δ vs OpenCLIP (ms) | 95% CI | classification |
|---|---:|---:|---:|---|
| M_T1 | 9.1480 | -0.1187 | [-0.1391, -0.0803] | FASTER |
| vision | 9.1683 | -0.0985 | [-0.1396, -0.0180] | FASTER |
| text | 9.1727 | -0.0940 | [-0.1397, -0.0071] | FASTER |
| dual | 9.1655 | -0.1012 | [-0.1449, -0.0160] | FASTER |
| openclip_vit_b32_quickgelu_openai | 9.2667 | — | — | REFERENCE |
