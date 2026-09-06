# TokenShift latency profile

This is a profile-only result. No training decision was applied automatically.
The input remained 224×224; 2×2 internal merging reduced 196 patches to 49.

| Arm | mean Q3 (ms) | 95% bootstrap CI | paired Δ vs control (ms) |
|---|---:|---:|---:|
| no_merge | 9.1216 | [9.0455, 9.2309] | — |
| merge_after_block_8 | 8.1377 | [8.0881, 8.1973] | -0.9840 |
| merge_after_block_6 | 7.5153 | [7.4733, 7.5674] | -1.6064 |
