# TokenShift frozen-vs-LoRA result

No arm was selected automatically. Flickr30k test remained sealed.

| Arm | validation mean R@1 | gain vs parent (pp) | mean Q3 (ms) | parameters |
|---|---:|---:|---:|---:|
| frozen_block8 | 41.907% | -12.581 | 8.2041 | 2,896,389 |
| frozen_block6 | 38.537% | -15.950 | 7.5730 | 2,896,389 |
| dual_block8 | 57.725% | -5.690 | 8.2027 | 4,862,469 |
| dual_block6 | 52.554% | -10.861 | 7.5555 | 4,862,469 |
