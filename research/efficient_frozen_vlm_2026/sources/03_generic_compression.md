# S04/S05/S08 — Generic compression baselines

- ToMe: https://openreview.net/forum?id=JroZRaRw7Eu
- DynamicViT: https://proceedings.neurips.cc/paper/2021/hash/747d3443e319a22747fbb873e8b2f9f2-Abstract.html
- Matryoshka Representation Learning: https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html

Training-free token merging, learned token pruning and nested embedding
dimensions are mature primitives. They are required baselines, not defensible
standalone contributions. Matryoshka dimensions primarily reduce storage and
search cost, not encoder latency.
