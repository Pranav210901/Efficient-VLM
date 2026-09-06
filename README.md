# Efficient Frozen Vision-Language Alignment

Code for a dissertation study on the question:

> Can modern frozen self-supervised vision encoders, combined with sub-5M-parameter
> adaptation, approach compact jointly pretrained vision-language models under a
> fixed inference budget?

The short answer is **no, not in this setting** — but the gap is measurable, the
mechanism behind part of it is identifiable, and the boundary is worth recording.
This repository holds the code that built and measured the two models. Trained
weights live on the Hugging Face Hub (links below).

---

## The two models

Both share a frozen DINOv3 ViT-S/16 vision tower at 224 px [1] and a frozen
all-MiniLM-L6-v2 text tower, with residual projections into a shared normalised
384-d embedding space. MobileCLIP2-S0 [2] supplies a distillation signal
**during training only**; it is not part of either deployed model. Training uses
queue-free InfoNCE over all COCO captions [3] at batch 1024.

**M_T1** — fully frozen backbones. The only trainable parts are a two-block,
256-d learned vision-token aggregator (`C4`), a 128-d four-head learned-query
text aggregator, and the two projection heads.

**FreezeShift dual** — identical, plus rank-128 LoRA [4] on the attention
projections of the last four blocks of *both* towers.

| | M_T1 | FreezeShift dual |
|---|---:|---:|
| Deployed parameters | 47,196,549 | 49,162,629 |
| Trainable at inference | 2,896,389 | 4,862,469 |
| Trainable share | 6.1% | 9.9% |

The trainable set is exactly: token aggregator (1,252,225), image projection
(739,201), text projection (739,201), text token aggregator (165,761), logit
scale (1) — plus, for FreezeShift, vision LoRA (1,179,648) and text LoRA
(786,432). Everything else is byte-identical to the upstream encoders.

## Results

**These are two different splits and must not be read as one number.**
Flickr30k *validation* guided configuration and checkpoint selection, so it is
optimistically biased. Flickr30k *test* was opened once, after the models were
frozen, with no training of any kind.

### Sealed test — Flickr30k, mean bidirectional R@1, three seeds

| Model | Test R@1 | Deployed params |
|---|---:|---:|
| M_T1 (fully frozen) | 52.90 ± 0.87 | 47.2M |
| FreezeShift dual | 62.20 ± 0.45 | 49.2M |
| OpenCLIP ViT-B/32 [5] | 68.22 | 151.3M |
| MobileCLIP2-S0 [2] | 78.25 | 74.8M |
| SigLIP2 ViT-B/32 [6] | 80.46 | 376.9M |

FreezeShift retains 79.5% of MobileCLIP2-S0's split-matched score at 65.7% of
its size. It does not reach it, and neither model reaches OpenCLIP.

### Development validation, for selection provenance only

M_T1 54.487%, FreezeShift dual 63.416%, OpenCLIP 69.941% — under
Flickr-validation epoch selection. The **published checkpoints** export the
COCO-dev-selected epoch instead, which scores 54.36% / 63.04% on the same
validation split. Quote the sealed-test figures for any claim about
performance.

### Zero-shot classification (sealed)

Transfer is weak and this is a real limitation, not a footnote:

| | CIFAR-100 | Oxford-IIIT Pet | EuroSAT |
|---|---:|---:|---:|
| M_T1 | 36.39 | 9.90 | 24.41 |
| FreezeShift dual | 37.85 | 7.96 | 21.98 |
| OpenCLIP ViT-B/32 | 64.58 | 83.10 | 34.44 |

Retrieval alignment on COCO-style captions did not transfer to fine-grained
category recognition. Pet classification is near chance-adjacent.

### Latency

Same-allocation, batch 64, native bf16, 100 timed iterations per repetition on
an RTX PRO 6000 Blackwell. Mean full-stack Q3: M_T1 9.148 ms, FreezeShift
9.166 ms, against a locally measured OpenCLIP reference — 0.119 ms and 0.101 ms
faster respectively (paired 95% bootstrap CI for FreezeShift: −0.145 to
−0.016 ms). The speed advantage is real but small; the accuracy deficit is
large. That trade is the finding.

## What did not work

Recorded because negative results are results:

- **Cross-batch memory queues.** Stale negatives became harmful after a single
  optimiser step, and the damage was modality-asymmetric — the
  overlap-restricted text response was roughly 7.2× steeper than the image
  response. Matched embedding drift did not imply matched retrieval damage.
  This is the study's clearest mechanistic finding and it argues against the
  XBM-style [7] assumption in this setting.
- **Token merging (TokenShift).** ToMe-style [8] 2×2 merging after block 8 or 6
  cut latency but cost 12.58pp (frozen parent) and 5.69pp (dual-LoRA parent).
  Neither depth was promoted.
- **Guarded hard-negative mining.** Stopped by its own pre-registered safety
  gate as infeasible.
- **Learning-rate retuning** at the final stage did not improve the selected
  model.

## Literature context

**Frozen-encoder alignment.** Locking one tower is established: LiT [9] froze
the image tower and trained text against it. Recent work aligns *both* frozen
towers with small trainable bridges — FreezeAlign [10], SAIL [11], ShareLock
[12], and STRUCTURE [13], which exploits geometric structure when paired data
is scarce. This study sits in that family and adds a specific negative boundary:
under a fixed latency budget and sub-5M adaptation, frozen alignment did not
close the gap to compact jointly pretrained models.

**Token aggregation.** Replacing mean-pooling with learned queries traces to
Set Transformer [14]; attentive probing was recently revisited for efficiency
by Psomas et al. [15], and SigLIP2 [6] uses a MAP head. The `C4` aggregator here
is a straightforward instance, not a novel mechanism — its contribution is the
measured scaling behaviour, which saturated well below the 5M budget.

**Memory queues.** MoCo [16] introduced the momentum queue; XBM [17] applied
cross-batch memory to metric learning, and adaptive normalisation [18] and hard
negative mixing [19] address the staleness it induces. The queue results here
are consistent with staleness being the dominant failure mode rather than the
loss family.

**Distillation and compact VLMs.** TinyCLIP [20], CLIP-KD [21], and MobileCLIP
[22] / MobileCLIP2 [2] show that compact CLIP-family models reach strong
retrieval through large-scale reinforced training. MobileCLIP2-S0 is the
primary compact reference here precisely because it is the honest comparison —
and it wins.

**The modality gap** [23] and reproducibility norms in scaling studies [5]
inform the evaluation protocol: three seeds, pre-registered predictions,
sealed test, and explicit disclosure of selection effects.

## Reproduction

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-lock.txt
```

Datasets are not redistributed. COCO and Flickr30k must be acquired locally;
`src/alignment_v3/splits.py` regenerates the seeded split definitions
deterministically. Set `ALIGNMENT_DATA_ROOT` if images live outside the repo.

```bash
# train M_T1
python -m src.alignment_v3.text_aggregation_study --pipeline configs/text_aggregation_study/pipeline_m_t1.yaml

# train FreezeShift dual
python -m freezeshift.runner --pipeline freezeshift/configs/pipeline_dual.yaml

# sealed evaluation (expects checkpoints/ populated from the Hub)
python -m src.alignment_v3.final_zero_shot evaluate --pipeline configs/final_zero_shot/pipeline.yaml
python -m src.alignment_v3.final_zero_shot report   --pipeline configs/final_zero_shot/pipeline.yaml
```

`slurm/` and `freezeshift/slurm/` hold the batch templates that actually ran the
study, including GPU type, allocation, and array layout.

### Tests

```bash
.venv/bin/python -m pytest tests/ freezeshift/tests/
```

17 of 18 tests pass without weights. `test_student_exports_are_inference_only_and_fingerprint_valid`
needs `checkpoints/` populated from the model repos above; all 18 pass once it is.

Configs use `extends:` inheritance; the M_T1 pipeline resolves through the
token-aggregator and token-projection lineage back to `configs/alignment_v3/base.yaml`.

## Models

- `ppokhrel2109/mt1-frozen-vlm` — M_T1, seed 43
- `ppokhrel2109/freezeshift-dual-vlm` — FreezeShift dual, seed 44

Seeds were selected on **validation**, never on test. Each repo carries full
weights and an adapter-only variant.

## Licensing and attribution

**This code** is released under the MIT licence (`LICENSE`).

**The models it builds are research artifacts and are not for commercial use.**
Components carry their own terms, and they are not all permissive:

| Component | Terms |
|---|---|
| DINOv3 ViT-S/16 (vision tower) | Meta **DINOv3 License** — see `LICENSE-DINOv3.md` |
| all-MiniLM-L6-v2 (text tower) | Apache-2.0 |
| MobileCLIP2-S0 (training teacher) | Apple ML Research Model License — **research purposes only** |
| timm, OpenCLIP | Apache-2.0 / MIT |

Under the DINOv3 License, distributing DINO Materials "or any derivative works
thereof" to a third party requires doing so under that Agreement's terms and
providing a copy of it — hence `LICENSE-DINOv3.md` here and in the model repos.
Research publications using DINO Materials must acknowledge them. The Agreement
also prohibits military, weapons, espionage, and nuclear applications, and
requires Trade Control compliance.

MobileCLIP2-S0 was used **as a frozen teacher during training only**; its
weights are not redistributed. Apple's licence permits use "exclusively for
Research Purposes" and excludes commercial exploitation. It defines Model
Derivatives as artifacts created by modifying, retraining, or fine-tuning the
Apple model — which these separately-architected students are not — but it does
not explicitly address distillation. Treat the released models as research-only.

**Datasets** are not redistributed: COCO (CC BY 4.0 annotations, images under
Flickr terms), Flickr30k, CC3M, CIFAR-100, Oxford-IIIT Pet, EuroSAT. Acquire
each under its own terms.

## References

[1] Siméoni et al. DINOv3. 2025.
[2] Faghri et al. MobileCLIP2: Improving Multi-Modal Reinforced Training. 2025.
[3] Lin et al. Microsoft COCO: Common Objects in Context. ECCV 2014.
[4] Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
[5] Cherti et al. Reproducible Scaling Laws for Contrastive Language-Image Learning. CVPR 2023.
[6] Tschannen et al. SigLIP 2. 2025.
[7] Wang et al. Cross-Batch Memory for Embedding Learning. CVPR 2020.
[8] Bolya et al. Token Merging: Your ViT But Faster. 2022.
[9] Zhai et al. LiT: Zero-Shot Transfer with Locked-image Text Tuning. CVPR 2022.
[10] Maniparambil et al. Harnessing Frozen Unimodal Encoders for Flexible Multimodal Alignment. CVPR 2025.
[11] Zhang, Yang and Agrawal. Assessing and Learning Alignment of Unimodal Vision and Language Models. CVPR 2025.
[12] Ruthardt et al. Better Language Models Exhibit Higher Visual Alignment. 2026.
[13] Gröger et al. With Limited Data for Multimodal Alignment, Let the STRUCTURE Guide You. NeurIPS 2025.
[14] Lee et al. Set Transformer. ICML 2019.
[15] Psomas et al. Attention, Please! Revisiting Attentive Probing Through the Lens of Efficiency. ICLR 2026.
[16] He et al. Momentum Contrast for Unsupervised Visual Representation Learning. CVPR 2020.
[17] Wang et al. Cross-Batch Memory for Embedding Learning. CVPR 2020.
[18] Ajanthan et al. Adaptive Cross Batch Normalization for Metric Learning. 2023.
[19] Kalantidis et al. Hard Negative Mixing for Contrastive Learning. NeurIPS 2020.
[20] Wu et al. TinyCLIP: CLIP Distillation via Affinity Mimicking and Weight Inheritance. ICCV 2023.
[21] Yang et al. CLIP-KD: An Empirical Study of CLIP Model Distillation. CVPR 2024.
[22] Vasu et al. MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training. CVPR 2024.
[23] Liang et al. Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning. NeurIPS 2022.
