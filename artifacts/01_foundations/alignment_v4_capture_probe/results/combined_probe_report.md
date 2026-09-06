# Alignment v4 — distillation capture probe

**Main dissertation question:** Can modern frozen SSL vision encoders, combined with compatibility-based pair selection and sub-5M-parameter adaptation, approach compact jointly pretrained vision-language models under a fixed inference budget?

**Overall probe decision:** `INCONCLUSIVE`

| teacher_label   |   baseline_R@1 |   distilled_R@1 |   teacher_R@1 |   delta_R@1 |   capture_fraction | decision            |
|:----------------|---------------:|----------------:|--------------:|------------:|-------------------:|:--------------------|
| MobileCLIP2-S0  |       0.133036 |        0.223578 |      0.52982  |   0.0905425 |           0.228191 | INCONCLUSIVE_MIDDLE |
| SigLIP2-B/32    |       0.133036 |        0.232916 |      0.568807 |   0.09988   |           0.229203 | INCONCLUSIVE_MIDDLE |

This is a diagnostic decision, not a positive answer to the dissertation question. Teacher anchors are sealed COCO-val measurements while student probe results use the disjoint COCO-dev split, so capture fractions are directional and the absolute student thresholds are the primary gate.
