# S06/S07/S09/S10 — Structured pruning, cascades and exits

- MoPE-CLIP: https://openaccess.thecvf.com/content/CVPR2024/html/Lin_MoPE-CLIP_Structured_Pruning_for_Efficient_Vision-Language_Models_with_Module-wise_Pruning_CVPR_2024_paper.html
- Bi-Encoder Cascades: https://openaccess.thecvf.com/content/ICCV2023W/RCV/papers/Honig_Bi-Encoder_Cascades_for_Efficient_Image_Search_ICCVW_2023_paper.pdf
- Multiple-Exit Tuning: https://arxiv.org/abs/2409.13999
- ICAR: https://arxiv.org/abs/2512.15372

CLIP-aware structural pruning, lifetime-cost bi-encoder cascades, parameter-
efficient exits and adaptive image-text retrieval exits all exist. Therefore a
plain early-exit router or cheap-to-expensive retrieval cascade is not novel.
The remaining possible distinction is independent modality routing within a
single compatible space formed from heterogeneous frozen unimodal experts.
