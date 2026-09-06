# S02/S03/S11 — VLM token routing prior art

- MADTP, CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/html/Cao_MADTP_Multimodal_Alignment-Guided_Dynamic_Token_Pruning_for_Accelerating_Vision-Language_Transformer_CVPR_2024_paper.html
- Patch Ranking, WACV 2025: https://openaccess.thecvf.com/content/WACV2025/html/Wu_Patch_Ranking_Token_Pruning_as_Ranking_Prediction_for_Efficient_CLIP_WACV_2025_paper.html
- Attentive Mask CLIP, ICCV 2023: https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Attentive_Mask_CLIP_ICCV_2023_paper.html

These sources make generic multimodal token pruning a high-collision claim.
Patch Ranking reports removing 40% of CLIP patch tokens with a small average
accuracy loss. Attentive Mask CLIP conditions removal on paired text during
training. MADTP uses multimodal alignment to guide dynamic pruning and includes
retrieval. A new contribution must differ in setting, objective and deployment
semantics, not merely use a new token score.
