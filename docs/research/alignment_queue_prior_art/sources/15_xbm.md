# Cross-Batch Memory

- Paper: *Cross-Batch Memory for Embedding Learning*
- Venue: CVPR 2020
- URL: https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html
- Verification: full paper and supplementary material inspected.
- Verified evidence: FIFO historical-embedding memory, explicit drift at
  intervals 10/100/1,000, a memory-ratio sweep at fixed batch, and a batch-size
  study. It reports saturation at moderate memory sizes and reduced batch-size
  sensitivity with XBM.
- Short excerpt: “features drift exceptionally slow.”
- Relevance: closest queue precedent and strongest adversarial source against
  claiming feature staleness, memory-ratio ablation or batch/memory interaction
  itself as new. Its regime is supervised unimodal metric learning and it does
  not separate two modality-specific queues.
