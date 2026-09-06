# Selection artifact authority

`recipe.json` is the only authoritative locked Wave 0 recipe.

## Permanent student-pair decision

The main study permanently retains **DINOv3 ViT-S/16 +
all-MiniLM-L6-v2** (`student_pair_changed: false`). E5 is not adopted.

- The queue study, 18-cell screen, dose-response curve, v4 teacher caches and
  all 31 claim-bearing runs use MiniLM. Switching pairs would make the
  factorial explain a queue curve measured on a different student.
- E5's 1.56pp corrected-recipe gain carries no headline claim: it does not
  reach OpenCLIP, alter the queue mechanism or change the pair-ranking finding.
- E5 instead demonstrates that the recipe correction generalises across text
  encoders, so the queue effect is not a MiniLM-specific artifact.
- At the end of the project only, ViT-S/16 + E5 may be reported as one clearly
  separated “best available configuration” point on the efficiency frontier.
  It is not part of the main study and must not be run now.

- `finalists.json` records the historical sigmoid finalist selection before
  controls C/D and winner confirmation. It is not the locked recipe.
- `lr_sweep.json` and `lr_sweep_results.csv` record the historical sigmoid LR
  sweep. They remain valid measurements but do not select the locked recipe.
- `screening_results.csv` records the original 18-cell screen. It remains valid
  historical evidence but does not describe the corrected locked recipe.
- `measured_results.csv` is the consolidated measured-results table generated
  by the Wave 0 report.
