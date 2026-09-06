# Untouched compositional evaluation

No learning, prompt tuning, checkpoint selection or threshold tuning was performed.

## Primary: SugarCrepe overall accuracy

| model_id                          | label                             | kind      | dataset    | metric           |   n |     mean |          sd |      min |      max |
|:----------------------------------|:----------------------------------|:----------|:-----------|:-----------------|----:|---------:|------------:|---------:|---------:|
| mt1_strict_frozen                 | M_T1 strict-frozen C4             | student   | sugarcrepe | overall_accuracy |   3 | 0.770071 | 0.00234037  | 0.767408 | 0.771801 |
| mt1_dual_lora                     | M_T1 C4 dual-LoRA PEFT            | student   | sugarcrepe | overall_accuracy |   3 | 0.783828 | 0.000768678 | 0.783384 | 0.784716 |
| openclip_vit_b32_quickgelu_openai | openclip_vit_b32_quickgelu_openai | reference | sugarcrepe | overall_accuracy |   1 | 0.765544 | 0           | 0.765544 | 0.765544 |
| mobileclip2_s0_dfndr2b            | mobileclip2_s0_dfndr2b            | reference | sugarcrepe | overall_accuracy |   1 | 0.824258 | 0           | 0.824258 | 0.824258 |
| siglip2_vit_b32_256_webli         | siglip2_vit_b32_256_webli         | reference | sugarcrepe | overall_accuracy |   1 | 0.850353 | 0           | 0.850353 | 0.850353 |

## Complete diagnostics

| model_id                          | label                             | kind      | dataset    | metric               |   n |      mean |          sd |       min |       max |
|:----------------------------------|:----------------------------------|:----------|:-----------|:---------------------|----:|----------:|------------:|----------:|----------:|
| mt1_strict_frozen                 | M_T1 strict-frozen C4             | student   | sugarcrepe | overall_accuracy     |   3 | 0.770071  | 0.00234037  | 0.767408  | 0.771801  |
| mt1_strict_frozen                 | M_T1 strict-frozen C4             | student   | sugarcrepe | positive_score       |   3 | 0.556424  | 0.000504672 | 0.555882  | 0.55688   |
| mt1_strict_frozen                 | M_T1 strict-frozen C4             | student   | sugarcrepe | negative_score       |   3 | 0.5191    | 0.000423723 | 0.518701  | 0.519545  |
| mt1_strict_frozen                 | M_T1 strict-frozen C4             | student   | sugarcrepe | score_margin         |   3 | 0.0373234 | 0.000136528 | 0.0371812 | 0.0374534 |
| mt1_strict_frozen                 | M_T1 strict-frozen C4             | student   | sugarcrepe | accuracy/add_att     |   3 | 0.645472  | 0.0154518   | 0.635838  | 0.663295  |
| mt1_strict_frozen                 | M_T1 strict-frozen C4             | student   | sugarcrepe | accuracy/add_obj     |   3 | 0.79195   | 0.00431049  | 0.788555  | 0.796799  |
| mt1_strict_frozen                 | M_T1 strict-frozen C4             | student   | sugarcrepe | accuracy/replace_att |   3 | 0.793993  | 0.0110874   | 0.781726  | 0.803299  |
| mt1_strict_frozen                 | M_T1 strict-frozen C4             | student   | sugarcrepe | accuracy/replace_obj |   3 | 0.91707   | 0.00631979  | 0.909806  | 0.921308  |
| mt1_strict_frozen                 | M_T1 strict-frozen C4             | student   | sugarcrepe | accuracy/replace_rel |   3 | 0.723566  | 0.00925523  | 0.714082  | 0.732575  |
| mt1_strict_frozen                 | M_T1 strict-frozen C4             | student   | sugarcrepe | accuracy/swap_att    |   3 | 0.598098  | 0.00625125  | 0.593093  | 0.605105  |
| mt1_strict_frozen                 | M_T1 strict-frozen C4             | student   | sugarcrepe | accuracy/swap_obj    |   3 | 0.604082  | 0.0147165   | 0.587755  | 0.616327  |
| mt1_dual_lora                     | M_T1 C4 dual-LoRA PEFT            | student   | sugarcrepe | overall_accuracy     |   3 | 0.783828  | 0.000768678 | 0.783384  | 0.784716  |
| mt1_dual_lora                     | M_T1 C4 dual-LoRA PEFT            | student   | sugarcrepe | positive_score       |   3 | 0.577315  | 0.00174495  | 0.575394  | 0.578803  |
| mt1_dual_lora                     | M_T1 C4 dual-LoRA PEFT            | student   | sugarcrepe | negative_score       |   3 | 0.534625  | 0.00171722  | 0.532685  | 0.535951  |
| mt1_dual_lora                     | M_T1 C4 dual-LoRA PEFT            | student   | sugarcrepe | score_margin         |   3 | 0.0426899 | 0.000172298 | 0.0425088 | 0.0428518 |
| mt1_dual_lora                     | M_T1 C4 dual-LoRA PEFT            | student   | sugarcrepe | accuracy/add_att     |   3 | 0.61368   | 0.00584026  | 0.606936  | 0.617052  |
| mt1_dual_lora                     | M_T1 C4 dual-LoRA PEFT            | student   | sugarcrepe | accuracy/add_obj     |   3 | 0.80375   | 0.00195998  | 0.802619  | 0.806014  |
| mt1_dual_lora                     | M_T1 C4 dual-LoRA PEFT            | student   | sugarcrepe | accuracy/replace_att |   3 | 0.823181  | 0.00638735  | 0.817259  | 0.829949  |
| mt1_dual_lora                     | M_T1 C4 dual-LoRA PEFT            | student   | sugarcrepe | accuracy/replace_obj |   3 | 0.927966  | 0.00242132  | 0.925545  | 0.930387  |
| mt1_dual_lora                     | M_T1 C4 dual-LoRA PEFT            | student   | sugarcrepe | accuracy/replace_rel |   3 | 0.727122  | 0.0048413   | 0.723329  | 0.732575  |
| mt1_dual_lora                     | M_T1 C4 dual-LoRA PEFT            | student   | sugarcrepe | accuracy/swap_att    |   3 | 0.665666  | 0.0213051   | 0.642643  | 0.684685  |
| mt1_dual_lora                     | M_T1 C4 dual-LoRA PEFT            | student   | sugarcrepe | accuracy/swap_obj    |   3 | 0.644898  | 0.0389363   | 0.604082  | 0.681633  |
| openclip_vit_b32_quickgelu_openai | openclip_vit_b32_quickgelu_openai | reference | sugarcrepe | overall_accuracy     |   1 | 0.765544  | 0           | 0.765544  | 0.765544  |
| openclip_vit_b32_quickgelu_openai | openclip_vit_b32_quickgelu_openai | reference | sugarcrepe | positive_score       |   1 | 0.309062  | 0           | 0.309062  | 0.309062  |
| openclip_vit_b32_quickgelu_openai | openclip_vit_b32_quickgelu_openai | reference | sugarcrepe | negative_score       |   1 | 0.292326  | 0           | 0.292326  | 0.292326  |
| openclip_vit_b32_quickgelu_openai | openclip_vit_b32_quickgelu_openai | reference | sugarcrepe | score_margin         |   1 | 0.0167356 | 0           | 0.0167356 | 0.0167356 |
| openclip_vit_b32_quickgelu_openai | openclip_vit_b32_quickgelu_openai | reference | sugarcrepe | accuracy/add_att     |   1 | 0.683526  | 0           | 0.683526  | 0.683526  |
| openclip_vit_b32_quickgelu_openai | openclip_vit_b32_quickgelu_openai | reference | sugarcrepe | accuracy/add_obj     |   1 | 0.770611  | 0           | 0.770611  | 0.770611  |
| openclip_vit_b32_quickgelu_openai | openclip_vit_b32_quickgelu_openai | reference | sugarcrepe | accuracy/replace_att |   1 | 0.800761  | 0           | 0.800761  | 0.800761  |
| openclip_vit_b32_quickgelu_openai | openclip_vit_b32_quickgelu_openai | reference | sugarcrepe | accuracy/replace_obj |   1 | 0.909806  | 0           | 0.909806  | 0.909806  |
| openclip_vit_b32_quickgelu_openai | openclip_vit_b32_quickgelu_openai | reference | sugarcrepe | accuracy/replace_rel |   1 | 0.697013  | 0           | 0.697013  | 0.697013  |
| openclip_vit_b32_quickgelu_openai | openclip_vit_b32_quickgelu_openai | reference | sugarcrepe | accuracy/swap_att    |   1 | 0.636637  | 0           | 0.636637  | 0.636637  |
| openclip_vit_b32_quickgelu_openai | openclip_vit_b32_quickgelu_openai | reference | sugarcrepe | accuracy/swap_obj    |   1 | 0.612245  | 0           | 0.612245  | 0.612245  |
| mobileclip2_s0_dfndr2b            | mobileclip2_s0_dfndr2b            | reference | sugarcrepe | overall_accuracy     |   1 | 0.824258  | 0           | 0.824258  | 0.824258  |
| mobileclip2_s0_dfndr2b            | mobileclip2_s0_dfndr2b            | reference | sugarcrepe | positive_score       |   1 | 0.301315  | 0           | 0.301315  | 0.301315  |
| mobileclip2_s0_dfndr2b            | mobileclip2_s0_dfndr2b            | reference | sugarcrepe | negative_score       |   1 | 0.274621  | 0           | 0.274621  | 0.274621  |
| mobileclip2_s0_dfndr2b            | mobileclip2_s0_dfndr2b            | reference | sugarcrepe | score_margin         |   1 | 0.0266946 | 0           | 0.0266946 | 0.0266946 |
| mobileclip2_s0_dfndr2b            | mobileclip2_s0_dfndr2b            | reference | sugarcrepe | accuracy/add_att     |   1 | 0.786127  | 0           | 0.786127  | 0.786127  |
| mobileclip2_s0_dfndr2b            | mobileclip2_s0_dfndr2b            | reference | sugarcrepe | accuracy/add_obj     |   1 | 0.872454  | 0           | 0.872454  | 0.872454  |
| mobileclip2_s0_dfndr2b            | mobileclip2_s0_dfndr2b            | reference | sugarcrepe | accuracy/replace_att |   1 | 0.85533   | 0           | 0.85533   | 0.85533   |
| mobileclip2_s0_dfndr2b            | mobileclip2_s0_dfndr2b            | reference | sugarcrepe | accuracy/replace_obj |   1 | 0.943705  | 0           | 0.943705  | 0.943705  |
| mobileclip2_s0_dfndr2b            | mobileclip2_s0_dfndr2b            | reference | sugarcrepe | accuracy/replace_rel |   1 | 0.701991  | 0           | 0.701991  | 0.701991  |
| mobileclip2_s0_dfndr2b            | mobileclip2_s0_dfndr2b            | reference | sugarcrepe | accuracy/swap_att    |   1 | 0.714715  | 0           | 0.714715  | 0.714715  |
| mobileclip2_s0_dfndr2b            | mobileclip2_s0_dfndr2b            | reference | sugarcrepe | accuracy/swap_obj    |   1 | 0.620408  | 0           | 0.620408  | 0.620408  |
| siglip2_vit_b32_256_webli         | siglip2_vit_b32_256_webli         | reference | sugarcrepe | overall_accuracy     |   1 | 0.850353  | 0           | 0.850353  | 0.850353  |
| siglip2_vit_b32_256_webli         | siglip2_vit_b32_256_webli         | reference | sugarcrepe | positive_score       |   1 | 0.144182  | 0           | 0.144182  | 0.144182  |
| siglip2_vit_b32_256_webli         | siglip2_vit_b32_256_webli         | reference | sugarcrepe | negative_score       |   1 | 0.122319  | 0           | 0.122319  | 0.122319  |
| siglip2_vit_b32_256_webli         | siglip2_vit_b32_256_webli         | reference | sugarcrepe | score_margin         |   1 | 0.0218629 | 0           | 0.0218629 | 0.0218629 |
| siglip2_vit_b32_256_webli         | siglip2_vit_b32_256_webli         | reference | sugarcrepe | accuracy/add_att     |   1 | 0.82659   | 0           | 0.82659   | 0.82659   |
| siglip2_vit_b32_256_webli         | siglip2_vit_b32_256_webli         | reference | sugarcrepe | accuracy/add_obj     |   1 | 0.901067  | 0           | 0.901067  | 0.901067  |
| siglip2_vit_b32_256_webli         | siglip2_vit_b32_256_webli         | reference | sugarcrepe | accuracy/replace_att |   1 | 0.860406  | 0           | 0.860406  | 0.860406  |
| siglip2_vit_b32_256_webli         | siglip2_vit_b32_256_webli         | reference | sugarcrepe | accuracy/replace_obj |   1 | 0.952179  | 0           | 0.952179  | 0.952179  |
| siglip2_vit_b32_256_webli         | siglip2_vit_b32_256_webli         | reference | sugarcrepe | accuracy/replace_rel |   1 | 0.737553  | 0           | 0.737553  | 0.737553  |
| siglip2_vit_b32_256_webli         | siglip2_vit_b32_256_webli         | reference | sugarcrepe | accuracy/swap_att    |   1 | 0.765766  | 0           | 0.765766  | 0.765766  |
| siglip2_vit_b32_256_webli         | siglip2_vit_b32_256_webli         | reference | sugarcrepe | accuracy/swap_obj    |   1 | 0.64898   | 0           | 0.64898   | 0.64898   |
