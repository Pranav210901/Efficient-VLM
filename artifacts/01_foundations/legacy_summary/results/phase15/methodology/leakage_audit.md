# Evaluation and leakage audit

The protocol was frozen before Phase 2. Existing CIFAR-100 test, Pets test, and complete-EuroSAT results are retained as exploratory/development evidence; they are not untouched confirmatory results.

## Going-forward policy

- COCO val2017 is development and expert-selection evidence. It is not described as an untouched final test set.
- CIFAR-100 selection uses a deterministic subset of the official training split; official test is reserved for final reporting.
- Oxford-IIIT Pets selection uses a deterministic partition of trainval; official test is reserved for final reporting.
- EuroSAT uses the fixed stratified manifest under data/splits; its stored test partition is reserved for final reporting.
- Winoground and SugarCrepe are evaluation-only and cannot supply training negatives.

## Untouched external validation

Flickr30k external retrieval data currently available: **no**. Until an external benchmark is installed, no fully untouched external retrieval result is claimed.

## Data-usage table

| dataset         | task           | train_split                            | development_split                                  | selection_split                                    | final_test_split                | used_for_checkpoint_selection   | used_for_expert_selection   | used_for_final_reporting   | historical_result_status                      |
|:----------------|:---------------|:---------------------------------------|:---------------------------------------------------|:---------------------------------------------------|:--------------------------------|:--------------------------------|:----------------------------|:---------------------------|:----------------------------------------------|
| coco            | retrieval      | train2017                              | val2017                                            | val2017                                            | flickr30k_external_if_available | True                            | True                        | False                      | exploratory_development                       |
| cifar100        | classification | official_train_train_partition         | deterministic_official_train_development_partition | deterministic_official_train_development_partition | official_test                   | False                           | True                        | True                       | exploratory_development_test_used             |
| oxford_iiit_pet | classification | deterministic_trainval_train_partition | deterministic_trainval_development_partition       | deterministic_trainval_development_partition       | official_test                   | False                           | True                        | True                       | exploratory_development_test_used             |
| eurosat         | classification | deterministic_stratified_train         | deterministic_stratified_development               | deterministic_stratified_development               | deterministic_stratified_test   | False                           | True                        | True                       | exploratory_development_complete_dataset_used |
| winoground      | compositional  | unavailable                            | unavailable                                        | evaluation_only                                    | evaluation_only                 | False                           | False                       | True                       | unavailable_or_evaluation_only                |
| sugarcrepe      | compositional  | unavailable                            | unavailable                                        | evaluation_only                                    | evaluation_only                 | False                           | False                       | True                       | unavailable_or_evaluation_only                |
