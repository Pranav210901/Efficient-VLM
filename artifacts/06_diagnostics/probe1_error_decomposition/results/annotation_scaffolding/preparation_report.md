# Probe 1 human-annotation preparation

Preparation only: no labels were assigned, no rows were resampled, no source export was changed, and the blinded audit was not unblinded.

## Files and integrity

| Role | Absolute path | Rows | SHA-256 |
|---|---|---:|---|
| Blinded audit source | `/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm/results/probe1_error_decomposition/audit_sample_blinded.csv` | 100 | `d93fe93cc84bbff18b6b9fb551b6340320ad840fa1ff1d606873c03459218ece` |
| Failure source | `/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm/results/probe1_error_decomposition/failure_sample.csv` | 50 | `991ee92c44c7d7958cc2d1535ec5cc4ccf7fd7b281e1034d5497fe92c4847c50` |
| Protected blinding key | `/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm/results/probe1_error_decomposition/audit_key.csv` | 100 | `3108102bc7e22bf5e4ab9b5e1bfd6e5057db1d4c78f1cba3dbc122393fb74ae1` |
| Audit annotation template | `/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm/results/probe1_error_decomposition/annotation_scaffolding/audit_sample_annotation_template.csv` | 100 | `f2bc67934d29d185692c5ca7478cd515c464954437db991ffae000027393a9d6` |
| Failure annotation template | `/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm/results/probe1_error_decomposition/annotation_scaffolding/failure_sample_annotation_template.csv` | 50 | `601dd46a405e0e00d08b646cc609fb4de7f34b48470ede9dd0ef2c11720a2c26` |

## 1. Schemas

### `audit_sample_blinded.csv`

| Column | dtype | Description |
|---|---|---|
| `audit_id` | string | Opaque audit-row identifier used to join to the separately protected blinding key. |
| `query_caption` | string | Caption query shown to the human annotator. |
| `candidate_A` | string/path | Blinded candidate image path labelled A. |
| `candidate_B` | string/path | Blinded candidate image path labelled B. |
| `candidate_C` | string/path | Blinded candidate image path labelled C. |
| `candidate_D` | string/path | Blinded candidate image path labelled D. |
| `candidate_E` | string/path | Blinded candidate image path labelled E. |
| `rubric_categories` | JSON string | Pre-existing serialized rubric suggestions carried by the export; not human annotations. |
| `candidate_F` | nullable string/path | Optional sixth blinded candidate, present when the positive item was not already among the five retrieved candidates. |

### `failure_sample.csv`

| Column | dtype | Description |
|---|---|---|
| `direction` | categorical string | Retrieval direction; the exported rows are t2i because i2t did not meet the 50-item floor. |
| `query_id` | string | Canonical caption identifier for the t2i query. |
| `query` | string | Human-readable query caption. |
| `positive_id` | string | Canonical identifier of the annotated positive image. |
| `positive_rank_seed42` | integer | Positive image rank under display seed 42. |
| `positive_item` | JSON string | Annotated positive image identifier and path. |
| `ranks_all_seeds` | JSON string | Positive rank for each of C4 seeds 42, 43 and 44. |
| `top10_seed42` | JSON string | Ordered top-10 retrieved image identifiers for display seed 42. |
| `retrieved_items_seed42` | JSON string | Ordered top-10 retrieved image identifiers and paths for display seed 42. |
| `predeclared_categories` | JSON string | Pre-existing serialized failure-category suggestions; no category has been assigned. |

### Blinding contract

The blinded CSV exposes only `audit_id`, the query caption, randomly lettered candidate image paths A–F, and the serialized pre-existing rubric suggestions. It withholds `query_id`, `positive_id`, `positive_rank`, retrieval `partition`, TF-IDF bin, BGE bin, and the mapping from candidate letters to canonical image IDs.

The reproducibility key is stored separately at `/mnt/fast/nobackup/scratch4weeks/pp01184/alignment_vlm/results/probe1_error_decomposition/audit_key.csv` with columns `audit_id, query_id, positive_id, positive_rank, partition, tfidf_cross_bin, bge_cross_bin, candidate_mapping`. This preparation read it only to produce aggregate bin counts; it never joined key rows to blinded rows and never copied key fields into an annotation template.

## 2. Row counts and sampling

- Blinded ambiguity audit: **100 rows**. Source population: seed-42 C4 t2i queries. It initially sampled up to 11 rows from each of 9 `partition × TF-IDF-cross-bin` cells using seed `20260731`, redistributed deficits by largest remaining capacity, and assigned the final slot to the largest remaining stratum.
- Failure export: **50 rows**, all t2i. Direction was the only sampling stratum: i2t had 47 all-three-seed outside-top-10 queries and exported 0 because it missed the 50-row floor; t2i had 533 and sampled 50 uniformly without replacement using seed `20260732`. There was no further ambiguity-bin or failure-category stratification.

Audit sample population before redistribution:

```json
{
  "rank_1|low": 797,
  "rank_1|medium": 789,
  "rank_1|high": 765,
  "rank_2_10|low": 610,
  "rank_2_10|medium": 630,
  "rank_2_10|high": 634,
  "outside_top_10|low": 283,
  "outside_top_10|medium": 271,
  "outside_top_10|high": 291
}
```

## 3. Row examples

All fields are printed. Random displays use fixed seed `20260801`; selected zero-based source-row indices are reported.

### Blinded audit — first five

```csv
audit_id,query_caption,candidate_A,candidate_B,candidate_C,candidate_D,candidate_E,rubric_categories,candidate_F
audit_000,A female UPS worker stacks boxes.,data/flickr30k/validation_images/1520.jpg,data/flickr30k/validation_images/1365.jpg,data/flickr30k/validation_images/17914.jpg,data/flickr30k/validation_images/1716.jpg,data/flickr30k/validation_images/13261.jpg,"[""annotated_image_uniquely_correct"", ""retrieved_alternative_equally_plausible"", ""retrieved_alternative_related_but_incorrect"", ""query_intrinsically_ambiguous_or_generic"", ""insufficient_evidence""]",
audit_001,Two Indian children in formal costume happily performing a ritual dance.,data/flickr30k/validation_images/30369.jpg,data/flickr30k/validation_images/18956.jpg,data/flickr30k/validation_images/28126.jpg,data/flickr30k/validation_images/29492.jpg,data/flickr30k/validation_images/23998.jpg,"[""annotated_image_uniquely_correct"", ""retrieved_alternative_equally_plausible"", ""retrieved_alternative_related_but_incorrect"", ""query_intrinsically_ambiguous_or_generic"", ""insufficient_evidence""]",
audit_002,Two children jumping on a screened in blue and black trampoline while outside surrounded by trees.,data/flickr30k/validation_images/26186.jpg,data/flickr30k/validation_images/15582.jpg,data/flickr30k/validation_images/9907.jpg,data/flickr30k/validation_images/19165.jpg,data/flickr30k/validation_images/30177.jpg,"[""annotated_image_uniquely_correct"", ""retrieved_alternative_equally_plausible"", ""retrieved_alternative_related_but_incorrect"", ""query_intrinsically_ambiguous_or_generic"", ""insufficient_evidence""]",
audit_003,A man on a bicycle rides amongst automobile traffic.,data/flickr30k/validation_images/27310.jpg,data/flickr30k/validation_images/12449.jpg,data/flickr30k/validation_images/22872.jpg,data/flickr30k/validation_images/28590.jpg,data/flickr30k/validation_images/15204.jpg,"[""annotated_image_uniquely_correct"", ""retrieved_alternative_equally_plausible"", ""retrieved_alternative_related_but_incorrect"", ""query_intrinsically_ambiguous_or_generic"", ""insufficient_evidence""]",
audit_004,This old Asian woman rests and fans herself on an apparently hot day in the city.,data/flickr30k/validation_images/10449.jpg,data/flickr30k/validation_images/24490.jpg,data/flickr30k/validation_images/22916.jpg,data/flickr30k/validation_images/25007.jpg,data/flickr30k/validation_images/22889.jpg,"[""annotated_image_uniquely_correct"", ""retrieved_alternative_equally_plausible"", ""retrieved_alternative_related_but_incorrect"", ""query_intrinsically_ambiguous_or_generic"", ""insufficient_evidence""]",
```

### Blinded audit — random five (indices [35, 69, 70, 85, 90])

```csv
audit_id,query_caption,candidate_A,candidate_B,candidate_C,candidate_D,candidate_E,rubric_categories,candidate_F
audit_035,A mid-age women busy in making woolen sweaters sitting in her stall.,data/flickr30k/validation_images/27357.jpg,data/flickr30k/validation_images/12507.jpg,data/flickr30k/validation_images/17030.jpg,data/flickr30k/validation_images/26141.jpg,data/flickr30k/validation_images/4768.jpg,"[""annotated_image_uniquely_correct"", ""retrieved_alternative_equally_plausible"", ""retrieved_alternative_related_but_incorrect"", ""query_intrinsically_ambiguous_or_generic"", ""insufficient_evidence""]",data/flickr30k/validation_images/16632.jpg
audit_069,"Something seems to be better than nothing as a young adult male plays a violin near the entrance of a store with his violin case open ready to accept the ""tips"" of his fans.",data/flickr30k/validation_images/22187.jpg,data/flickr30k/validation_images/4561.jpg,data/flickr30k/validation_images/27707.jpg,data/flickr30k/validation_images/16541.jpg,data/flickr30k/validation_images/17212.jpg,"[""annotated_image_uniquely_correct"", ""retrieved_alternative_equally_plausible"", ""retrieved_alternative_related_but_incorrect"", ""query_intrinsically_ambiguous_or_generic"", ""insufficient_evidence""]",data/flickr30k/validation_images/29640.jpg
audit_070,"A man in a gray cap, a pink tank top and short skirt, pink-and-black striped socks, and boots juggles knives in front of a brick building whose sign reads ""The Pump Room"" while onlookers watch.",data/flickr30k/validation_images/19652.jpg,data/flickr30k/validation_images/462.jpg,data/flickr30k/validation_images/15800.jpg,data/flickr30k/validation_images/22747.jpg,data/flickr30k/validation_images/10885.jpg,"[""annotated_image_uniquely_correct"", ""retrieved_alternative_equally_plausible"", ""retrieved_alternative_related_but_incorrect"", ""query_intrinsically_ambiguous_or_generic"", ""insufficient_evidence""]",data/flickr30k/validation_images/6533.jpg
audit_085,A man in a red and white striped shirts sits on a stool at a hotdog stand.,data/flickr30k/validation_images/16664.jpg,data/flickr30k/validation_images/6143.jpg,data/flickr30k/validation_images/900.jpg,data/flickr30k/validation_images/16632.jpg,data/flickr30k/validation_images/20401.jpg,"[""annotated_image_uniquely_correct"", ""retrieved_alternative_equally_plausible"", ""retrieved_alternative_related_but_incorrect"", ""query_intrinsically_ambiguous_or_generic"", ""insufficient_evidence""]",data/flickr30k/validation_images/30040.jpg
audit_090,A large crowd is standing around the start line.,data/flickr30k/validation_images/20098.jpg,data/flickr30k/validation_images/2089.jpg,data/flickr30k/validation_images/15638.jpg,data/flickr30k/validation_images/6278.jpg,data/flickr30k/validation_images/12879.jpg,"[""annotated_image_uniquely_correct"", ""retrieved_alternative_equally_plausible"", ""retrieved_alternative_related_but_incorrect"", ""query_intrinsically_ambiguous_or_generic"", ""insufficient_evidence""]",data/flickr30k/validation_images/8777.jpg
```

### Failure sample — first five

```csv
direction,query_id,query,positive_id,positive_rank_seed42,positive_item,ranks_all_seeds,top10_seed42,retrieved_items_seed42,predeclared_categories
t2i,18952::caption_1,A brunet male in a brown blazer talking to an audience at a comedian show.,18952,37,"{""image_id"": ""18952"", ""image_path"": ""data/flickr30k/validation_images/18952.jpg""}","{""42.0"": 37, ""43.0"": 67, ""44.0"": 33}","[""26268"", ""26632"", ""28430"", ""11190"", ""27707"", ""4840"", ""10741"", ""22291"", ""12521"", ""29095""]","[{""image_id"": ""26268"", ""image_path"": ""data/flickr30k/validation_images/26268.jpg""}, {""image_id"": ""26632"", ""image_path"": ""data/flickr30k/validation_images/26632.jpg""}, {""image_id"": ""28430"", ""image_path"": ""data/flickr30k/validation_images/28430.jpg""}, {""image_id"": ""11190"", ""image_path"": ""data/flickr30k/validation_images/11190.jpg""}, {""image_id"": ""27707"", ""image_path"": ""data/flickr30k/validation_images/27707.jpg""}, {""image_id"": ""4840"", ""image_path"": ""data/flickr30k/validation_images/4840.jpg""}, {""image_id"": ""10741"", ""image_path"": ""data/flickr30k/validation_images/10741.jpg""}, {""image_id"": ""22291"", ""image_path"": ""data/flickr30k/validation_images/22291.jpg""}, {""image_id"": ""12521"", ""image_path"": ""data/flickr30k/validation_images/12521.jpg""}, {""image_id"": ""29095"", ""image_path"": ""data/flickr30k/validation_images/29095.jpg""}]","[""fine_grained_attribute"", ""counting_quantity"", ""spatial_relation"", ""action_interaction"", ""object_category_confusion"", ""scene_context_confusion"", ""text_ocr_content"", ""annotation_ambiguity"", ""other""]"
t2i,2256::caption_0,"A young woman and older woman wear traditional saris as they spin textiles, three people are pictured at only the waists, and wear modern clothes.",2256,83,"{""image_id"": ""2256"", ""image_path"": ""data/flickr30k/validation_images/2256.jpg""}","{""42.0"": 83, ""43.0"": 20, ""44.0"": 19}","[""20853"", ""5752"", ""14199"", ""29492"", ""1478"", ""28990"", ""1162"", ""26141"", ""22099"", ""18440""]","[{""image_id"": ""20853"", ""image_path"": ""data/flickr30k/validation_images/20853.jpg""}, {""image_id"": ""5752"", ""image_path"": ""data/flickr30k/validation_images/5752.jpg""}, {""image_id"": ""14199"", ""image_path"": ""data/flickr30k/validation_images/14199.jpg""}, {""image_id"": ""29492"", ""image_path"": ""data/flickr30k/validation_images/29492.jpg""}, {""image_id"": ""1478"", ""image_path"": ""data/flickr30k/validation_images/1478.jpg""}, {""image_id"": ""28990"", ""image_path"": ""data/flickr30k/validation_images/28990.jpg""}, {""image_id"": ""1162"", ""image_path"": ""data/flickr30k/validation_images/1162.jpg""}, {""image_id"": ""26141"", ""image_path"": ""data/flickr30k/validation_images/26141.jpg""}, {""image_id"": ""22099"", ""image_path"": ""data/flickr30k/validation_images/22099.jpg""}, {""image_id"": ""18440"", ""image_path"": ""data/flickr30k/validation_images/18440.jpg""}]","[""fine_grained_attribute"", ""counting_quantity"", ""spatial_relation"", ""action_interaction"", ""object_category_confusion"", ""scene_context_confusion"", ""text_ocr_content"", ""annotation_ambiguity"", ""other""]"
t2i,10810::caption_4,Somebody in the street with some sweatpants on.,10810,49,"{""image_id"": ""10810"", ""image_path"": ""data/flickr30k/validation_images/10810.jpg""}","{""42.0"": 49, ""43.0"": 52, ""44.0"": 59}","[""23945"", ""22003"", ""25205"", ""1271"", ""7330"", ""22213"", ""19253"", ""15329"", ""21091"", ""27246""]","[{""image_id"": ""23945"", ""image_path"": ""data/flickr30k/validation_images/23945.jpg""}, {""image_id"": ""22003"", ""image_path"": ""data/flickr30k/validation_images/22003.jpg""}, {""image_id"": ""25205"", ""image_path"": ""data/flickr30k/validation_images/25205.jpg""}, {""image_id"": ""1271"", ""image_path"": ""data/flickr30k/validation_images/1271.jpg""}, {""image_id"": ""7330"", ""image_path"": ""data/flickr30k/validation_images/7330.jpg""}, {""image_id"": ""22213"", ""image_path"": ""data/flickr30k/validation_images/22213.jpg""}, {""image_id"": ""19253"", ""image_path"": ""data/flickr30k/validation_images/19253.jpg""}, {""image_id"": ""15329"", ""image_path"": ""data/flickr30k/validation_images/15329.jpg""}, {""image_id"": ""21091"", ""image_path"": ""data/flickr30k/validation_images/21091.jpg""}, {""image_id"": ""27246"", ""image_path"": ""data/flickr30k/validation_images/27246.jpg""}]","[""fine_grained_attribute"", ""counting_quantity"", ""spatial_relation"", ""action_interaction"", ""object_category_confusion"", ""scene_context_confusion"", ""text_ocr_content"", ""annotation_ambiguity"", ""other""]"
t2i,5553::caption_0,Two young couples posing for a picture in exercise clothing.,5553,96,"{""image_id"": ""5553"", ""image_path"": ""data/flickr30k/validation_images/5553.jpg""}","{""42.0"": 96, ""43.0"": 63, ""44.0"": 59}","[""30177"", ""8339"", ""20478"", ""9174"", ""14824"", ""22639"", ""30369"", ""29884"", ""4054"", ""10244""]","[{""image_id"": ""30177"", ""image_path"": ""data/flickr30k/validation_images/30177.jpg""}, {""image_id"": ""8339"", ""image_path"": ""data/flickr30k/validation_images/8339.jpg""}, {""image_id"": ""20478"", ""image_path"": ""data/flickr30k/validation_images/20478.jpg""}, {""image_id"": ""9174"", ""image_path"": ""data/flickr30k/validation_images/9174.jpg""}, {""image_id"": ""14824"", ""image_path"": ""data/flickr30k/validation_images/14824.jpg""}, {""image_id"": ""22639"", ""image_path"": ""data/flickr30k/validation_images/22639.jpg""}, {""image_id"": ""30369"", ""image_path"": ""data/flickr30k/validation_images/30369.jpg""}, {""image_id"": ""29884"", ""image_path"": ""data/flickr30k/validation_images/29884.jpg""}, {""image_id"": ""4054"", ""image_path"": ""data/flickr30k/validation_images/4054.jpg""}, {""image_id"": ""10244"", ""image_path"": ""data/flickr30k/validation_images/10244.jpg""}]","[""fine_grained_attribute"", ""counting_quantity"", ""spatial_relation"", ""action_interaction"", ""object_category_confusion"", ""scene_context_confusion"", ""text_ocr_content"", ""annotation_ambiguity"", ""other""]"
t2i,20129::caption_2,A man holding a sign and a man playing a guitar are speaking in front of a crowd.,20129,222,"{""image_id"": ""20129"", ""image_path"": ""data/flickr30k/validation_images/20129.jpg""}","{""42.0"": 222, ""43.0"": 190, ""44.0"": 188}","[""19141"", ""3304"", ""20171"", ""17212"", ""27707"", ""4561"", ""30188"", ""26268"", ""18126"", ""1103""]","[{""image_id"": ""19141"", ""image_path"": ""data/flickr30k/validation_images/19141.jpg""}, {""image_id"": ""3304"", ""image_path"": ""data/flickr30k/validation_images/3304.jpg""}, {""image_id"": ""20171"", ""image_path"": ""data/flickr30k/validation_images/20171.jpg""}, {""image_id"": ""17212"", ""image_path"": ""data/flickr30k/validation_images/17212.jpg""}, {""image_id"": ""27707"", ""image_path"": ""data/flickr30k/validation_images/27707.jpg""}, {""image_id"": ""4561"", ""image_path"": ""data/flickr30k/validation_images/4561.jpg""}, {""image_id"": ""30188"", ""image_path"": ""data/flickr30k/validation_images/30188.jpg""}, {""image_id"": ""26268"", ""image_path"": ""data/flickr30k/validation_images/26268.jpg""}, {""image_id"": ""18126"", ""image_path"": ""data/flickr30k/validation_images/18126.jpg""}, {""image_id"": ""1103"", ""image_path"": ""data/flickr30k/validation_images/1103.jpg""}]","[""fine_grained_attribute"", ""counting_quantity"", ""spatial_relation"", ""action_interaction"", ""object_category_confusion"", ""scene_context_confusion"", ""text_ocr_content"", ""annotation_ambiguity"", ""other""]"
```

### Failure sample — random five (indices [6, 9, 14, 42, 45])

```csv
direction,query_id,query,positive_id,positive_rank_seed42,positive_item,ranks_all_seeds,top10_seed42,retrieved_items_seed42,predeclared_categories
t2i,30971::caption_0,"Two men are using phone-booths that are located shortly outside of the downtown area of a city, by some trees.",30971,38,"{""image_id"": ""30971"", ""image_path"": ""data/flickr30k/validation_images/30971.jpg""}","{""42.0"": 38, ""43.0"": 23, ""44.0"": 34}","[""21441"", ""19688"", ""19956"", ""12018"", ""16365"", ""22936"", ""20138"", ""14842"", ""22916"", ""21130""]","[{""image_id"": ""21441"", ""image_path"": ""data/flickr30k/validation_images/21441.jpg""}, {""image_id"": ""19688"", ""image_path"": ""data/flickr30k/validation_images/19688.jpg""}, {""image_id"": ""19956"", ""image_path"": ""data/flickr30k/validation_images/19956.jpg""}, {""image_id"": ""12018"", ""image_path"": ""data/flickr30k/validation_images/12018.jpg""}, {""image_id"": ""16365"", ""image_path"": ""data/flickr30k/validation_images/16365.jpg""}, {""image_id"": ""22936"", ""image_path"": ""data/flickr30k/validation_images/22936.jpg""}, {""image_id"": ""20138"", ""image_path"": ""data/flickr30k/validation_images/20138.jpg""}, {""image_id"": ""14842"", ""image_path"": ""data/flickr30k/validation_images/14842.jpg""}, {""image_id"": ""22916"", ""image_path"": ""data/flickr30k/validation_images/22916.jpg""}, {""image_id"": ""21130"", ""image_path"": ""data/flickr30k/validation_images/21130.jpg""}]","[""fine_grained_attribute"", ""counting_quantity"", ""spatial_relation"", ""action_interaction"", ""object_category_confusion"", ""scene_context_confusion"", ""text_ocr_content"", ""annotation_ambiguity"", ""other""]"
t2i,11539::caption_4,THe dog takes a rest in the park.,11539,35,"{""image_id"": ""11539"", ""image_path"": ""data/flickr30k/validation_images/11539.jpg""}","{""42.0"": 35, ""43.0"": 12, ""44.0"": 31}","[""13094"", ""9758"", ""5950"", ""7156"", ""9724"", ""2324"", ""10084"", ""9291"", ""14001"", ""1074""]","[{""image_id"": ""13094"", ""image_path"": ""data/flickr30k/validation_images/13094.jpg""}, {""image_id"": ""9758"", ""image_path"": ""data/flickr30k/validation_images/9758.jpg""}, {""image_id"": ""5950"", ""image_path"": ""data/flickr30k/validation_images/5950.jpg""}, {""image_id"": ""7156"", ""image_path"": ""data/flickr30k/validation_images/7156.jpg""}, {""image_id"": ""9724"", ""image_path"": ""data/flickr30k/validation_images/9724.jpg""}, {""image_id"": ""2324"", ""image_path"": ""data/flickr30k/validation_images/2324.jpg""}, {""image_id"": ""10084"", ""image_path"": ""data/flickr30k/validation_images/10084.jpg""}, {""image_id"": ""9291"", ""image_path"": ""data/flickr30k/validation_images/9291.jpg""}, {""image_id"": ""14001"", ""image_path"": ""data/flickr30k/validation_images/14001.jpg""}, {""image_id"": ""1074"", ""image_path"": ""data/flickr30k/validation_images/1074.jpg""}]","[""fine_grained_attribute"", ""counting_quantity"", ""spatial_relation"", ""action_interaction"", ""object_category_confusion"", ""scene_context_confusion"", ""text_ocr_content"", ""annotation_ambiguity"", ""other""]"
t2i,25731::caption_1,Two boys are participating in a boxing match.,25731,69,"{""image_id"": ""25731"", ""image_path"": ""data/flickr30k/validation_images/25731.jpg""}","{""42.0"": 69, ""43.0"": 67, ""44.0"": 16}","[""17598"", ""28087"", ""18956"", ""6049"", ""29105"", ""17095"", ""6377"", ""30177"", ""28292"", ""20439""]","[{""image_id"": ""17598"", ""image_path"": ""data/flickr30k/validation_images/17598.jpg""}, {""image_id"": ""28087"", ""image_path"": ""data/flickr30k/validation_images/28087.jpg""}, {""image_id"": ""18956"", ""image_path"": ""data/flickr30k/validation_images/18956.jpg""}, {""image_id"": ""6049"", ""image_path"": ""data/flickr30k/validation_images/6049.jpg""}, {""image_id"": ""29105"", ""image_path"": ""data/flickr30k/validation_images/29105.jpg""}, {""image_id"": ""17095"", ""image_path"": ""data/flickr30k/validation_images/17095.jpg""}, {""image_id"": ""6377"", ""image_path"": ""data/flickr30k/validation_images/6377.jpg""}, {""image_id"": ""30177"", ""image_path"": ""data/flickr30k/validation_images/30177.jpg""}, {""image_id"": ""28292"", ""image_path"": ""data/flickr30k/validation_images/28292.jpg""}, {""image_id"": ""20439"", ""image_path"": ""data/flickr30k/validation_images/20439.jpg""}]","[""fine_grained_attribute"", ""counting_quantity"", ""spatial_relation"", ""action_interaction"", ""object_category_confusion"", ""scene_context_confusion"", ""text_ocr_content"", ""annotation_ambiguity"", ""other""]"
t2i,13477::caption_0,A brown and white dog is jumping in the air with a tennis ball in its mouth.,13477,20,"{""image_id"": ""13477"", ""image_path"": ""data/flickr30k/validation_images/13477.jpg""}","{""42.0"": 20, ""43.0"": 24, ""44.0"": 26}","[""14001"", ""19824"", ""2324"", ""9291"", ""9488"", ""5062"", ""13351"", ""4053"", ""8303"", ""7156""]","[{""image_id"": ""14001"", ""image_path"": ""data/flickr30k/validation_images/14001.jpg""}, {""image_id"": ""19824"", ""image_path"": ""data/flickr30k/validation_images/19824.jpg""}, {""image_id"": ""2324"", ""image_path"": ""data/flickr30k/validation_images/2324.jpg""}, {""image_id"": ""9291"", ""image_path"": ""data/flickr30k/validation_images/9291.jpg""}, {""image_id"": ""9488"", ""image_path"": ""data/flickr30k/validation_images/9488.jpg""}, {""image_id"": ""5062"", ""image_path"": ""data/flickr30k/validation_images/5062.jpg""}, {""image_id"": ""13351"", ""image_path"": ""data/flickr30k/validation_images/13351.jpg""}, {""image_id"": ""4053"", ""image_path"": ""data/flickr30k/validation_images/4053.jpg""}, {""image_id"": ""8303"", ""image_path"": ""data/flickr30k/validation_images/8303.jpg""}, {""image_id"": ""7156"", ""image_path"": ""data/flickr30k/validation_images/7156.jpg""}]","[""fine_grained_attribute"", ""counting_quantity"", ""spatial_relation"", ""action_interaction"", ""object_category_confusion"", ""scene_context_confusion"", ""text_ocr_content"", ""annotation_ambiguity"", ""other""]"
t2i,10392::caption_1,Two women with their baby strollers walking along a leaves covered street.,10392,71,"{""image_id"": ""10392"", ""image_path"": ""data/flickr30k/validation_images/10392.jpg""}","{""42.0"": 71, ""43.0"": 13, ""44.0"": 45}","[""23219"", ""10963"", ""14842"", ""3203"", ""13943"", ""18959"", ""28787"", ""22916"", ""10557"", ""19544""]","[{""image_id"": ""23219"", ""image_path"": ""data/flickr30k/validation_images/23219.jpg""}, {""image_id"": ""10963"", ""image_path"": ""data/flickr30k/validation_images/10963.jpg""}, {""image_id"": ""14842"", ""image_path"": ""data/flickr30k/validation_images/14842.jpg""}, {""image_id"": ""3203"", ""image_path"": ""data/flickr30k/validation_images/3203.jpg""}, {""image_id"": ""13943"", ""image_path"": ""data/flickr30k/validation_images/13943.jpg""}, {""image_id"": ""18959"", ""image_path"": ""data/flickr30k/validation_images/18959.jpg""}, {""image_id"": ""28787"", ""image_path"": ""data/flickr30k/validation_images/28787.jpg""}, {""image_id"": ""22916"", ""image_path"": ""data/flickr30k/validation_images/22916.jpg""}, {""image_id"": ""10557"", ""image_path"": ""data/flickr30k/validation_images/10557.jpg""}, {""image_id"": ""19544"", ""image_path"": ""data/flickr30k/validation_images/19544.jpg""}]","[""fine_grained_attribute"", ""counting_quantity"", ""spatial_relation"", ""action_interaction"", ""object_category_confusion"", ""scene_context_confusion"", ""text_ocr_content"", ""annotation_ambiguity"", ""other""]"
```

## 4. Existing distributions

TF-IDF is the primary ambiguity metric. BGE is a robustness check only. Their source-level cross-axis Spearman correlation is 0.5211 and weighted bin kappa is 0.3538; both are reported, but their agreement is not treated as validation.

Aggregate distributions from the protected key (not joined to displayed rows):

```json
{
  "partition": {
    "outside_top_10": 33,
    "rank_1": 34,
    "rank_2_10": 33
  },
  "tfidf_cross_bin": {
    "high": 33,
    "low": 34,
    "medium": 33
  },
  "bge_cross_bin": {
    "high": 37,
    "low": 32,
    "medium": 31
  },
  "partition_x_tfidf": {
    "outside_top_10|high": 11,
    "outside_top_10|low": 11,
    "outside_top_10|medium": 11,
    "rank_1|high": 11,
    "rank_1|low": 12,
    "rank_1|medium": 11,
    "rank_2_10|high": 11,
    "rank_2_10|low": 11,
    "rank_2_10|medium": 11
  }
}
```

Failure sample direction and per-seed outcome distributions:

```json
{
  "direction": {
    "t2i": 50
  },
  "per_seed_outside_top10": {
    "seed_42.0|outside_top10=True": 50,
    "seed_43.0|outside_top10=True": 50,
    "seed_44.0|outside_top10=True": 50
  }
}
```

The serialized `rubric_categories` and `predeclared_categories` columns contain suggestion lists, not assigned categorical outcomes, so no label-frequency distribution is reported.

## 5. Annotation templates

Each template carries the existing item identifier and display fields required for human judgement, plus empty `rubric_label_placeholder` and `human_notes` columns. No rubric categories were invented or assigned.
