# Local datasets

Dataset payloads are intentionally excluded from the GitHub submission. They
are large, locally generated or downloaded, and may carry redistribution terms
different from this repository. The small Flickr30k validation and sealed-test
CSV manifests are retained for split provenance and integrity tests; images are
not redistributed. Experiment configurations preserve every expected path and
dataset revision.

Expected local layout:

```text
data/
├── coco/          # train2017, val2017, annotations, and prepared caption CSVs
├── flickr30k/     # images plus Karpathy validation/test CSVs
└── cc3m_v1_1/     # pixparse/cc3m-wds shards, manifests, ledgers, and caches
```

Rebuild COCO with `scripts/download_coco.sh` and Flickr30k with
`scripts/fetch_flickr30k.py`. The qualified CC3M mirror, pinned revision, shard
policy, and acquisition commands are documented in
`docs/cc3m_acquisition_protocol.md`, `configs/cc3m_scale/pipeline.yaml`, and
`scripts/submit_cc3m_acquisition.sh`.

The local copies used for the dissertation were moved, not deleted, during
submission cleanup. Their exact recovery location is recorded in the ignored
quarantine manifest described by `docs/github_submission_package.md`.
