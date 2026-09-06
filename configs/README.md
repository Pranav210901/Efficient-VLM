# Configuration namespace

Configuration directories retain their historical implementation names because
their literal contents participate in fingerprints and resume validation.
They are therefore **not** physically renamed during project cleanup.

Use [`../experiments/README.md`](../experiments/README.md) to find the research
headline for a configuration family, and
[`../experiments/layout.yaml`](../experiments/layout.yaml) to find the grouped
artifact destination.

New experiment families should receive a stable experiment ID in
`experiments/registry.yaml` before a new configuration directory is added.
