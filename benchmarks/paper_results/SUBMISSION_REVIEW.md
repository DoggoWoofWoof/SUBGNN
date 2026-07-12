# Submission Bundle Review

## Included

- `final_results/`: canonical production benchmark summaries and completion audit.
- `ablations/`: paper-facing method evidence, including Arxiv design ablations, locked objective/loss ablations, paired tests, and v7 continuation summaries.
- `diagnostics/`: older locked diagnostics retained for context and sanity checks.
- `glasgow_benchmark_arxiv_all.csv` and `glasgow_benchmark_corafull_all.csv`: legacy Glasgow evidence retained for historical comparison.

## Main Claim Coverage

- Production datasets: MAG, Arxiv, Cora.
- Production methods: neural, random, mean-feature, mean-RRF, topo-feature, and filter-all components.
- Query families: single, degree-k-hop, k-hop, multi-coarse, multi-fine, random-walk, negative-label, negative-structure.
- Query sizes: 20, 50, 100.
- Seeds: 20260607 and 20260608.

## Ablation Coverage

- Arxiv design ablations: no overlap, no component restriction, no signature, no exact-label, and final neural component.
- Arxiv objective/loss evidence: locked matched-budget aggregate, paired McNemar tests, and seed-level retrieval summaries.
- Arxiv v7 continuation evidence: retrieval summaries for both locked query seeds.

## Intentionally Not Included

- Raw Lightning/Modal cache directories.
- Large raw per-query ablation CSVs where compact summaries are sufficient for paper review.
- Local `.pth` checkpoint binaries; hashes are documented in `models/README.md`.
- Probe, spot-partial, and duplicate temp run folders; these are quarantined under `archive/non_submission_20260620/`.

## Review Result

No missing paper-facing benchmark summary was found after the final Cora/Arxiv completion jobs. The only remaining large evidence not copied into this folder is raw provenance data under `runs/`, which is intentionally excluded from the submission bundle.
