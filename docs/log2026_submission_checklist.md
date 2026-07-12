# LoG 2026 Submission Checklist

## Frozen Artifact

- Submission PDF: `paper/jigsaw_log2026_submission.pdf`
- SHA-256: `324A6118EFBEE0985A3EB9EF05AC8044E3C98E61637436177D2CE889EC2EAA12`
- Page layout: 19 total pages; body ends on page 9; references start on page 10; appendices start on page 11.
- Anonymity: review mode enabled, anonymous author metadata, no URI annotations, no local path/name leakage found in text or raw PDF bytes.

## Required Final Checks

Run from the repository root:

```powershell
python scripts\analysis\reproduce_paper_numbers.py
```

Expected result: `ALL REQUIRED CHECKS PASSED`.

The checker covers:

- MAG deployed walk-aware matrix: Jigsaw 88.6%, Mean-RRF 85.4%, family rates, size slices, and McNemar counts.
- Cross-dataset label-selectivity selector: Cora/Arxiv rank medians, per-query win counts, and sign-test strength.
- Inference-time retrieval-remedy foreclosure: multivector, fine-grain, overlap, diffusion, and stitch probes.
- Optional GNN-PE diagnostic provenance. The e=4 compact summary is local; the e=128 compact answer summary is not local after cleanup, so the checker warns unless `--strict-optional` is requested.

## Reviewer-Facing LoG Positioning

Jigsaw is best positioned as a learning-for-systems paper: the learning component does not replace exact matching, but makes exact subgraph verification usable under a tunable memory budget by ranking graph regions before invoking Glasgow. The contribution is not that a neural retriever universally dominates classical indexes; the paper now shows the opposite when labels are near-unique, then gives a statistically clean selectivity-based selector and a tested negative result for cheap retrieval fixes. For LoG, the learning content is the coverage-trained region retriever and the empirical characterization of when learned retrieval is useful; the systems content is the exact, memory-bounded verification pipeline.

## Do Not Reopen Before Submission

- Do not add new experiments or tables unless the out-of-core baseline comparison is deliberately started as a new project.
- Do not change LoG body text unless page 9 is rechecked afterward.
- Do not regenerate the submission PDF without updating the SHA-256 above.
- Do not source paper numbers from legacy `final_*_summary.csv` neural rows; use `benchmarks/paper_results/final_results/HEADLINE_NUMBERS.csv` and `benchmarks/paper_results/CANONICAL_SOURCES.md`.
