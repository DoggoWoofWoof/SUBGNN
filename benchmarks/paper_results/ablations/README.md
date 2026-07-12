# Ablation Evidence

This folder contains paper-facing ablation evidence. Raw per-query and training logs remain under `runs/` and are intentionally ignored by Git.

Files:

- `arxiv_design_ablation_summary.csv`: Arxiv production summary rows for the final neural method and no-overlap/no-component/no-signature/no-exact-label ablations.
- `fair_ablation_locked_aggregate.*`: locked, matched-budget objective/loss ablation aggregates and paired McNemar tests.
- `retrieval_arxiv_khop_fair_ablation_*_summary.csv`: seed-level retrieval summaries behind the locked objective/loss ablation.
- `retrieval_arxiv_khop_v7_candidates_*_summary.csv`: v7 continuation retrieval summaries used to support the final Arxiv model choice.

The paper-facing `arxiv_design_ablation_summary.csv` uses the same normalized leading schema as the final results bundle, including explicit `dataset` and `seed` columns. It has been de-duplicated against partial and blank aggregate rows; older raw diagnostics remain outside this folder under `runs/` or the non-submission archive.
