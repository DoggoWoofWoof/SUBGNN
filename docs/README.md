# docs/ index

Design docs, runbooks, audits, and strategy notes for the Jigsaw paper.

## Paper verification & status
- [paper_claim_audit.md](paper_claim_audit.md) — every paper claim → code/data evidence map; lists the one artifact inconsistency to fix (stale Cora/Arxiv multi rows) before release.
- [final_submission_audit.md](final_submission_audit.md) — reviewer-concern → evidence mapping; locks the canonical paper artifacts.
- [log2026_submission_checklist.md](log2026_submission_checklist.md) — final LoG PDF hash, page/anonymity checks, reproducibility command, and reviewer-facing positioning note.
- [benchmark_metrics_reference.md](benchmark_metrics_reference.md) — definitions of the benchmark metrics/columns.

## Planning & venue strategy
- [acceptance_plan.md](acceptance_plan.md) — upcoming venue deadlines (VLDB/WSDM/ICDE/KDD) + per-item readiness, launch commands, and sequencing to a clear accept.
- [resubmission_runbook.md](resubmission_runbook.md) — steps to merge corrected reruns and refresh the canonical summary.
- [ecml_pkdd_mlg_paper_strategy.md](ecml_pkdd_mlg_paper_strategy.md) — earlier venue strategy note.

## Method & protocol references
- [jigsaw_loss_and_retrieval_method.md](jigsaw_loss_and_retrieval_method.md) — FullCov objective + retrieval method.
- [production_benchmark_protocol.md](production_benchmark_protocol.md) — production benchmark protocol (seeds, families, sizes).
- [query_generator_alignment.md](query_generator_alignment.md) — query-generator alignment notes.
- [arxiv_khop_control_vs_scheduler_seed42.md](arxiv_khop_control_vs_scheduler_seed42.md), [multiview_retrieval_ablation.md](multiview_retrieval_ablation.md) — specific ablation notes.

## Runbooks
- [targeted_experiment_runbook.md](targeted_experiment_runbook.md) — how to run targeted experiments cheaply (don't re-run the full grid); offline probe + selective/bridge overlap flags.
- [walk_aware_retrain_runbook.md](walk_aware_retrain_runbook.md) — staged walk-aware MAG retraining (the random_walk fix) + cheap eval.

## Repo organization (history)
- [repo_cleanup_plan.md](repo_cleanup_plan.md) — disk/artifact cleanup plan.
- [scripts_cleanup_plan.md](scripts_cleanup_plan.md) — scripts/ consolidation plan (now executed: see scripts/README.md).
- [repo_cleanup_audit.md](repo_cleanup_audit.md) — earlier archive-based cleanup history.

Related: [scripts/README.md](../scripts/README.md) (scripts layout), [README.md](../README.md) (repo overview), and the diagnostic findings in `runs/diagnostics/` (e.g. `candidate_shrinkage_findings.md`, `random_walk_analysis.md`).
