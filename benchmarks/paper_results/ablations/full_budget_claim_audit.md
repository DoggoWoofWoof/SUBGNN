# Full-budget and 100% claim audit

Audit date: 2026-08-01

This audit checks every paper-facing use of an exhaustive partition budget and
every reported 100% value in `samplepaper.tex`, `jigsaw_log2026.tex`, and
`jigsaw_ecmlpkdd.tex`.

## Corrections made

1. The Arxiv operator ablation previously used `K=200/200`. That made
   partition retrieval exhaustive and made overlap vacuous. The table and figure
   now use matched half-partition budgets: Arxiv `K=100/200` and MAG
   `K=1000/2000`. The reproducible source is
   `operator_ablation_half_budget_summary.csv`.
2. The small-graph scaling diagnostic previously used absolute `K=20` for both
   Cora and Arxiv, which was exhaustive on Cora. It now uses the same 15 K-hop
   queries at `K=10/20` and `K=100/200`. The reproducible source is
   `scaling_half_budget_paired_summary.csv`.
3. The production table previously placed half-budget solve rates beside
   Cora/Arxiv costs from cascades that could continue to exhaustive `K=20/200`.
   Those incomparable cost cells are now omitted. MAG cost remains reported
   because its production endpoint `K=1000/2000` is the matched half budget.

## Remaining 100% values

- **Exhaustive retrieval:** Cora/Arxiv aggregate 100% values occur only at
  FilterAll or the all-partition endpoint and are labeled exhaustive.
- **Half-budget family slices:** Individual easy families such as `single` and
  `multi_fine` can legitimately reach 100% while the aggregate remains below
  100%; their captions state the half-partition budget.
- **Negative accuracy:** Cora/Arxiv 100% values in negative columns are audited
  correct no-match rates, not positive retrieval or solve rates.
- **Classical exact solvers:** Cora/Arxiv 100% values for CFL-Match, DP-iso, and
  GQL are whole-graph resident baselines and are explicitly labeled as such.
- **Fixed-budget retrieval metrics:** FullCov@K values name their budget in the
  metric and do not imply exhaustive retrieval.

No other paper-facing 100% or full-budget claim is used as evidence for bounded
partition selection or for an operator being unnecessary.
