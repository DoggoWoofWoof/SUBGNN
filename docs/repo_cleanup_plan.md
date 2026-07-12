# Jigsaw Repository Cleanup Plan

Date: 2026-06-21
Author: automated audit (investigation only; NO files were deleted or modified)

## Scope and method

Disk footprint (measured 2026-06-21):

| Top-level | Size |
| --- | ---: |
| `runs/` | ~29 GB |
| `archive/` | ~5.1 GB |
| `data/` | ~1.9 GB |
| `.venv_modal/` | ~864 MB |
| `models/` | ~280 MB |
| `tmp/` | ~68 MB |
| `cache/` | ~11 MB |
| Everything else (src, scripts, paper, docs, benchmarks, tests, results) | < 10 MB |

Important context for the owner:

- The git repo tracks only **39 files**. `runs/`, `archive/`, `cache/`, `data/`,
  `tmp/`, `.venv_modal/`, `models/*.pth`, and `reviews.pdf` are all **git-ignored**
  (see `.gitignore`). So this cleanup is about reclaiming **disk space**, not about
  the committed tree. Deleting any of the items below will not change `git status`
  for tracked files.
- This plan supersedes nothing in `docs/repo_cleanup_audit.md` /
  `docs/final_submission_audit.md`; those are archive-based content audits. See the
  "Prior audits" section for what they already recommended.

## Prior audits (summary)

- `docs/repo_cleanup_audit.md`: An archive-based cleanup history. It established the
  canonical paper evidence as `benchmarks/paper_results/`, designated
  `scripts/train_jigsaw_model.py` as the canonical trainer and `src/model.py` as the
  canonical model, moved legacy benchmarks/scripts/figures into `archive/` (not
  deleted), and on June 15 compacted Lightning/MAG run folders (kept
  `runs/lightning_cache/kutta_v2_final_backup/` as the canonical MAG RGCN artifact,
  removed duplicate/corrupt Lightning backups and generated venv caches). It also
  generated de-duplicated combined CSV/log bundles under `runs/combined/`.
- `docs/final_submission_audit.md`: A results-acceptance audit (not a file audit).
  It maps reviewer concerns to evidence, locks the paper claims (FullCov-aligned
  retrieval + exact verification, GraphSAGE encoder-transfer, MAG as a negative
  scalability diagnostic), and names the canonical manuscript artifacts
  (`paper/samplepaper.tex` / `.pdf`) plus the specific `runs/logs/*aggregate.md`
  and `benchmarks/paper_results/diagnostics/*` files that back each table. These
  named log/CSV artifacts must be treated as KEEP.

---

## Safe to delete

High confidence. Temp/scratch, regenerable virtualenv caches, compiler/latex
aux, smoke tests, and explicitly-quarantined caches. Estimated reclaim ~1.2 GB
(excluding `.venv_modal`, which is regenerable but listed separately because some
users prefer to keep it). All paths relative to repo root.

| Path / glob | Size | Why safe |
| --- | ---: | --- |
| `tmp/` (whole dir: `tmp/query_gen_smoke/`, `tmp/pdf_render/`) | ~68 MB | Git-ignored scratch dir. `query_gen_smoke` is a query-generator smoke test (~64 MB); `pdf_render` is a transient PDF render scratch (~5.9 MB). Neither is paper evidence. |
| `runs/quarantine_disconnected_query_caches/` | ~182 MB | Explicitly named "quarantine"; contains `README.md` + `moved_files.txt` documenting disconnected/bad query caches that were already pulled out of the active workflow. |
| `**/__pycache__/`, `**/*.pyc`, `**/*.pyo` | small (mostly inside `.venv_modal`) | Python bytecode caches; always regenerable. 10,309 `.pyc` files, almost all under `.venv_modal/Lib/site-packages`. |
| `paper/*.aux`, `paper/*.log`, `paper/*.out`, `paper/*.toc`, `paper/*.synctex.gz`, `paper/*.fls`, `paper/*.fdb_latexmk` | tiny | LaTeX build aux; regenerated on every compile. Already git-ignored. Currently present: `paper/samplepaper.aux`, `paper/samplepaper.log`. **Do NOT touch `paper/*.tex`, `paper/*.pdf`, `paper/*.png`.** |
| `runs/paper_render_20260610_final/`, `runs/paper_visual_june9_final/`, `runs/whitepaper_render_20260610_final/` | ~5.5 MB total | Rendered page-image QA folders, git-ignored (`runs/paper_render*/`, `runs/paper_visual*/`). The canonical PDF lives in `paper/`. Regenerable via the render scripts. |
| `runs/lightning_mag_smoke/` | ~1.5 MB | MAG Lightning smoke-test output; not paper evidence (audit treats smoke runs as disposable). |

Optional / regenerable but listed separately:

| Path | Size | Note |
| --- | ---: | --- |
| `.venv_modal/` | ~864 MB | Git-ignored Python virtualenv. Fully regenerable from `requirements_lightning_rgcn.txt` / Modal config. Safe to delete to reclaim space, but only if you can rebuild the env. Mark as safe-but-confirm. |

---

## Review (large regenerable artifacts — do NOT auto-delete)

These are large `.pt` caches that can be regenerated from data + scripts, but may
be needed for fast local analysis. Decide per-item; do not bulk-delete.

| Path / glob | Size | Note |
| --- | ---: | --- |
| `runs/lightning_mag_full/**/overlap_cascade/*prepared_hierarchy.pt` (and the other `*prepared_hierarchy.pt` under `runs/`) | **~11.4 GB total across 17 copies** (~1.5 GB each) | The single biggest reclaim opportunity. Many identical-named per-run copies of the MAG prepared hierarchy. Almost certainly the same regenerable cache duplicated across runs. **Review: keep one canonical copy, delete the rest after confirming they are not the active analysis input.** |
| `runs/**/overlap_cascade/*signature_tokens.pt` | **~4.7 GB total across 41 copies** (~134 MB each) | Per-run MAG signature-token caches, regenerable. Heavily duplicated across the 19 `runs/lightning_mag_full/*` run dirs. Review for dedup. |
| `runs/lightning_mag_full/` | ~13 GB | 19 per-config MAG run dirs (topo / meanrrf / neural / random / filterall x seeds x resume versions). Bulk is the `.pt` caches above. Per-run `results/` and `logs/` are small and may hold provenance — review before deleting whole dirs. |
| `runs/migration/` | ~3.5 GB | `mag_darkphoenix/` (~1.7 GB), `mag_rgcn_pes_to_swathi/` (~1.6 GB), `fair_ablation/` (~323 MB). Migration staging copies of MAG hierarchies. Note `mag_darkphoenix/` has BOTH `mag_hierarchies_..._v1.pt` and `..._v1.clean.pt` — they are **not** byte-identical (different MD5), so the `.clean.pt` is a distinct cleaned copy, not a trivial dup. Review which is canonical. |
| `runs/modal_migration/` | ~278 MB | `kuttakamina9895/` + `pilgnnteam/` Modal-download staging. Review whether already consumed into `runs/lightning_*`. |
| `runs/lightning_mag_benchmark_package/`, `..._connected_v3/`, `..._package_connected_v3/` | ~1.7 GB each | Each carries a ~1.5 GB `mag_hierarchies_type_rel_2000_fine5_finecov_v1.pt`. The three copies under benchmark packages + `runs/lightning_cache/kutta_v2_final_backup/` are byte-identical (~1.6 GB each, same size). **Keep the kutta backup copy (audit-canonical); the package copies are dedup candidates.** |
| `runs/lightning_production_benchmark_package_v1/`, `_v2_patched/`, `_connected_v3/` | ~793 MB each | Each holds a ~660 MB `data/Cora/.../data_undirected.pt` (regenerable PyG download). v1 likely superseded by v2_patched / connected_v3 — review. |
| `runs/lcr_mag_v3_probe/` vs `runs/lcr_mag_v4_probe/` | ~1.8 GB each | Successive overlap-cascade probes; **v3 is likely superseded by v4** (v4 is newer, 2026-06-21 16:29 vs v3 14:46, and has one more result file). Each contains its own 1.5 GB `prepared_hierarchy.pt`. Review: confirm v4 is the keeper, then v3's hierarchy `.pt` is a strong delete candidate. |
| `runs/combined/` | ~699 MB | De-duplicated combined CSV/log bundles from the June-15 audit, plus `logs_by_model/`. The combined summary CSVs are useful provenance; the per-model log copies may be redundant. Review. |
| `runs/lightning_cache/` (beyond `kutta_v2_final_backup/`) | ~1.7 GB total | `kutta_v2_final_backup/` is audit-canonical — KEEP. Any other subfolders here are review candidates. |
| `runs/lightning_completion/` (~351 MB), `runs/mrrf1070/` (~156 MB), `runs/lcr_arxiv_v3/` (~80 MB), `runs/lightning_connected_reruns/` (~91 MB), `runs/models/` (~31 MB) | as noted | Per-experiment outputs; small relative to the `.pt` caches. Review individually; likely retain results/logs, drop any large `.pt`. |
| `data/Physics/` (~1.2 GB), `data/Cora/` (~705 MB), `data/PubMed/` (~50 MB) | ~1.9 GB | Downloaded PyG datasets (git-ignored). Regenerable on next run but slow to re-download. Review: Physics was only smoke-tested (per final_submission_audit) and may be droppable; Cora is used by the canonical CORA benchmark. |

---

## Keep (do not touch)

| Path | Why |
| --- | --- |
| `paper/` (`*.tex`, `*.pdf`, `*.png`, `benchmark_summary.md`) | Canonical manuscript + figures (final_submission_audit names `paper/samplepaper.tex`/`.pdf`). |
| `models/*.pth` + `models/README.md` + `models/training_log_*.txt` | Trained model checkpoints (final + versioned). Includes the canonical arxiv/cora/mag jigsaw models and the GraphSAGE/FullCov variants referenced by the audits. |
| `runs/lightning_cache/kutta_v2_final_backup/` | Audit-designated canonical MAG RGCN artifact (checkpoint + overlap hierarchy + best-FullCov model + training log). |
| `runs/diagnostics/` | Final diagnostic CSVs/markdown (candidate shrinkage, partition stats, r1/r2 findings). Small (~106 KB), high value. |
| `runs/logs/*aggregate.md` and `*_summary.csv` named in `docs/final_submission_audit.md` | Each backs a specific paper table (fair_ablation, graphsage_standard/fullcov, mag_retrieval, retrieval timing, stitch/prefix-seed probes). |
| `benchmarks/paper_results/**` | Canonical paper benchmark bundle (final_*_summary.csv, glasgow_*_all.csv, diagnostics, manifest). |
| `benchmarks/` (tracked) | Committed benchmark CSVs allowed by `.gitignore` `!benchmarks/**`. |
| `src/`, `scripts/` (source `.py`/`.ipynb`/`.ps1`/`.sh`), `tests/`, `docs/` | Source code, runnable scripts, tests, and documentation. |
| `archive/` | Provenance archive from prior audits; the owner explicitly chose to archive rather than delete. Out of scope for this cleanup unless the owner reconsiders (it holds ~5.1 GB; see Uncertain). |
| `cache/cora_hierarchy.pkl` (~11 MB) | Cora hierarchy cache; cheap and used by Cora benchmarks. |
| `reviews.pdf` | Explicitly retained per prior audit. |

---

## Uncertain (confirm before any action)

| Path | Why uncertain |
| --- | --- |
| `archive/non_submission_20260620/` (~5.1 GB) | The dominant chunk of `archive/`. Contains `run_partials_and_probes/` (partial/spot runs, rolling `*_partial_per_query.csv`, latex aux) and `paper_clutter/`. By the "safe-delete" rubric these partials/aux would qualify, BUT they live under `archive/`, which the owner deliberately created to preserve provenance and chose not to delete in prior audits. **Do not delete without explicit owner confirmation** — this is the single largest gray-area reclaim (~5 GB). |
| `runs/migration/mag_darkphoenix/*.clean.pt` vs `*.pt` | Different MD5 hashes, so not a trivial duplicate. Unclear which is the authoritative cleaned hierarchy. Confirm before deleting either. |
| `runs/lcr_mag_v3_probe/` | Likely superseded by `v4_probe`, but both are very recent (today). Confirm v4 fully replaces v3 before deleting v3's 1.5 GB hierarchy. |
| `runs/lightning_mag_benchmark_code_overlay_v1..v16/` (15 dirs, ~3 MB total) | Tiny code-overlay snapshots used to patch Modal runs. Low space value; keep unless owner confirms the overlays are obsolete. Not worth deleting for space. |
| `runs/lightning_production_benchmark_package_v1/` | Superseded-version candidate (v2_patched / connected_v3 exist), but production benchmark provenance — confirm the newer packages reproduce its results before deleting its 660 MB Cora `.pt`. |
| `data/Physics/` (~1.2 GB) | Only smoke-tested per final_submission_audit; probably not needed for the paper, but confirm no pending Physics run depends on it. |
| `.venv_modal/` | Safe in principle (regenerable env), but deleting breaks local Modal tooling until rebuilt — confirm the owner can recreate it. |

---

## Reclaimable-space estimate

- **Safe to delete (excluding `.venv_modal`):** ~1.2 GB
  (tmp ~68 MB + quarantine ~182 MB + render/visual/whitepaper ~5.5 MB + smoke ~1.5 MB
  + pyc/latex aux negligible; note most `.pyc` are inside `.venv_modal`).
- **Safe-but-confirm (`.venv_modal`):** +~864 MB.
- **Review (regenerable `.pt` caches + duplicated hierarchies), realistic dedup target:**
  on the order of **15–20 GB** — dominated by the 17 `prepared_hierarchy.pt` (~11.4 GB),
  41 `signature_tokens.pt` (~4.7 GB), and the duplicated `mag_hierarchies_*.pt` /
  Cora `data_undirected.pt` copies. This is where the real space is, but it needs
  per-item confirmation.
- **Uncertain (archive partials, ~5 GB):** only with explicit owner sign-off.
