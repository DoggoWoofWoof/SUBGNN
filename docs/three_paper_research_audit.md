# Three-Paper Final Research Audit

Date: 2026-07-11

Scope:

- `paper/samplepaper.tex` / `paper/samplepaper.pdf`
- `paper/jigsaw_log2026.tex` / `paper/jigsaw_log2026.pdf`
- `paper/jigsaw_ecmlpkdd.tex` / `paper/jigsaw_ecmlpkdd.pdf`

I audited and repaired the three manuscripts as research papers: argument flow,
internal consistency, cross-version consistency, provenance of load-bearing
numbers, venue/readability risk, and rendered PDF health.

## Final Verdict

The three papers now carry a consistent and defensible story:

- Jigsaw is a retrieval-constrained exact-verification system.
- Exactness is guaranteed inside the retrieved candidate; global completeness is
  controlled by retrieval coverage and budget.
- The central systems claim is bounded-memory exact verification without
  whole-graph residence, not faster resident matching.
- Cora and Arxiv are label-rich sanity/cost regimes; MAG is the real stress case.
- The deployed MAG result is the walk-aware 88.6% row, with FilterAll framed as
  an upper-bound ceiling rather than a scalable learned policy.

The highest-risk inconsistencies found in the first pass have been fixed across
the relevant manuscripts and rebuilt into the PDFs.

## Fixes Applied

1. Fixed the rendered LoG typo `99.3o98.7` by restoring the intended
   `99.3 -> 98.7` math arrow.

2. Corrected the query-family saturation claim in all three versions. The text
   now says single and multi-fine templates saturate, while local K-hop variants
   remain high but not perfect on MAG.

3. Removed the LoG Cora budget contradiction. The appendix now says the curve
   rises steeply to 100% by the full Cora budget, matching the reported
   `64.0/82.7/94.2/100` progression.

4. Reframed memory/latency captions away from resident-vs-resident speed. The
   papers now describe cold-load end-to-end latency and explicitly state that
   the edge is load avoidance, not faster resident matching.

5. Clarified the 2.4 GB memory headline as a conservative streaming serve and
   kept the 1.9 GB figure scoped to the optimized frontier.

6. Replaced ambiguous provenance wording with explicit references to
   `HEADLINE_NUMBERS.csv` and `CANONICAL_SOURCES.md`; legacy normalized
   summaries are now described as provenance/diagnostic artifacts, not the
   headline source of truth.

7. Clarified the ambiguous "all three retrievers" phrase as the learned
   retrievers for Cora, Arxiv, and MAG.

8. Tightened ECML and LNCS wording enough to remove new overfull boxes while
   preserving the claims.

9. Added and preserved the final acceptance de-riskers from the text-fix pass:
   out-of-core/distributed prior art is cited and differentiated, the 0/15
   infeasibility claim is scoped to the CP-based Glasgow solver, label hashing
   is framed as exact hashed-label semantics rather than raw-attribute equality,
   and offline partitioning/training/embedding cost is called out as an
   amortized static-graph assumption.

10. Compressed the ECML abstract, related-work paragraph, and limitations
    paragraph enough to recover the 16-page submission target without removing
    the new prior-art and limitation caveats.

## Verification

Builds:

- `samplepaper.pdf`: rebuilt successfully, 31 pages.
- `jigsaw_log2026.pdf`: rebuilt successfully, 18 pages.
- `jigsaw_ecmlpkdd.pdf`: rebuilt successfully, 16 pages.

Log checks:

- No LaTeX errors.
- No unresolved references.
- No unresolved citations.
- No rerun warnings after the second compile.
- No overfull boxes after the final wording fixes.

Rendered PDF text checks:

- No `99.3o98.7`.
- No stale "both systems resident" caption language.
- No stale "in-memory per-query latency" claim.
- No stale "single, K-hop, degree-K-hop and multi-fine saturate" claim.
- No stale "all three retrievers" phrase.
- No stale "normalized paper bundle" source-of-truth phrase.

Rendered layout checks:

- Full contact sheets for all three PDFs showed no blank pages, clipped figure
  pages, missing plots, or page-count spillover.
- Focused render checks covered abstracts, production-matrix provenance,
  memory/latency captions, and query-family captions.

## Residual Notes

The remaining LaTeX warnings are low-risk typesetting warnings:

- `amsmath` warns that it cannot redefine `\vec`, inherited from the class/package
  stack.
- Some underfull boxes remain in dense paragraphs and figure-heavy pages.
- LoG reports several float specifier changes from `h` to `ht`, which is normal
  for its layout.
- MiKTeX prints a local update reminder; it is environmental, not a manuscript
  issue.

No remaining issue blocks the three PDFs from being used as clean reviewed
artifacts. Further improvement would be editorial rather than corrective:
shorten the abstracts and metric-heavy paragraphs if a stricter venue page/style
constraint becomes the priority.
