# OpenReview Submission Metadata

**Keywords:** subgraph isomorphism, exact subgraph matching, graph retrieval, graph neural networks, coverage-conditioned training, constraint programming, scalable graph algorithms, retrieval-augmented verification

**TLDR:** Jigsaw uses a FullCov-trained GNN to choose which graph regions Glasgow should load, enabling 88.6% exact recovery on full MAG under 2.4 GB residence, while direct Glasgow solves 0/15 and whole-graph residence uses 10.2 GB.

**Abstract:** Exact subgraph-isomorphism solvers such as Glasgow are sound and complete, but million-node attributed graphs are difficult to serve: continuous features must be mapped to discrete matching labels, and whole-graph residence can exceed memory or leave the verifier an unmanageable search domain. Under our tested configuration, direct CP-based Glasgow solves 0 of 15 sampled queries on full OGBN-MAG (1.9M nodes) and 0/15 on OGBN-Arxiv (169K), while solving 15/15 on Cora (19.8K) in approximately 2.7 s. Jigsaw turns learned graph retrieval into an execution layer for exact search. A GNN trained with FullCov, an objective that rewards coverage of every partition touched by a match, ranks hierarchical graph partitions. Jigsaw loads a bounded set, adds only one-hop boundary nodes from neighboring partitions, prunes by typed signatures and exact labels, and gives the resulting candidate to Glasgow. Any accepted match is exact inside that candidate. Because retrieval and verification are separate, Jigsaw can select a learned or classical ranker from measured label selectivity. A classical label index leads under near-unique labels (96.7% versus 88.6% for the learned ranker), while learned retrieval becomes the better partition ranker as labels coarsen. FullCov improves fixed-budget coverage on locked Arxiv K-hop tests (FullCov@50 39.0% to 48.8%, FullCov@100 66.0% to 82.0%; exact McNemar p = 9.8 x 10^-6, 4.0 x 10^-10). Across Cora, Arxiv, and MAG (two seeds, eight families, three sizes, six policies), Jigsaw recovers 88.6% of MAG positives with learned retrieval and 98.4% with exhaustive partition filtering; audited no-match accuracy is 99.3-100.0%. At matched half-partition budgets, boundary overlap nearly doubles MAG recovery (44.4% to 88.9%), while query-derived pruning provably preserves every covered match. Candidate construction, not exact solving, dominates latency. A two-pass streaming serve keeps only a partition cache resident, using 2.4 GB on MAG instead of 10.2 GB for whole-graph residence (4.2x less).

**PDF:** [pdf](https://openreview.net/attachment?id=O5YdNfGW2P&name=pdf)

**Submission Type:** Full paper proceedings track submission (max 9 main pages).

**Email Sharing:** We authorize the sharing of all author emails with Program Chairs.

**Data Release:** We authorize the release of our submission and author names to the public in the event of acceptance.
