# Jigsaw Loss And Retrieval Method

This document describes the implemented Jigsaw training objective and retrieval
pipeline in mathematical detail. It also explains why each component was
introduced, gives worked examples, and separates the final method from
exploratory experiments.

The source of truth is:

- [`scripts/train_jigsaw_model.py`](../scripts/train_jigsaw_model.py): training,
  query generation, live-positive encoding, and checkpoint selection
- [`scripts/coverage_losses.py`](../scripts/coverage_losses.py): FullCov-aligned
  partition coverage objective
- [`scripts/retrieval_strategies.py`](../scripts/retrieval_strategies.py):
  deterministic dynamic retrieval and reciprocal-rank fusion
- [`scripts/benchmark_retrieval.py`](../scripts/benchmark_retrieval.py):
  retrieval-only evaluation and metrics

## 1. Problem Definition

The full graph is partitioned hierarchically:

- coarse partitions are the units retrieved before verification;
- each coarse partition contains several fine partitions;
- the weighted coarse-partition graph records how strongly coarse partitions
  touch in the original graph.

For a query graph \(Q\), let:

- \(f_\theta(G) \in \mathbb{R}^{128}\) be the normalized graph embedding
  produced by the shared GIN encoder;
- \(z_q = f_\theta(Q)\) be the query embedding;
- \(P(Q)\) be the set of all coarse partitions containing query nodes;
- \(R_K(Q)\) be the first \(K\) retrieved coarse partitions.

The primary retrieval objective is **FullCov@K**:

\[
\operatorname{FullCov@K}(Q)
=
\mathbf{1}\left[P(Q) \subseteq R_K(Q)\right].
\]

Partition recall is secondary:

\[
\operatorname{Recall@K}(Q)
=
\frac{|P(Q) \cap R_K(Q)|}{|P(Q)|}.
\]

Recall and FullCov are not interchangeable. If a query touches ten true
partitions and retrieval finds nine, recall is \(0.9\), but FullCov is zero.
The missing partition may contain the only valid completion of the query, so
exact verification cannot succeed in the retrieved region.

### Method at a glance

The method solves two related but different problems:

1. **Training:** use known query-to-partition membership to teach the encoder
   to rank every required partition highly, with special pressure on the
   weakest required partition.
2. **Retrieval:** at inference time, use only the learned neural ranking and
   the graph between partitions. The true required partitions are hidden and
   used only afterward to calculate FullCov and recall.

In plain terms, InfoNCE learns what a generally relevant region looks like.
The coverage loss then asks, "which required partition is currently easiest
to miss?" and concentrates training pressure there. Dynamic retrieval starts
from the model's top 20 and cautiously follows strong partition boundaries,
while periodically returning to the neural ranking so it does not drift too
far locally.

## 2. Encoder

The final encoder is a shared six-layer residual GIN:

1. input features are projected to 256 hidden dimensions;
2. each GIN layer uses a two-layer MLP, residual connection, layer
   normalization, ReLU, and dropout \(0.1\);
3. mean, max, and sum pooling are applied to every GIN layer;
4. all 18 pooled vectors are concatenated;
5. a residual readout MLP produces a normalized 128-dimensional graph
   embedding.

Because embeddings are normalized,

\[
\lVert z_q - z_p \rVert_2^2
=
2 - 2 z_q^\top z_p.
\]

Therefore, the dot-product similarities optimized during training and the L2
distances used by FAISS at retrieval time induce the same ranking.

## 3. Final Training Objective

The implemented total objective is:

\[
\mathcal{L}_{\text{total}}
=
0.2\,\mathcal{L}_{\text{fine-NCE}}
+
0.8\,\mathcal{L}_{\text{coarse-NCE}}
+
1.0\,\mathcal{L}_{\text{coverage}}.
\]

The final configuration deliberately disables two exploratory terms:

\[
\beta_{\text{node}} = 0,
\qquad
\gamma_{\text{fine-coverage}} = 0.
\]

Thus node-level alignment and fine-partition coverage are not part of the
adopted method.

Final Arxiv parameters:

| Parameter | Value |
| --- | ---: |
| Fine/coarse InfoNCE mixture | \(0.2 / 0.8\) |
| Coverage weight | \(1.0\) |
| Coverage temperature | \(0.05\) |
| Positive aggregation | CVaR |
| CVaR fraction | \(0.25\) |
| Initial coverage K | \(20\) |
| K bucket increment | \(10\) |
| Top-K barrier weight | \(0.35\) |
| Top-K margin | \(0.0\) |
| Maximum live positive partitions per batch | \(24\) |
| Node loss weight | \(0.0\) |
| Fine-partition coverage weight | \(0.0\) |

### 3.1 Hierarchical InfoNCE

Each training sample contains:

- a query graph \(Q_i\);
- a fine or stitched positive graph \(F_i\);
- a coarser positive graph \(C_i\);
- optional hard-negative coarse graphs \(H_{i,h}\).

For a query-positive pair, the implementation uses:

\[
\operatorname{sim}(a,b)
=
\frac{a^\top b}{\tau_{\text{NCE}}},
\qquad
\tau_{\text{NCE}} = 0.1.
\]

The fine InfoNCE loss uses the paired fine positive and other batch positives
as negatives:

\[
\mathcal{L}_{\text{fine-NCE},i}
=
-\log
\frac{\exp(\operatorname{sim}(z_{q_i},z_{f_i}))}
{\exp(\operatorname{sim}(z_{q_i},z_{f_i}))
+
\sum_{j\ne i}\exp(\operatorname{sim}(z_{q_i},z_{f_j}))}.
\]

The coarse InfoNCE loss additionally contains explicit hard-negative coarse
partitions:

\[
\mathcal{L}_{\text{coarse-NCE},i}
=
-\log
\frac{\exp(\operatorname{sim}(z_{q_i},z_{c_i}))}
{\exp(\operatorname{sim}(z_{q_i},z_{c_i}))
+
\sum_{j\ne i}\exp(\operatorname{sim}(z_{q_i},z_{c_j}))
+
\sum_h\exp(\operatorname{sim}(z_{q_i},z_{h_{i,h}}))}.
\]

This objective teaches broad representation quality: a query should resemble
its valid fine and coarse regions more than unrelated or neighboring regions.
However, it does not explicitly require **every** partition touched by a
multi-partition query to rank highly. That missing requirement motivates the
coverage objective.

### 3.2 All-Positive Partition Coverage

Let all coarse-partition embeddings be
\(\{p_1,\ldots,p_M\}\). For query \(Q\), define the temperature-scaled score:

\[
s_j = \frac{z_q^\top p_j}{\tau_{\text{cov}}},
\qquad
\tau_{\text{cov}} = 0.05.
\]

Let \(P=P(Q)\) be the set of required positive partition IDs. For every
\(j\in P\), the loss first computes a multi-class cross-entropy term:

\[
c_j
=
-\log
\frac{\exp(s_j)}
{\sum_{m=1}^{M}\exp(s_m)}.
\]

Unlike a single-positive or positive-log-sum-exp objective, this produces a
separate loss for every required partition. One strong positive cannot
completely hide a missing weak positive.

The objective also compares every positive with the hardest negative:

\[
h = \max_{n\notin P}s_n,
\]

\[
m_j = \operatorname{softplus}(h-s_j)
=
\log\left(1+\exp(h-s_j)\right).
\]

The base per-positive coverage term is:

\[
\ell_j = c_j + 0.25\,m_j.
\]

The cross-entropy distributes probability mass toward each required
partition. The margin term directly penalizes any required partition that
falls below the strongest incorrect partition.

### 3.3 Worst-Positive CVaR Aggregation

If a query touches many partitions, averaging their losses can let several
easy positives dilute one badly ranked positive. FullCov fails because of the
single worst positive, so the final objective emphasizes the worst positive
losses.

For \(p=|P|\) positives and CVaR fraction \(\rho=0.25\), define:

\[
r = \max(1,\lceil \rho p\rceil).
\]

Let \(\operatorname{TopLargest}_r(\{\ell_j:j\in P\})\) be the \(r\) largest
per-positive losses. Then:

\[
\mathcal{L}_{\text{base-coverage}}
=
\frac{1}{r}
\sum_{\ell\in\operatorname{TopLargest}_r}\ell.
\]

With one to four true partitions, CVaR selects only the single weakest
positive. With eight true partitions, it averages the two weakest. This
closely matches FullCov's bottleneck behavior.

CVaR alone was not sufficient in the matched ablation. It becomes useful only
when combined with the rank barrier and live positive embeddings.

### 3.4 FullCov Top-K Barrier

The base coverage loss improves positive scores, but it does not explicitly
express the ranking condition required by FullCov@K.

Suppose a query has \(p\) required positives and the desired retrieval budget
is \(K\), with \(p\le K\). For all positives to appear inside the first \(K\)
positions, the weakest positive must outrank the

\[
(K-p+1)\text{-th highest negative}.
\]

Why: the top K can contain at most \(K-p\) negatives if it must also contain
all \(p\) positives. Therefore, every positive should beat the next negative,
the \((K-p+1)\)-th highest one.

Let:

\[
t_K
=
\operatorname{kthLargest}
\left(
\{s_n:n\notin P\},
\ K-p+1
\right).
\]

If fewer than \(K-p+1\) negatives exist, the implementation safely uses the
lowest available negative as the threshold. This matters only when the
requested retrieval budget approaches the total partition count.

For each positive:

\[
b_j
=
\operatorname{softplus}(t_K + \delta - s_j),
\]

where the final margin is \(\delta=0\). These barrier terms are aggregated
using the same worst-positive CVaR rule:

\[
\mathcal{L}_{\text{barrier}}
=
\operatorname{CVaR}_{0.25}(\{b_j:j\in P\}).
\]

The complete coverage objective is:

\[
\boxed{
\mathcal{L}_{\text{coverage}}
=
\operatorname{CVaR}_{0.25}(\{\ell_j:j\in P\})
+
0.35\,
\operatorname{CVaR}_{0.25}(\{b_j:j\in P\})
}.
\]

This objective uses ground-truth partition membership only as a supervised
training target. It does not make the truth available to the retrieval
algorithm.

#### Queries broader than K

If a query touches more than K partitions, FullCov@K is mathematically
impossible. Such rows are not dropped. Instead, the effective K grows in
buckets:

\[
K_{\text{eff}}
=
\begin{cases}
K_0, & p\le K_0,\\
\left\lceil \frac{p}{B}\right\rceil B, & p>K_0,
\end{cases}
\]

where Arxiv uses \(K_0=20\) and bucket size \(B=10\).

Examples:

- \(p=17 \Rightarrow K_{\text{eff}}=20\);
- \(p=23 \Rightarrow K_{\text{eff}}=30\);
- \(p=37 \Rightarrow K_{\text{eff}}=40\).

This preserves a meaningful FullCov boundary for broad training queries
instead of skipping them.

### 3.5 Live Positive Re-Encoding

Computing all partition embeddings with gradients every step would be
expensive. Jigsaw therefore keeps a detached cache of every coarse-partition
embedding and refreshes it periodically.

If the coverage loss used only cached embeddings:

\[
p_j = \operatorname{stopgrad}(f_\theta(G_j)),
\]

then its gradient would update the query encoder path but not the positive
partition path:

\[
\frac{\partial\mathcal{L}_{\text{coverage}}}{\partial\theta}
\text{ flows mainly through }z_q.
\]

The final objective selects up to 24 true partitions that appear most
frequently in the current batch, re-encodes them with gradients enabled, and
replaces their cached rows:

\[
\tilde p_j =
\begin{cases}
f_\theta(G_j), & j\text{ selected as a live positive},\\
\operatorname{stopgrad}(p_j^{\text{cache}}), & \text{otherwise}.
\end{cases}
\]

Now the loss updates both sides:

\[
\frac{\partial\mathcal{L}_{\text{coverage}}}{\partial\theta}
=
\left(
\frac{\partial\mathcal{L}}{\partial z_q}
\right)
\left(
\frac{\partial z_q}{\partial\theta}
\right)
+
\sum_{j\in P_{\text{live}}}
\left(
\frac{\partial\mathcal{L}}{\partial \tilde p_j}
\right)
\left(
\frac{\partial \tilde p_j}{\partial\theta}
\right).
\]

This is the key difference between merely asking the query embedding to chase
stale partition vectors and jointly teaching the encoder how required
partitions should be represented.

## 4. Worked Loss Example

Consider five partitions with already temperature-scaled scores:

| Partition | Required? | Score |
| --- | --- | ---: |
| A | yes | 3.0 |
| B | yes | 0.5 |
| C | no | 2.0 |
| D | no | 1.0 |
| E | no | -1.0 |

The softmax denominator is:

\[
Z=e^3+e^{0.5}+e^2+e^1+e^{-1}\approx32.21.
\]

Per-positive cross-entropies:

\[
c_A=-\log(e^3/Z)\approx0.472,
\]

\[
c_B=-\log(e^{0.5}/Z)\approx2.972.
\]

The hardest negative is C with score \(h=2.0\). Therefore:

\[
m_A=\operatorname{softplus}(2.0-3.0)\approx0.313,
\]

\[
m_B=\operatorname{softplus}(2.0-0.5)\approx1.701.
\]

Base per-positive terms:

\[
\ell_A=0.472+0.25(0.313)\approx0.550,
\]

\[
\ell_B=2.972+0.25(1.701)\approx3.397.
\]

With two positives and CVaR fraction \(0.25\), the loss selects the single
worst positive, B:

\[
\mathcal{L}_{\text{base-coverage}}=3.397.
\]

Now request FullCov@3. Since \(p=2\):

\[
K-p+1=3-2+1=2.
\]

The second-highest negative is D with threshold \(t_K=1.0\):

\[
b_A=\operatorname{softplus}(1.0-3.0)\approx0.127,
\]

\[
b_B=\operatorname{softplus}(1.0-0.5)\approx0.974.
\]

CVaR again selects B, so:

\[
\mathcal{L}_{\text{coverage}}
\approx
3.397+0.35(0.974)
=
3.738.
\]

The easy positive A does not hide B. The gradient focuses on lifting B above
the exact negative threshold required for all positives to fit inside top 3.

For an illustrative contrastive component, suppose:

\[
\mathcal{L}_{\text{fine-NCE}}=0.049,
\qquad
\mathcal{L}_{\text{coarse-NCE}}=0.680.
\]

Then:

\[
\mathcal{L}_{\text{total}}
=
0.2(0.049)+0.8(0.680)+3.738
\approx4.292.
\]

## 5. Retrieval Runtime Pipeline

The primary retrieval pipeline never reads true partition IDs, query labels,
or solver output while selecting candidates. Truth is used only afterward to
measure coverage.

### Step 1: Build the partition index

For every coarse partition \(G_j\):

\[
p_j=f_\theta(G_j).
\]

The normalized embeddings are stored in a FAISS `IndexFlatL2` index. The
coarse-partition graph is also built:

- one node per coarse partition;
- an edge between partitions that touch in the original graph;
- edge weight equal to the number of cross-partition graph edges.

### Step 2: Encode the query

The retrieval benchmark uses fixed-seed, connected 20-node query blobs sampled
inside three-hop neighborhoods. The query is encoded once:

\[
z_q=f_\theta(Q).
\]

### Step 3: Produce the complete neural ranking

FAISS ranks every coarse partition by ascending L2 distance:

\[
\pi_{\text{neural}}
=
\operatorname{argsort}_j \lVert z_q-p_j\rVert_2^2.
\]

Because embeddings are normalized, this is equivalent to descending dot
product. The complete ranking is retained so dynamic retrieval can later
teleport beyond the initial seed.

### Step 4A: Fixed neural retrieval, the primary model-quality result

Fixed retrieval simply returns:

\[
R_K(Q)=\pi_{\text{neural}}[1:K],
\]

for K \(=20,50,100\) on Arxiv.

This is the cleanest measurement of the learned model because no graph
expansion or extra retriever changes the order.

### Step 4B: Locked dynamic retrieval, the primary efficient system result

The locked dynamic method is:

`coarse_hybrid_mw0.5_teleport10`

It starts from the neural top 20 and expands to budgets 50, 75, or 100 using
the weighted coarse-partition graph.

Let \(S\) be the selected set. For an unselected frontier partition \(u\),
define:

#### Neural rank score

For zero-based neural rank \(r(u)\) among \(M\) partitions:

\[
\operatorname{model}(u)
=
1-\frac{r(u)}{M-1}.
\]

#### Boundary weight

For every selected neighbor \(v\in S\), let \(w_{vu}\) be the number of
cross-partition edges:

\[
W(u)
=
\sum_{v\in S\cap N(u)}\log(1+w_{vu}).
\]

The logarithm prevents one extremely heavy edge from completely dominating.

#### Boundary support

\[
C(u)=|S\cap N(u)|.
\]

At each expansion step, frontier weight and support are normalized by the
largest current frontier values:

\[
\widehat W(u)=\frac{W(u)}{\max_x W(x)},
\qquad
\widehat C(u)=\frac{C(u)}{\max_x C(x)}.
\]

The combined boundary score is:

\[
\operatorname{boundary}(u)
=
0.8\widehat W(u)+0.2\widehat C(u).
\]

The locked hybrid score uses equal neural and boundary weight:

\[
\operatorname{hybrid}(u)
=
0.5\operatorname{model}(u)
+
0.5\operatorname{boundary}(u).
\]

The highest-scoring frontier partition is added. Every tenth addition, the
algorithm ignores the frontier and adds the highest-ranked unselected neural
partition. This **teleport** opens a new frontier and prevents the expansion
from becoming trapped around an imperfect initial top-20 seed.

If the frontier becomes empty, the algorithm also falls back to the best
remaining neural partition.

### Dynamic retrieval pseudocode

```text
selected = first 20 partitions in the neural ranking
frontier = graph neighbors of selected

while len(selected) < requested_budget:
    if this is every tenth expansion:
        next = best unselected partition in the neural ranking
    else if frontier is not empty:
        score each frontier partition using:
            0.5 * neural_rank_score
          + 0.5 * normalized_boundary_score
        next = highest-scoring frontier partition
    else:
        next = best unselected partition in the neural ranking

    add next to selected
    update frontier weights and support counts

return selected
```

The procedure never checks whether a candidate is truly required. Its only
signals are neural rank, cross-partition edge weight, and how many selected
partitions support the candidate.

## 6. Worked Dynamic Retrieval Example

The real method seeds 20 partitions and teleports every ten additions. The
following smaller example uses seed size 3 and teleport interval 3 so the
calculation fits on the page; the algorithm is otherwise identical.

Assume eight partitions with neural order:

\[
[P_1,P_2,P_3,P_5,P_7,P_4,P_6,P_8].
\]

The seed set is:

\[
S=\{P_1,P_2,P_3\}.
\]

Suppose the frontier contains \(P_4,P_5,P_6\) with the following edges from
the selected set:

- \(P_4\): edge weights 9 from \(P_1\) and 3 from \(P_2\);
- \(P_5\): edge weight 2 from \(P_2\);
- \(P_6\): edge weight 15 from \(P_3\).

Boundary weights:

\[
W(P_4)=\log(10)+\log(4)\approx3.689,
\]

\[
W(P_5)=\log(3)\approx1.099,
\]

\[
W(P_6)=\log(16)\approx2.773.
\]

Support counts:

\[
C(P_4)=2,\quad C(P_5)=1,\quad C(P_6)=1.
\]

With \(M=8\), their neural rank scores are approximately:

\[
\operatorname{model}(P_4)=1-5/7=0.286,
\]

\[
\operatorname{model}(P_5)=1-3/7=0.571,
\]

\[
\operatorname{model}(P_6)=1-6/7=0.143.
\]

After normalizing frontier weight and support:

\[
\operatorname{boundary}(P_4)=1.000,
\]

\[
\operatorname{boundary}(P_5)\approx0.338,
\]

\[
\operatorname{boundary}(P_6)\approx0.702.
\]

Locked equal-weight hybrid scores:

\[
\operatorname{hybrid}(P_4)\approx0.643,
\]

\[
\operatorname{hybrid}(P_5)\approx0.455,
\]

\[
\operatorname{hybrid}(P_6)\approx0.422.
\]

The method selects \(P_4\), even though \(P_5\) has the better neural rank,
because \(P_4\) has stronger and broader boundary evidence. After adding
\(P_4\), its neighbors enter or gain weight in the frontier. On every third
addition in this small example, the method teleports to the best remaining
neural candidate, preventing purely local drift.

## 7. Metrics Example

Suppose the true partitions are:

\[
P(Q)=\{P_1,P_4,P_7\},
\]

and top-5 retrieval returns:

\[
R_5(Q)=\{P_1,P_3,P_4,P_6,P_2\}.
\]

Then:

\[
\operatorname{Recall@5}=\frac{2}{3},
\]

but:

\[
\operatorname{FullCov@5}=0
\]

because \(P_7\) is missing.

If \(P_7\) appears at rank 8, the maximum true-partition rank is:

\[
\max_{p\in P(Q)}\operatorname{rank}(p)=8.
\]

This metric directly answers the smallest fixed K that could achieve FullCov
for that query.

## 8. Optional And Exploratory Retrieval Variants

These methods are implemented and useful as diagnostics, but they are not the
primary model-quality claim.

### Global-fine parent ranking

Every fine partition is embedded and ranked globally. Fine IDs are mapped to
their coarse parents, preserving first appearance. This creates a second
coarse ranking that may recover partitions whose local fine structure matches
the query.

### Reciprocal-rank fusion

Multiple rankings can be combined without using truth:

\[
\operatorname{RRF}(u)
=
\sum_{\pi}
\frac{1}{c+\operatorname{rank}_{\pi}(u)+1},
\qquad c=20.
\]

RRF is used for coarse/fine consensus and optional cross-model ensembles.

### Cross-model fusion

Rankings from multiple checkpoints can be fused. This can slightly improve
coverage, but it multiplies model-loading and index-building cost. It must be
reported as an optional high-cost system, not the default Jigsaw method.

### Multi-view query fusion

The same model was also tested by encoding connected fragments of each query
and fusing their partition rankings. It produced small, unstable gains at
selected budgets but did not consistently beat the complete-query fixed
ranking. The full ablation and paired analysis are in
[`multiview_retrieval_ablation.md`](multiview_retrieval_ablation.md).

### Fine-boundary and label-pruned verification

Fine-boundary expansion and label pruning helped diagnose whether retrieval or
verification caused failures. Label-based pruning should not be presented as
the learned retrieval contribution because it uses extra semantic
information. It is a verification optimization.

### Glasgow

Glasgow is not used to select the final model. It is run only after retrieval
in a small end-to-end verification table. It is exact inside the retrieved
candidate region; it cannot recover a solution from a missing partition.

### Progressive overlap-and-pruning cascade

The MAG rescue path adds an exact-verification cascade after the normal neural
ranking. It keeps the partition count fixed and increases the candidate budget
only if the previous bucket fails:

```text
ranking = FAISS(query_embedding, partition_embedding_index)
for budget in [20, 50, 100]:
    selected = top budget partitions from ranking
    candidate = selected partition nodes + one-hop boundary overlap
    candidate = prune by query-compatible signature tokens
    candidate = prune by exact query labels
    if exact verifier finds a match:
        stop and report this budget
```

FAISS is only the ranker. It stores the offline partition embeddings and
returns the nearest partitions to the query embedding. In our current small
partition indexes this is `IndexFlatL2`, so it computes exact vector distances
over all coarse partitions rather than proving graph containment. The exact
solver still determines whether the final candidate subgraph contains a match.

Why the cascade helps MAG:

- hard partition FullCov is too strict when true nodes lie just outside a
  retrieved partition boundary;
- one-hop overlap recovers many boundary nodes without changing the 2,000
  partition count;
- signature pruning cuts the overlap candidate from hundreds of thousands of
  nodes;
- exact label pruning cuts it further before Glasgow/VF2-style verification.

Verified best MAG cascade result, 100 locked 20-node k-hop queries:

| Stage | Result |
| --- | ---: |
| Solved at budget 20 | 33/100 |
| Additional solved at budget 50 | 12/100 |
| Additional solved at budget 100 | 9/100 |
| Total solved | 54/100 |
| Avg overlap nodes | 338,036 |
| Avg signature nodes | 51,067 |
| Avg label-pruned nodes | 21,659 |
| Avg retrieval time | 7.6 ms |
| Avg candidate build+prune time | 7.48 s |
| Avg solver time | 7.09 s |

This also answers whether retrieval is doing anything. On the same MAG setting,
random K=50 contains only 2/100 20-node queries, while filtering all 2,000
partitions contains 100/100 but leaves about 2.4M candidate nodes. The useful
pipeline is therefore retrieval first, pruning second, exact verification
third.

Verified Arxiv cascade result, 100 locked 20-node k-hop queries:

| Stage | Result |
| --- | ---: |
| Solved at budget 20 | 85/100 |
| Additional solved at budget 50 | 9/100 |
| Additional solved at budget 100 | 5/100 |
| Total solved | 99/100 |
| Avg label-pruned nodes | 23.3 |
| Avg retrieval time | 5.0 ms |
| Avg candidate build+prune time | 108 ms |
| Avg solver time | 24.7 ms |

Cora with the existing model solves 100/100 at K=20 for query sizes 20, 50,
and 100. This is expected because Cora has only 20 coarse partitions, so it is
a timing/sanity check rather than a retrieval-quality result.

## 9. Development Progression

The retrieval method evolved as follows:

1. **Fixed coarse top-K:** exposed that K=20 often under-covered multi-coarse
   and k-hop queries.
2. **Larger fixed K:** K=50 and K=100 improved recall and FullCov, but broad
   candidate regions increased downstream verification cost.
3. **Fine-boundary and label-pruned diagnostics:** showed that more structured
   stitching could recover many queries, but high budgets and label assistance
   were not a satisfactory primary claim.
4. **Top-20 boundary expansion:** implemented the intended idea of starting
   small and expanding through neighboring partitions.
5. **Hybrid boundary plus neural ranking:** prevented graph-local evidence from
   overriding the learned model.
6. **Periodic neural teleport:** prevented boundary expansion from becoming
   trapped around a flawed seed.
7. **Global-fine and cross-model consensus:** tested complementary signals, but
   added complexity or cost and did not replace the primary single-model
   result.
8. **FullCov-aligned final objective:** improved the underlying fixed neural
   ranking, reducing dependence on retrieval heuristics.

## 10. What The Final Results Establish

The causal matched-budget comparison uses models trained from scratch for the
same 9,000 optimizer steps and evaluated on the same 200 locked queries.

### Fixed neural ranking

| Objective | FullCov@20 | FullCov@50 | FullCov@100 |
| --- | ---: | ---: | ---: |
| Mean-positive control | 87/400 (21.8%) | 156/400 (39.0%) | 264/400 (66.0%) |
| Final objective | **104/400 (26.0%)** | **195/400 (48.8%)** | **328/400 (82.0%)** |

The denominator is 400 because two training replicates are each evaluated on
the same 200 queries. Paired exact McNemar tests give:

- K=20: \(p=0.0046\);
- K=50: \(p=9.8\times10^{-6}\);
- K=100: \(p=4.0\times10^{-10}\).

The final objective clearly improves the learned fixed ranking.

### Locked dynamic retrieval

| Objective | FullCov B=50 | FullCov B=75 | FullCov B=100 |
| --- | ---: | ---: | ---: |
| Control | 171/400 (42.8%) | 248/400 (62.0%) | 315/400 (78.8%) |
| Final objective | **187/400 (46.8%)** | **264/400 (66.0%)** | **321/400 (80.2%)** |

The dynamic gains are smaller and are not statistically clear in the paired
comparison. This is why fixed ranking is the primary model-quality result and
dynamic retrieval is the secondary efficient-system result.

### Final versus strongest checkpoint

The clean final objective is the strongest **fairly tested training method**.
The V7 continuation checkpoint is still the strongest **existing operational
checkpoint** because it inherited earlier trained weights and received more
total optimization:

- use clean `final_seed7101_best_fullcov` for causal objective claims and new
  from-scratch datasets;
- use `coverage_v7_continue_from_v6_best_fullcov` as the separately labeled
  strongest staged-system checkpoint.

## 11. Final Method Summary

The defensible final Jigsaw story is:

1. encode queries and partitions with a shared residual GIN;
2. learn general region similarity with hierarchical InfoNCE;
3. explicitly optimize every required coarse partition;
4. focus gradients on the weakest required partitions with CVaR;
5. push those weak positives above the exact FullCov@K negative boundary;
6. re-encode a bounded number of true partitions live so coverage gradients
   update both query and partition representations;
7. report fixed neural ranking as the primary learned-model result;
8. optionally expand from top 20 using neural-boundary hybrid retrieval with
   periodic teleports;
9. invoke exact verification only after successful retrieval.
