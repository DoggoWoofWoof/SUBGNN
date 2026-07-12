"""Pure retrieval-selection strategies used by Jigsaw benchmarks."""

from collections import defaultdict
import math


def unique_ordered(values):
    seen = set()
    result = []
    for value in values:
        value = int(value)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def reciprocal_rank_fusion(rankings, rank_constant=20):
    """Fuse ranked ID lists while preserving evidence from every retriever."""
    scores = defaultdict(float)
    best_rank = {}
    for ranking in rankings:
        for rank, item_id in enumerate(unique_ordered(ranking)):
            scores[item_id] += 1.0 / (rank_constant + rank + 1)
            best_rank[item_id] = min(best_rank.get(item_id, rank), rank)
    return [
        item_id
        for item_id, _ in sorted(
            scores.items(),
            key=lambda item: (-item[1], best_rank[item[0]], item[0]),
        )
    ]


def fine_parent_ranking(ranked_fine_ids, fine_to_coarse):
    """Map global fine retrieval into a unique coarse-parent ranking."""
    return unique_ordered(
        fine_to_coarse[int(fine_id)]
        for fine_id in ranked_fine_ids
        if int(fine_id) in fine_to_coarse
    )


def multi_view_consensus_rankings(
    full_ranking,
    view_rankings,
    support_depth=20,
    rank_constant=20,
):
    """
    Fuse full-query and connected-view rankings without using retrieval truth.

    Returns complementary deterministic rankings:
    - RRF rewards broad agreement.
    - occurrence prioritizes candidates appearing in many view top lists.
    - fusion balances the full query, view consensus, and specialist evidence.
    - specialist gives more weight to a partition strongly supported by one view.
    """
    full_ranking = unique_ordered(full_ranking)
    view_rankings = [unique_ordered(ranking) for ranking in view_rankings]
    if not full_ranking:
        return {
            "multiview_rrf": [],
            "multiview_occurrence": [],
            "multiview_fusion": [],
            "multiview_specialist": [],
            "multiview_conservative": [],
            "multiview_full_max": [],
        }
    if not view_rankings:
        return {
            "multiview_rrf": full_ranking,
            "multiview_occurrence": full_ranking,
            "multiview_fusion": full_ranking,
            "multiview_specialist": full_ranking,
            "multiview_conservative": full_ranking,
            "multiview_full_max": full_ranking,
        }

    rank_constant = max(1, int(rank_constant))
    support_depth = max(1, int(support_depth))
    all_rankings = [full_ranking, *view_rankings]
    rank_maps = [
        {item_id: rank for rank, item_id in enumerate(ranking)}
        for ranking in all_rankings
    ]
    candidates = unique_ordered(
        item_id for ranking in all_rankings for item_id in ranking
    )

    def reciprocal_score(rank):
        return (rank_constant + 1.0) / (rank_constant + rank + 1.0)

    scores = {}
    for item_id in candidates:
        full_rank = rank_maps[0].get(item_id, len(full_ranking))
        view_ranks = [
            rank_map.get(item_id, len(ranking))
            for rank_map, ranking in zip(rank_maps[1:], view_rankings)
        ]
        view_scores = [reciprocal_score(rank) for rank in view_ranks]
        support = sum(rank < support_depth for rank in view_ranks) / len(view_ranks)
        full_score = reciprocal_score(full_rank)
        mean_score = sum(view_scores) / len(view_scores)
        max_score = max(view_scores)
        scores[item_id] = {
            "full": full_score,
            "support": support,
            "mean": mean_score,
            "max": max_score,
            "full_rank": full_rank,
        }

    def ranked_by(key):
        return sorted(
            candidates,
            key=lambda item_id: (
                -key(scores[item_id]),
                -scores[item_id]["support"],
                -scores[item_id]["max"],
                scores[item_id]["full_rank"],
                item_id,
            ),
        )

    occurrence = sorted(
        candidates,
        key=lambda item_id: (
            -scores[item_id]["support"],
            -scores[item_id]["mean"],
            -scores[item_id]["max"],
            scores[item_id]["full_rank"],
            item_id,
        ),
    )
    return {
        "multiview_rrf": reciprocal_rank_fusion(all_rankings, rank_constant),
        "multiview_occurrence": occurrence,
        "multiview_fusion": ranked_by(
            lambda score: (
                0.35 * score["full"]
                + 0.25 * score["support"]
                + 0.20 * score["mean"]
                + 0.20 * score["max"]
            )
        ),
        "multiview_specialist": ranked_by(
            lambda score: (
                0.25 * score["full"]
                + 0.15 * score["mean"]
                + 0.60 * score["max"]
            )
        ),
        "multiview_conservative": ranked_by(
            lambda score: (
                0.70 * score["full"]
                + 0.10 * score["support"]
                + 0.10 * score["mean"]
                + 0.10 * score["max"]
            )
        ),
        "multiview_full_max": ranked_by(
            lambda score: 0.75 * score["full"] + 0.25 * score["max"]
        ),
    }


def hybrid_boundary_expand(
    seed_ranked_ids,
    score_ranked_ids,
    max_parts,
    coarse_part_graph,
    seed_count=20,
    model_weight=0.5,
    teleport_every=10,
):
    """
    Expand a neural seed set using both model rank and graph-boundary support.

    Boundary-only expansion can become trapped around an imperfect top-20 seed.
    This policy scores frontier candidates with both signals and periodically
    teleports to the best remaining neural candidate to open a new frontier.
    """
    score_ranked_ids = unique_ordered(score_ranked_ids)
    seed_ranked_ids = unique_ordered(seed_ranked_ids)
    if max_parts <= 0:
        return []
    if coarse_part_graph is None:
        return score_ranked_ids[:max_parts]

    model_weight = min(1.0, max(0.0, float(model_weight)))
    teleport_every = max(0, int(teleport_every))
    rank_pos = {item_id: rank for rank, item_id in enumerate(score_ranked_ids)}
    rank_denominator = max(1, len(score_ranked_ids) - 1)
    selected = []
    selected_set = set()
    frontier_weight = defaultdict(float)
    frontier_support = defaultdict(int)

    def model_score(item_id):
        rank = rank_pos.get(int(item_id))
        if rank is None:
            return 0.0
        return 1.0 - (rank / rank_denominator)

    def add_frontier(item_id):
        if not coarse_part_graph.has_node(item_id):
            return
        for neighbor in coarse_part_graph.neighbors(item_id):
            neighbor = int(neighbor)
            if neighbor in selected_set:
                continue
            weight = float(coarse_part_graph[item_id][neighbor].get("weight", 1.0))
            frontier_weight[neighbor] += math.log1p(max(0.0, weight))
            frontier_support[neighbor] += 1

    def add(item_id):
        item_id = int(item_id)
        if item_id in selected_set:
            return False
        if not coarse_part_graph.has_node(item_id) and item_id not in rank_pos:
            return False
        selected.append(item_id)
        selected_set.add(item_id)
        frontier_weight.pop(item_id, None)
        frontier_support.pop(item_id, None)
        add_frontier(item_id)
        return True

    initial = seed_ranked_ids[: min(max_parts, max(1, int(seed_count)))]
    for item_id in initial:
        add(item_id)

    additions = 0
    while len(selected) < max_parts:
        additions += 1
        teleport = teleport_every > 0 and additions % teleport_every == 0
        best_model = next(
            (item_id for item_id in score_ranked_ids if item_id not in selected_set),
            None,
        )
        if teleport and best_model is not None:
            add(best_model)
            continue

        candidates = [item_id for item_id in frontier_weight if item_id not in selected_set]
        if not candidates:
            if best_model is None:
                break
            add(best_model)
            continue

        max_boundary = max(frontier_weight[item_id] for item_id in candidates)
        max_support = max(frontier_support[item_id] for item_id in candidates)

        def combined_score(item_id):
            boundary = frontier_weight[item_id] / max(1e-12, max_boundary)
            support = frontier_support[item_id] / max(1, max_support)
            boundary_score = 0.8 * boundary + 0.2 * support
            return model_weight * model_score(item_id) + (1.0 - model_weight) * boundary_score

        best = min(
            candidates,
            key=lambda item_id: (
                -combined_score(item_id),
                rank_pos.get(item_id, len(score_ranked_ids) + 1),
                item_id,
            ),
        )
        add(best)

    return selected


def ranked_neighbor_stitch(
    score_ranked_ids,
    max_parts,
    coarse_part_graph,
    seed_count=10,
    pool_k=100,
    seed_ranked_ids=None,
    feature_ranked_ids=None,
    model_weight=0.55,
    feature_weight=0.15,
    restart_when_stuck=True,
):
    """
    Rank-aware neighbor stitching for fixed-budget retrieval.

    The selector starts from the top ``seed_count`` model-ranked partitions and
    then admits only top-``pool_k`` partitions that are neighbors of already
    selected seeds/components. When the frontier is exhausted, it can restart
    from the next high-ranked top-pool partition, creating a new component.
    Optional feature ranks act as a cheap secondary signal without using labels
    or retrieval truth.
    """
    score_ranked_ids = unique_ordered(score_ranked_ids)
    seed_ranked_ids = unique_ordered(seed_ranked_ids or score_ranked_ids)
    feature_ranked_ids = unique_ordered(feature_ranked_ids or [])
    if max_parts <= 0:
        return []
    if not score_ranked_ids:
        return []
    if coarse_part_graph is None:
        return score_ranked_ids[:max_parts]

    seed_count = max(1, int(seed_count))
    pool_k = max(seed_count, int(pool_k))
    model_weight = min(1.0, max(0.0, float(model_weight)))
    feature_weight = min(1.0, max(0.0, float(feature_weight)))
    boundary_weight = max(0.0, 1.0 - model_weight - feature_weight)

    pool = set(score_ranked_ids[: min(pool_k, len(score_ranked_ids))])
    rank_pos = {item_id: rank for rank, item_id in enumerate(score_ranked_ids)}
    feature_pos = {item_id: rank for rank, item_id in enumerate(feature_ranked_ids)}
    rank_denominator = max(1, min(pool_k, len(score_ranked_ids)) - 1)
    feature_denominator = max(1, len(feature_ranked_ids) - 1)

    selected = []
    selected_set = set()
    frontier_weight = defaultdict(float)
    frontier_support = defaultdict(int)

    def model_score(item_id):
        rank = rank_pos.get(int(item_id))
        if rank is None:
            return 0.0
        return max(0.0, 1.0 - (rank / rank_denominator))

    def feature_score(item_id):
        rank = feature_pos.get(int(item_id))
        if rank is None:
            return 0.0
        return max(0.0, 1.0 - (rank / feature_denominator))

    def add_frontier(item_id):
        if not coarse_part_graph.has_node(item_id):
            return
        for neighbor in coarse_part_graph.neighbors(item_id):
            neighbor = int(neighbor)
            if neighbor in selected_set or neighbor not in pool:
                continue
            edge_weight = float(coarse_part_graph[item_id][neighbor].get("weight", 1.0))
            frontier_weight[neighbor] += math.log1p(max(0.0, edge_weight))
            frontier_support[neighbor] += 1

    def add(item_id):
        item_id = int(item_id)
        if item_id in selected_set:
            return False
        if item_id not in pool and len(selected) < pool_k:
            return False
        selected.append(item_id)
        selected_set.add(item_id)
        frontier_weight.pop(item_id, None)
        frontier_support.pop(item_id, None)
        add_frontier(item_id)
        return True

    for item_id in seed_ranked_ids[:seed_count]:
        if len(selected) >= max_parts:
            break
        add(item_id)

    def next_restart():
        return next(
            (
                item_id
                for item_id in score_ranked_ids[:pool_k]
                if item_id not in selected_set
            ),
            None,
        )

    while len(selected) < max_parts:
        candidates = [
            item_id
            for item_id in frontier_weight
            if item_id not in selected_set and item_id in pool
        ]
        if not candidates:
            if not restart_when_stuck:
                break
            restart = next_restart()
            if restart is None:
                break
            add(restart)
            continue

        max_boundary = max(frontier_weight[item_id] for item_id in candidates)
        max_support = max(frontier_support[item_id] for item_id in candidates)

        def combined_score(item_id):
            boundary = frontier_weight[item_id] / max(1e-12, max_boundary)
            support = frontier_support[item_id] / max(1, max_support)
            boundary_score = 0.75 * boundary + 0.25 * support
            return (
                model_weight * model_score(item_id)
                + feature_weight * feature_score(item_id)
                + boundary_weight * boundary_score
            )

        best = min(
            candidates,
            key=lambda item_id: (
                -combined_score(item_id),
                rank_pos.get(item_id, len(score_ranked_ids) + 1),
                feature_pos.get(item_id, len(feature_ranked_ids) + 1),
                item_id,
            ),
        )
        add(best)

    return selected


def prefix_preserving_rerank(
    base_ranked_ids,
    budget,
    coarse_part_graph=None,
    feature_ranked_ids=None,
    seed_count=5,
    model_weight=0.50,
    feature_weight=0.25,
):
    """
    Reorder the original top-budget bucket to improve a high-precision prefix.

    Unlike expansion/stitching, this never imports candidates from outside the
    original top-``budget`` set. Therefore FullCov@budget is identical to the
    base ranking while P@1/2/5/10 can change.
    """
    base_ranked_ids = unique_ordered(base_ranked_ids)
    feature_ranked_ids = unique_ordered(feature_ranked_ids or [])
    if budget <= 0 or not base_ranked_ids:
        return []

    bucket = base_ranked_ids[: min(int(budget), len(base_ranked_ids))]
    bucket_set = set(bucket)
    rank_pos = {item_id: rank for rank, item_id in enumerate(base_ranked_ids)}
    feature_pos = {item_id: rank for rank, item_id in enumerate(feature_ranked_ids)}
    rank_denominator = max(1, len(bucket) - 1)
    feature_denominator = max(1, len(feature_ranked_ids) - 1)
    seed_set = set(base_ranked_ids[: max(1, int(seed_count))])
    model_weight = min(1.0, max(0.0, float(model_weight)))
    feature_weight = min(1.0, max(0.0, float(feature_weight)))
    boundary_weight = max(0.0, 1.0 - model_weight - feature_weight)

    boundary_scores = defaultdict(float)
    if coarse_part_graph is not None:
        for item_id in bucket:
            item_id = int(item_id)
            if not coarse_part_graph.has_node(item_id):
                continue
            for neighbor in coarse_part_graph.neighbors(item_id):
                neighbor = int(neighbor)
                if neighbor not in bucket_set:
                    continue
                weight = float(coarse_part_graph[item_id][neighbor].get("weight", 1.0))
                edge_score = math.log1p(max(0.0, weight))
                if neighbor in seed_set:
                    boundary_scores[item_id] += 1.5 * edge_score
                else:
                    boundary_scores[item_id] += edge_score

    max_boundary = max(boundary_scores.values(), default=1.0)

    def model_score(item_id):
        rank = rank_pos.get(int(item_id), len(base_ranked_ids))
        return max(0.0, 1.0 - (rank / rank_denominator))

    def feature_score(item_id):
        rank = feature_pos.get(int(item_id))
        if rank is None:
            return 0.0
        return max(0.0, 1.0 - (rank / feature_denominator))

    def combined_score(item_id):
        boundary = boundary_scores[int(item_id)] / max(1e-12, max_boundary)
        return (
            model_weight * model_score(item_id)
            + feature_weight * feature_score(item_id)
            + boundary_weight * boundary
        )

    return sorted(
        bucket,
        key=lambda item_id: (
            -combined_score(item_id),
            rank_pos.get(item_id, len(base_ranked_ids) + 1),
            feature_pos.get(item_id, len(feature_ranked_ids) + 1),
            item_id,
        ),
    )
