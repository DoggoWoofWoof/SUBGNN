import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from retrieval_strategies import (  # noqa: E402
    fine_parent_ranking,
    hybrid_boundary_expand,
    multi_view_consensus_rankings,
    reciprocal_rank_fusion,
)


class TinyGraph:
    def __init__(self):
        self.edges = {}

    def add_weighted_edges_from(self, edges):
        for left, right, weight in edges:
            self.edges.setdefault(left, {})[right] = {"weight": weight}
            self.edges.setdefault(right, {})[left] = {"weight": weight}

    def has_node(self, node):
        return node in self.edges

    def neighbors(self, node):
        return self.edges.get(node, {})

    def __getitem__(self, node):
        return self.edges[node]


class RetrievalStrategiesTest(unittest.TestCase):
    def test_teleport_recovers_non_frontier_model_candidate(self):
        graph = TinyGraph()
        graph.add_weighted_edges_from([(0, 1, 10), (1, 2, 10), (8, 9, 10)])
        ranked = [0, 8, 1, 2, 9]

        selected = hybrid_boundary_expand(
            seed_ranked_ids=[0],
            score_ranked_ids=ranked,
            max_parts=3,
            coarse_part_graph=graph,
            seed_count=1,
            model_weight=0.5,
            teleport_every=2,
        )

        self.assertEqual(selected[0], 0)
        self.assertIn(8, selected)

    def test_global_fine_parents_can_change_fused_ranking(self):
        fine_parents = fine_parent_ranking([10, 11, 12], {10: 7, 11: 7, 12: 8})
        fused = reciprocal_rank_fusion([[0, 1, 2], fine_parents])

        self.assertEqual(fine_parents, [7, 8])
        self.assertIn(7, fused[:3])

    def test_consensus_rewards_agreement_across_retrievers(self):
        fused = reciprocal_rank_fusion([[1, 2, 3], [4, 2, 5]])

        self.assertEqual(fused[0], 2)

    def test_multiview_occurrence_rewards_repeated_view_support(self):
        rankings = multi_view_consensus_rankings(
            full_ranking=[1, 2, 3, 4, 5],
            view_rankings=[
                [4, 2, 1, 3, 5],
                [5, 2, 1, 3, 4],
                [3, 2, 1, 4, 5],
            ],
            support_depth=2,
        )

        self.assertEqual(rankings["multiview_occurrence"][0], 2)

    def test_multiview_specialist_preserves_strong_single_view_candidate(self):
        rankings = multi_view_consensus_rankings(
            full_ranking=[1, 2, 3, 4, 5, 6],
            view_rankings=[
                [6, 1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5, 6],
                [1, 2, 3, 4, 5, 6],
            ],
            support_depth=2,
        )

        self.assertLess(
            rankings["multiview_specialist"].index(6),
            [1, 2, 3, 4, 5, 6].index(6),
        )


if __name__ == "__main__":
    unittest.main()
