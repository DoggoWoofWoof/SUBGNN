import sys
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark_glasgow as bench  # noqa: E402
import benchmark_overlap_glasgow_cascade as cascade  # noqa: E402


def make_query(edge_pairs, num_nodes=4):
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    return Data(x=torch.ones(num_nodes, 1), edge_index=edge_index, num_nodes=num_nodes)


class QueryCacheValidationTests(unittest.TestCase):
    def test_subgraph_connectivity_detects_disconnected_queries(self):
        connected = make_query([(0, 1), (1, 2), (2, 3), (1, 0), (2, 1), (3, 2)])
        disconnected = make_query([(0, 1), (1, 0), (2, 3), (3, 2)])

        self.assertTrue(bench._subgraph_is_connected(connected))
        self.assertFalse(bench._subgraph_is_connected(disconnected))

    def test_cached_multi_coarse_queries_must_be_connected(self):
        disconnected = make_query([(0, 1), (1, 0), (2, 3), (3, 2)])
        queries = [
            {
                "query_type": "multi_coarse",
                "target_query_size": 20,
                "query": disconnected,
            }
        ]

        self.assertFalse(cascade._queries_match_spec(queries, count=1, sizes=[20], query_types="multi_coarse"))


if __name__ == "__main__":
    unittest.main()
