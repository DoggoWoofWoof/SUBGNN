import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark_overlap_glasgow_cascade import (  # noqa: E402
    derive_query_labels,
    derive_query_signature_tokens,
    prune_nodes_by_signature,
)
from benchmark_retrieval import _build_node_signature_tokens  # noqa: E402


def test_feature_labels_ignore_global_ids():
    query = Data(
        x=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
    )
    query.global_id = torch.tensor([100, 200])
    expected = derive_query_labels(query, "feature")
    query.global_id = torch.tensor([9_999, 8_888])
    assert derive_query_labels(query, "feature") == expected


def test_signature_pruning_uses_query_payload_not_planted_ids():
    target = Data(
        x=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
    )
    query = Data(
        x=target.x[torch.tensor([0, 1])].clone(),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
    )
    target_tokens = _build_node_signature_tokens(target)["type_feat32"]
    expected_query_tokens = derive_query_signature_tokens(query, "type_feat32")
    expected = {
        node
        for node in range(target.num_nodes)
        if int(target_tokens[node]) in set(expected_query_tokens.tolist())
    }

    query.global_id = torch.tensor([0, 1])
    first = prune_nodes_by_signature(
        set(range(target.num_nodes)), query, target_tokens, "type_feat32"
    )
    query.global_id = torch.tensor([3, 2])
    second = prune_nodes_by_signature(
        set(range(target.num_nodes)), query, target_tokens, "type_feat32"
    )

    assert first == expected
    assert second == expected


def test_legacy_mag_query_recovers_type_from_feature_payload():
    type_block = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    query = Data(
        x=torch.cat([torch.zeros((2, 128)), type_block, torch.zeros((2, 2))], dim=1),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
    )
    query.feature_schema = "mag_type_rel_v1"
    query.node_types = ["paper", "author"]
    tokens = derive_query_signature_tokens(query, "type")
    assert set(tokens.tolist()) == {0, 1}
