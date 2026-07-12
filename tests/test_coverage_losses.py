import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

try:
    import torch
    from coverage_losses import partition_coverage_loss  # noqa: E402
except ImportError:
    torch = None
    partition_coverage_loss = None


@unittest.skipIf(torch is None, "torch is not installed in the local Modal control venv")
class CoverageLossTest(unittest.TestCase):
    def test_cvar_emphasizes_worst_required_positive(self):
        query = torch.tensor([[1.0, 0.0]])
        partitions = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.0],
                [0.7, 0.0],
            ]
        )
        mean_loss = partition_coverage_loss(
            query,
            partitions,
            [[0, 1]],
            positive_aggregation="mean",
        )
        cvar_loss = partition_coverage_loss(
            query,
            partitions,
            [[0, 1]],
            positive_aggregation="cvar",
            cvar_fraction=0.5,
        )
        self.assertGreater(cvar_loss.item(), mean_loss.item())

    def test_live_positive_embedding_receives_gradient(self):
        query = torch.tensor([[1.0, 0.0]], requires_grad=True)
        partitions = torch.tensor([[0.0, 1.0], [0.5, 0.0], [0.4, 0.0]])
        live = torch.tensor([[1.0, 0.0]], requires_grad=True)
        loss = partition_coverage_loss(
            query,
            partitions,
            [[0]],
            live_partition_indices=torch.tensor([0]),
            live_partition_embeddings=live,
        )
        loss.backward()
        self.assertIsNotNone(live.grad)
        self.assertGreater(torch.linalg.vector_norm(live.grad).item(), 0.0)

    def test_topk_bucket_size_scales_broad_queries(self):
        query = torch.tensor([[1.0, 0.0]])
        partitions = torch.tensor(
            [[1.0, 0.0], [0.9, 0.0], [0.8, 0.0], [0.7, 0.0], [0.6, 0.0]]
        )
        small_bucket_loss = partition_coverage_loss(
            query,
            partitions,
            [[0, 1, 2]],
            target_topk=2,
            topk_bucket_size=1,
            topk_weight=1.0,
        )
        broad_bucket_loss = partition_coverage_loss(
            query,
            partitions,
            [[0, 1, 2]],
            target_topk=2,
            topk_bucket_size=10,
            topk_weight=1.0,
        )
        self.assertGreater(small_bucket_loss.item(), broad_bucket_loss.item())


if __name__ == "__main__":
    unittest.main()
