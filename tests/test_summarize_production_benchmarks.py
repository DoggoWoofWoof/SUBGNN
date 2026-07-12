import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SummarizeProductionBenchmarksTest(unittest.TestCase):
    def test_glob_skips_partial_per_query_csvs_by_default(self):
        repo = Path(__file__).resolve().parents[1]
        script = repo / "scripts" / "summarize_production_benchmarks.py"
        fieldnames = [
            "dataset",
            "seed",
            "model",
            "query_id",
            "query_type",
            "target_query_size",
            "method",
            "signature",
            "budget",
            "cascade_first_solved",
            "expected_match",
            "pruned_node_fullcov",
            "solver_timed_out",
            "cascade_total_candidate_time_seconds",
            "cascade_total_solver_time_seconds",
            "pruned_candidate_nodes",
            "component_solver_nodes",
            "node_reduction_factor",
            "retrieval_time_seconds",
            "candidate_time_seconds",
            "solver_time_seconds",
            "true_coarse_count",
            "coarse_precision_at_budget",
            "coarse_recall_at_budget",
            "max_true_coarse_rank",
            "edge_reduction_factor",
        ]

        row = {
            "dataset": "arxiv",
            "seed": "20260607",
            "model": "arxiv",
            "query_id": "q_final",
            "query_type": "k_hop",
            "target_query_size": "20",
            "method": "neural_component",
            "signature": "type_feat32",
            "budget": "20",
            "cascade_first_solved": "true",
            "expected_match": "true",
            "pruned_node_fullcov": "true",
            "solver_timed_out": "false",
            "cascade_total_candidate_time_seconds": "0.1",
            "cascade_total_solver_time_seconds": "0.2",
            "pruned_candidate_nodes": "5",
            "component_solver_nodes": "6",
            "node_reduction_factor": "2",
            "retrieval_time_seconds": "0.01",
            "candidate_time_seconds": "0.02",
            "solver_time_seconds": "0.03",
            "true_coarse_count": "1",
            "coarse_precision_at_budget": "1",
            "coarse_recall_at_budget": "1",
            "max_true_coarse_rank": "1",
            "edge_reduction_factor": "2",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            final_csv = tmp / "prod_arxiv_sizes20_neural_component_b20_per_query.csv"
            partial_csv = tmp / "prod_arxiv_sizes20_neural_component_b20_partial_per_query.csv"
            for path, query_id in ((final_csv, "q_final"), (partial_csv, "q_partial")):
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow({**row, "query_id": query_id})

            output = tmp / "summary.csv"
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    str(tmp / "*_per_query.csv"),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("[SKIP PARTIAL]", result.stderr)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["queries"], "1")
            self.assertEqual(rows[0]["file"], str(final_csv))

    def test_budget_columns_are_explicit_and_cumulative(self):
        repo = Path(__file__).resolve().parents[1]
        script = repo / "scripts" / "summarize_production_benchmarks.py"
        fieldnames = [
            "dataset",
            "seed",
            "model",
            "query_id",
            "query_type",
            "target_query_size",
            "method",
            "signature",
            "budget",
            "cascade_first_solved",
            "expected_match",
            "pruned_node_fullcov",
            "solver_timed_out",
            "cascade_total_candidate_time_seconds",
            "cascade_total_solver_time_seconds",
            "pruned_candidate_nodes",
            "component_solver_nodes",
            "node_reduction_factor",
            "retrieval_time_seconds",
            "candidate_time_seconds",
            "solver_time_seconds",
            "true_coarse_count",
            "coarse_precision_at_budget",
            "coarse_recall_at_budget",
            "max_true_coarse_rank",
            "edge_reduction_factor",
        ]

        base = {
            "dataset": "mag",
            "seed": "20260607",
            "model": "mag_rgcn_best",
            "query_type": "k_hop",
            "target_query_size": "50",
            "method": "neural_component",
            "signature": "type_rel_feat32",
            "expected_match": "true",
            "pruned_node_fullcov": "true",
            "solver_timed_out": "false",
            "cascade_total_candidate_time_seconds": "0.1",
            "cascade_total_solver_time_seconds": "0.2",
            "pruned_candidate_nodes": "5",
            "component_solver_nodes": "6",
            "node_reduction_factor": "2",
            "retrieval_time_seconds": "0.01",
            "candidate_time_seconds": "0.02",
            "solver_time_seconds": "0.03",
            "true_coarse_count": "1",
            "coarse_precision_at_budget": "1",
            "coarse_recall_at_budget": "1",
            "max_true_coarse_rank": "1",
            "edge_reduction_factor": "2",
        }

        rows = [
            {**base, "query_id": "q_a", "budget": "100", "cascade_first_solved": "true"},
            {**base, "query_id": "q_a", "budget": "20", "cascade_first_solved": "true"},
            {**base, "query_id": "q_b", "budget": "20", "cascade_first_solved": "false"},
            {**base, "query_id": "q_b", "budget": "50", "cascade_first_solved": "true"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "prod_mag_sizes50_neural_component_b100_per_query.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            output = tmp / "summary.csv"
            subprocess.run(
                [sys.executable, str(script), str(csv_path), "--output", str(output)],
                check=True,
                capture_output=True,
                text=True,
            )

            with output.open(newline="", encoding="utf-8") as handle:
                [row] = list(csv.DictReader(handle))
            self.assertEqual(row["first_solved_at_20"], "1")
            self.assertEqual(row["first_solved_at_50"], "1")
            self.assertEqual(row["first_solved_at_100"], "0")
            self.assertEqual(row["solved_by_20"], "1")
            self.assertEqual(row["solved_by_50"], "2")
            self.assertEqual(row["solved_by_100"], "2")


if __name__ == "__main__":
    unittest.main()
