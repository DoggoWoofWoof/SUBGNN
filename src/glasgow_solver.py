"""
Glasgow Subgraph Solver - External Binary Wrapper

This module wraps the Glasgow subgraph isomorphism solver for high-performance
subgraph matching. Falls back to VF2 if Glasgow is not available.

Installation:
    See: glasgow_solver_installation.md in artifacts
    
Usage:
    solver = GlasgowSolver(solver_path="path/to/glasgow_subgraph_solver.exe")
    result = solver.find_subgraph(query_graph, target_graph, ...)
"""

import os
import re
import subprocess
import time
import random
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data


@dataclass
class GlasgowResult:
    """Result from Glasgow solver."""
    found: bool
    num_solutions: int
    first_solution_accuracy: float  # Accuracy of first found solution
    time_to_first_solution: float   # Time to find ANY solution
    time_to_correct_solution: float # Time to find PERFECT solution
    time_to_all_solutions: float    # Total time to enumerate all
    mapping: Optional[Dict[int, int]] = None  # query_local -> target_local
    latency_seconds: float = 0.0


def feature_to_label(vector) -> str:
    """
    Convert feature vector to a hash label for Glasgow solver.
    Uses SHA256 for deterministic, collision-resistant mapping.
    """
    import hashlib
    import numpy as np
    
    if isinstance(vector, torch.Tensor):
        vector = vector.cpu().numpy()
    
    # Create deterministic hash from feature vector
    vector_bytes = vector.astype(np.float32).tobytes()
    return hashlib.sha256(vector_bytes).hexdigest()[:16]  # First 16 chars


def convert_graph_to_csv(
    data: Data,
    filename: str,
    global_id_to_label: Dict[int, str],
    local_to_global_map: Dict[int, int],
):
    """
    Convert PyTorch Geometric graph to CSV format for Glasgow solver.
    
    Format:
        Row per edge: source_label, target_label
    """
    import pandas as pd
    
    edge_list = []
    edge_index = data.edge_index
    
    for i in range(edge_index.shape[1]):
        src_local = edge_index[0, i].item()
        dst_local = edge_index[1, i].item()
        
        src_global = local_to_global_map.get(src_local, src_local)
        dst_global = local_to_global_map.get(dst_local, dst_local)
        
        src_label = global_id_to_label.get(src_global, f"node_{src_global}")
        dst_label = global_id_to_label.get(dst_global, f"node_{dst_global}")
        
        edge_list.append([src_label, dst_label])
    
    df = pd.DataFrame(edge_list, columns=["source", "target"])
    df.to_csv(filename, index=False)


class GlasgowSolver:
    """
    Wrapper for Glasgow Subgraph Isomorphism Solver.
    
    The Glasgow solver is a state-of-the-art external binary that performs
    subgraph isomorphism search with various optimizations.
    """
    
    def __init__(self, solver_path: str = None):
        """
        Initialize Glasgow solver.
        
        Args:
            solver_path: Path to glasgow_subgraph_solver.exe
                        If None, uses config.SOLVER_PATH
        """
        if solver_path is None:
            from src.config import SOLVER_PATH
            solver_path = SOLVER_PATH
        
        self.solver_path = solver_path
        self.available = os.path.exists(solver_path)
        
        if not self.available:
            print(f"[WARN] Glasgow solver not found at {solver_path}")
            print("[WARN] Will fall back to VF2 if called")
    
    def find_subgraph(
        self,
        query_data: Data,
        target_data: Data,
        query_global_ids: torch.Tensor,
        target_global_ids: torch.Tensor,
        global_id_to_label: Optional[Dict[int, str]] = None,
        timeout_seconds: float = 60.0,
        induced: bool = True,
        count_solutions: bool = True,
    ) -> GlasgowResult:
        """
        Find subgraph using Glasgow solver.
        
        Args:
            query_data: PyG graph to find
            target_data: PyG graph to search in
            query_global_ids: Global node IDs for query
            target_global_ids: Global node IDs for target
            global_id_to_label: Feature hash map (computed if None)
            timeout_seconds: Max time to run
            induced: If True, find induced subgraph
            count_solutions: If True, enumerate all solutions
            
        Returns:
            GlasgowResult with timing and accuracy metrics
        """
        if not self.available:
            # Fall back to VF2
            from src.vf2_matcher import vf2_verify_subgraph
            vf2_result = vf2_verify_subgraph(
                query_data, target_data,
                query_global_ids, target_global_ids,
                timeout_seconds=timeout_seconds
            )
            return GlasgowResult(
                found=vf2_result.found,
                num_solutions=vf2_result.num_solutions,
                first_solution_accuracy=vf2_result.node_accuracy if vf2_result.found else -1,
                time_to_first_solution=vf2_result.latency_seconds if vf2_result.found else -1,
                time_to_correct_solution=vf2_result.latency_seconds if vf2_result.found else -1,
                time_to_all_solutions=vf2_result.latency_seconds,
                latency_seconds=vf2_result.latency_seconds
            )
        
        # Build label map if not provided
        if global_id_to_label is None:
            global_id_to_label = {}
            # Hash query features
            if query_data.x is not None:
                for i, gid in enumerate(query_global_ids.tolist()):
                    global_id_to_label[gid] = feature_to_label(query_data.x[i])
            # Hash target features
            if target_data.x is not None:
                for i, gid in enumerate(target_global_ids.tolist()):
                    if gid not in global_id_to_label:
                        global_id_to_label[gid] = feature_to_label(target_data.x[i])
        
        # Create temp files for CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as qf:
            query_csv = qf.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tf:
            target_csv = tf.name
        
        try:
            # Build local-to-global maps
            q_local_to_global = {i: gid.item() if isinstance(gid, torch.Tensor) else gid 
                                for i, gid in enumerate(query_global_ids.tolist())}
            t_local_to_global = {i: gid.item() if isinstance(gid, torch.Tensor) else gid 
                                for i, gid in enumerate(target_global_ids.tolist())}
            
            # Convert graphs to CSV
            convert_graph_to_csv(query_data, query_csv, global_id_to_label, q_local_to_global)
            convert_graph_to_csv(target_data, target_csv, global_id_to_label, t_local_to_global)
            
            # Build command
            cmd = [self.solver_path]
            if count_solutions:
                cmd.extend(["--count-solutions", "--print-all-solutions"])
            if induced:
                cmd.append("--induced")
            cmd.extend([query_csv, target_csv])
            
            # Run solver
            return self._run_solver(
                cmd, q_local_to_global, t_local_to_global, timeout_seconds
            )
            
        finally:
            # Cleanup temp files
            try:
                os.unlink(query_csv)
                os.unlink(target_csv)
            except:
                pass
    
    def _run_solver(
        self,
        cmd: List[str],
        query_local_to_global: Dict[int, int],
        target_local_to_global: Dict[int, int],
        timeout_seconds: float
    ) -> GlasgowResult:
        """Run Glasgow solver and parse output."""
        
        mapping_pattern = re.compile(r"^mapping\s*=\s*(.*)$")
        solution_count_pattern = re.compile(r"^solution_count\s*=\s*(\d+)$")
        
        solution_found = False
        solution_count = 0
        first_accuracy = -1
        time_to_first = -1
        time_to_correct = -1
        time_to_all = -1
        best_mapping = None
        
        start_time = time.time()
        
        try:
            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            ) as proc:
                for line in proc.stdout:
                    # Check timeout
                    if time.time() - start_time > timeout_seconds:
                        proc.kill()
                        break
                    
                    line = line.strip()
                    
                    # Parse mapping
                    if mapping_match := mapping_pattern.match(line):
                        now = time.time()
                        if time_to_first < 0:
                            time_to_first = now - start_time
                        
                        pairs = re.findall(r"\((\d+)\s*->\s*(\d+)\)", mapping_match.group(1))
                        mapping = {int(q): int(t) for q, t in pairs}
                        
                        # Calculate accuracy
                        correct = 0
                        total = len(mapping)
                        for q_local, t_local in mapping.items():
                            q_gid = query_local_to_global.get(q_local)
                            t_gid = target_local_to_global.get(t_local)
                            if q_gid == t_gid:
                                correct += 1
                        
                        accuracy = (correct / total * 100) if total > 0 else 0
                        
                        if first_accuracy < 0:
                            first_accuracy = accuracy
                        
                        if accuracy == 100.0 and not solution_found:
                            time_to_correct = now - start_time
                            solution_found = True
                            best_mapping = mapping
                    
                    # Parse solution count
                    elif count_match := solution_count_pattern.match(line):
                        solution_count = int(count_match.group(1))
                        time_to_all = time.time() - start_time
                        
        except FileNotFoundError:
            print(f"[ERROR] Glasgow solver not found: {cmd[0]}")
            return GlasgowResult(
                found=False, num_solutions=0,
                first_solution_accuracy=-1,
                time_to_first_solution=-1,
                time_to_correct_solution=-1,
                time_to_all_solutions=-1,
                latency_seconds=time.time() - start_time
            )
        
        return GlasgowResult(
            found=solution_found,
            num_solutions=solution_count,
            first_solution_accuracy=first_accuracy,
            time_to_first_solution=time_to_first,
            time_to_correct_solution=time_to_correct,
            time_to_all_solutions=time_to_all,
            mapping=best_mapping,
            latency_seconds=time.time() - start_time
        )


# Singleton instance
_solver = None

def get_glasgow_solver(solver_path: str = None) -> GlasgowSolver:
    """Get or create Glasgow solver instance."""
    global _solver
    if _solver is None or (solver_path and solver_path != _solver.solver_path):
        _solver = GlasgowSolver(solver_path)
    return _solver
