import os
import sys
import tempfile
import torch
from torch_geometric.data import Data

# Add /app to sys.path to import from src
sys.path.append("/app")

from src.glasgow_solver import glasgow_solve

def test_glasgow_integration():
    print("🚀 Starting Glasgow Integration Test (Final CSV 4-file format)")
    
    # Create a simple triangle query
    q_data = Data(
        x=torch.tensor([[1.0], [2.0], [3.0]]), # These will be hashed
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]]),
        global_id=torch.tensor([100, 200, 300])
    )
    q_data.num_nodes = 3
    
    # Create a square+diag target
    t_data = Data(
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        edge_index=torch.tensor([
            [0, 1, 1, 0, 0, 2, 2, 0, 1, 2, 2, 1, 0, 3, 3, 0, 2, 3, 3, 2],
            [1, 0, 0, 1, 2, 0, 0, 2, 2, 1, 1, 2, 3, 0, 0, 3, 3, 2, 2, 3]
        ]),
        global_id=torch.tensor([100, 200, 300, 400])
    )
    t_data.num_nodes = 4
    
    print("\n--- Running glasgow_solve (Integration) ---")
    result = glasgow_solve(
        q_data, t_data, 
        max_solutions=1, 
        timeout_seconds=30, 
        is_debug=True,
        target_name="sanity_check"
    )
    
    print("\n--- Integration Result ---")
    print(f"Status: {result['status']}")
    print(f"Mappings Found: {len(result['embeddings'])}")
    if result['embeddings']:
        print(f"First Mapping: {result['embeddings'][0]}")
        print(f"Best Accuracy: {result['best_acc']}")
        
    if result['status'] == 'true' and result['best_acc'] == 1.0:
        print("\n✅ INTEGRATION TEST PASSED!")
    else:
        print("\n❌ INTEGRATION TEST FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    test_glasgow_integration()
