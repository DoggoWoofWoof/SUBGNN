import torch
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Paths (Windows-compatible, relative to project root)
DATA_ROOT_CORA = "data/Cora"
DATA_ROOT_ARXIV = "data/ogbn_arxiv"
DATA_ROOT_MAG = "data/ogbn_mag"

# Checkpoint Paths (Updated to match actual model locations)
CHECKPOINT_CORA = "models/cora-6_layer-model-jigsaw.pth"
CHECKPOINT_ARXIV = "models/arxiv-6_layer-model-jigsaw.pth"
CHECKPOINT_MAG = "models/mag-6_layer-model-jigsaw.pth"

# Model Hyperparameters
GIN_HIDDEN_NEURONS = 256
GIN_OUTPUT_NEURONS = 128
DROPOUT = 0.1

# NodeFeatureAugmentor Dims (MAG only)
TYPE_DIM = 16
NODE_DIM = 16

# Partitioning Targets (Dataset Specific)
PARTITION_CONFIGS = {
    'cora': {'coarse': 20, 'fine': 10},
    'arxiv': {'coarse': 170, 'fine': 10},
    'mag': {'coarse': 1997, 'fine': 10},
    'default': {'coarse': 100, 'fine': 10}
}

# Evaluation configs
FAISS_TOP_K = 5
SOLVER_PATH = "v1/build/glasgow_subgraph_solver.exe"

# Query Params
QUERY_NODES_MIN = 20
QUERY_NODES_MAX = 100
