# SUBGNN Evaluation Framework

A comprehensive framework for evaluating Subgraph GNNs using stratified subgraph sampling and VF2/Glasgow subgraph matching.

## Features

### 1. Hierarchical Graph Partitioning
- Decomposes large graphs (Cora, Arxiv, MAG) into coarse and fine partitions.
- Enables efficient localized search.

### 2. Stratified Query Generation (4 Types)
- **K-Hop**: Traditional k-hop subgraphs from random anchors.
- **Single Partition**: Queries strictly contained within one fine partition.
- **Sibling Walk**: Queries spanning multiple fine partitions within the same coarse region.
- **Multi-Coarse**: Complex queries spanning multiple coarse partitions (stitched).

### 3. Advanced Search & Verification
- **FAISS Retrieval**: Embedding-based coarse partition retrieval.
- **Node Stitching**: Dynamically stitches top-K coarse partitions and their neighbors.
- **VF2 Matching**: NetworkX-based subgraph isomorphism verification.
- **Glasgow Solver**: Optional support for the high-performance [Glasgow Subgraph Solver](https://github.com/ciaranm/glasgow-subgraph-solver).

## Folder Structure

- `src/`: Core implementation.
  - `evaluate.py`: Main evaluation script.
  - `query_generator.py`: Stratified query generation logic.
  - `vf2_matcher.py`: Python-based VF2 matcher.
  - `glasgow_solver.py`: Wrapper for external Glasgow binary.
  - `model.py`, `data.py`: GNN model and data loading.
- `models/`: Pre-trained model checkpoints (`.pth`).
- `scripts/`: Training scripts and notebooks.

## Installation

1. Install dependencies:
```bash
pip install torch torch_geometric ogb networkx faiss-cpu pandas matplotlib
```

2. (Optional) Install Glasgow Subgraph Solver:
   - Download `glasgow_subgraph_solver` binary to a known path.
   - Update `src/config.py` with the path to the binary.

## Usage

Run evaluation on supported datasets (cora, arxiv, mag):

```bash
# Evaluate on Cora
python -m src.evaluate --dataset cora --model_path models/cora-6_layer-model-jigsaw.pth --queries_per_partition 2 --output cora_results.csv

# Evaluate on Arxiv
python -m src.evaluate --dataset arxiv --model_path models/arxiv-6_layer-model-jigsaw.pth --queries_per_partition 2 --output arxiv_results.csv
```

## Metrics
The framework reports:
- **Retrieval**: Coarse Recall@1, Recall@K.
- **Verification**: VF2 match rate (Stitched vs Baseline).
- **Partitioning**: Ground Truth Partition Recall.
- **Detailed Timing**: Latency breakdown for embedding, FAISS, and VF2.
