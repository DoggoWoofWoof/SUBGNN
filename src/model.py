import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, LeakyReLU, Linear, ReLU, Sequential, LayerNorm
from torch_geometric.nn import GINConv, global_mean_pool, GATConv, global_max_pool, global_add_pool

class NodeFeatureAugmentor(nn.Module):
    def __init__(self, num_nodes: int, num_types: int, type_dim: int = 16, node_dim: int = 0):
        super().__init__()
        self.type_emb = nn.Embedding(num_types, type_dim)
        self.node_dim = node_dim
        self.node_emb = nn.Embedding(num_nodes, node_dim) if node_dim > 0 else None

    @property
    def added_dim(self) -> int:
        return self.type_emb.embedding_dim + (self.node_emb.embedding_dim if self.node_emb is not None else 0)

    def forward(self, data) -> torch.Tensor:
        # Check attributes existence to avoid errors
        if not hasattr(data, 'node_type') or data.node_type is None:
             pass
        
        pieces = [data.x, self.type_emb(data.node_type)]
        if self.node_emb is not None:
            gid = data.global_id if hasattr(data, "global_id") else torch.arange(data.num_nodes, device=data.x.device)
            pieces.append(self.node_emb(gid))
        return torch.cat(pieces, dim=1)

class ImprovedSubgraphEncoder(torch.nn.Module):
    def __init__(self, in_neurons, hidden_neurons, output_neurons, dropout=0.1, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.dropout = dropout

        # Strict adherence to mag_model architecture
        nn1 = Sequential(Linear(in_neurons, hidden_neurons), ReLU(), Dropout(dropout), Linear(hidden_neurons, hidden_neurons))
        self.conv1 = GINConv(nn1)
        nn2 = Sequential(Linear(hidden_neurons, hidden_neurons), ReLU(), Dropout(dropout), Linear(hidden_neurons, hidden_neurons))
        self.conv2 = GINConv(nn2)
        nn3 = Sequential(Linear(hidden_neurons, hidden_neurons), ReLU(), Dropout(dropout), Linear(hidden_neurons, hidden_neurons))
        self.conv3 = GINConv(nn3)
        nn4 = Sequential(Linear(hidden_neurons, hidden_neurons), ReLU(), Dropout(dropout), Linear(hidden_neurons, hidden_neurons))
        self.conv4 = GINConv(nn4)
        nn5 = Sequential(Linear(hidden_neurons, hidden_neurons), ReLU(), Dropout(dropout), Linear(hidden_neurons, hidden_neurons))
        self.conv5 = GINConv(nn5)
        nn6 = Sequential(Linear(hidden_neurons, hidden_neurons), ReLU(), Dropout(dropout), Linear(hidden_neurons, hidden_neurons))
        self.conv6 = GINConv(nn6)

        self.ln1 = LayerNorm(hidden_neurons); self.ln2 = LayerNorm(hidden_neurons); self.ln3 = LayerNorm(hidden_neurons)
        self.ln4 = LayerNorm(hidden_neurons); self.ln5 = LayerNorm(hidden_neurons); self.ln6 = LayerNorm(hidden_neurons)
        
        # Original MAG logic: Project input only if dimensions differ
        self.input_proj = Linear(in_neurons, hidden_neurons) if in_neurons != hidden_neurons else None
        
        self.use_multi_pool = True; readout_dim = hidden_neurons * 6 * 3

        self.readout_proj = Sequential(
            Linear(readout_dim, hidden_neurons * 2), ReLU(), Dropout(dropout),
            Linear(hidden_neurons * 2, hidden_neurons), ReLU(), Dropout(dropout),
            Linear(hidden_neurons, output_neurons)
        )
        self.readout_skip = Linear(readout_dim, output_neurons)

    def forward(self, x, edge_index, batch):
        layer_outputs = []
        
        # Apply input projection if it exists (handling high-dim inputs like Cora)
        x_res = self.input_proj(x) if self.input_proj is not None else x
        
        # IMPORTANT: conv1 in GIN takes 'x'. If in_neurons != hidden_neurons, the nn1 (Linear) inside conv1
        # expects in_neurons. So we pass 'x' (original dim) to conv1.
        # But for the residual connection, we need 'x_res' (hidden dim).
        
        # GINConv nn1: Linear(in_neurons, hidden_neurons) -> ... -> hidden_neurons
        h1 = self.conv1(x, edge_index)
        
        if self.use_residual and h1.shape == x_res.shape:
             h1 = h1 + x_res
             
        h1 = F.relu(self.ln1(h1))
        layer_outputs.append(h1)
        
        h2 = F.relu(self.ln2(self.conv2(h1, edge_index) + (h1 if self.use_residual else 0)))
        layer_outputs.append(h2)
        h3 = F.relu(self.ln3(self.conv3(h2, edge_index) + (h2 if self.use_residual else 0)))
        layer_outputs.append(h3)
        h4 = F.relu(self.ln4(self.conv4(h3, edge_index) + (h3 if self.use_residual else 0)))
        layer_outputs.append(h4)
        h5 = F.relu(self.ln5(self.conv5(h4, edge_index) + (h4 if self.use_residual else 0)))
        layer_outputs.append(h5)
        h6 = F.relu(self.ln6(self.conv6(h5, edge_index) + (h5 if self.use_residual else 0)))
        layer_outputs.append(h6)

        pooled_representations = []
        for layer_out in layer_outputs:
            pooled_representations.extend([global_mean_pool(layer_out, batch), global_max_pool(layer_out, batch), global_add_pool(layer_out, batch)])
        h_final = torch.cat(pooled_representations, dim=1)
        return F.normalize(self.readout_proj(h_final) + self.readout_skip(h_final), dim=1)

def info_nce_loss(queries, positives, temperature=0.1):
    logits = torch.matmul(queries, positives.T) / temperature
    labels = torch.arange(len(queries), device=queries.device)
    return F.cross_entropy(logits, labels)

def hierarchical_info_nce_loss(zq, z_fine, z_coarse, temperature=0.1, alpha=0.5):
    loss_fine = info_nce_loss(zq, z_fine, temperature)
    loss_coarse = info_nce_loss(zq, z_coarse, temperature)
    return (alpha * loss_fine) + ((1 - alpha) * loss_coarse)


def get_graph_embedding(data, encoder, device):
    """
    Get embedding for a single graph using the encoder.
    
    Args:
        data: PyG Data object
        encoder: ImprovedSubgraphEncoder model
        device: torch device
        
    Returns:
        Tensor of shape (1, output_dim)
    """
    from torch_geometric.data import Batch
    
    # Ensure model is in eval mode for inference
    was_training = encoder.training
    encoder.eval()
    
    # Move data to device
    data = data.to(device)
    
    # Create batch (single graph)
    batch_tensor = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    
    with torch.no_grad():
        embedding = encoder(data.x, data.edge_index, batch_tensor)
    
    # Restore training mode if needed
    if was_training:
        encoder.train()
    
    return embedding
