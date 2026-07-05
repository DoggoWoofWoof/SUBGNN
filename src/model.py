import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, LeakyReLU, Linear, ReLU, Sequential, LayerNorm
from torch_geometric.nn import GINConv, RGCNConv, global_mean_pool, global_max_pool, global_add_pool

class ImprovedSubgraphEncoder(torch.nn.Module):
    def __init__(self, in_neurons, hidden_neurons, output_neurons, dropout=0.1, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.dropout = dropout

        nn1 = Sequential(Linear(hidden_neurons, hidden_neurons), ReLU(), Dropout(dropout), Linear(hidden_neurons, hidden_neurons))
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
        self.input_proj = Linear(in_neurons, hidden_neurons)
        self.use_multi_pool = True; readout_dim = hidden_neurons * 6 * 3

        self.readout_proj = Sequential(
            Linear(readout_dim, hidden_neurons * 2), ReLU(), Dropout(dropout),
            Linear(hidden_neurons * 2, hidden_neurons), ReLU(), Dropout(dropout),
            Linear(hidden_neurons, output_neurons)
        )
        self.readout_skip = Linear(readout_dim, output_neurons)

    def forward(self, x, edge_index, batch):
        layer_outputs = []
        feat = x.x if hasattr(x, 'x') else x

        # Project to hidden space immediately
        feat = F.relu(self.input_proj(feat))
        x_res = feat 

        h1 = F.relu(self.ln1(self.conv1(feat, edge_index) + x_res))
        layer_outputs.append(h1)
        h2 = F.relu(self.ln2(self.conv2(h1, edge_index) + h1))
        layer_outputs.append(h2)
        h3 = F.relu(self.ln3(self.conv3(h2, edge_index) + h2))
        layer_outputs.append(h3)
        h4 = F.relu(self.ln4(self.conv4(h3, edge_index) + h3))
        layer_outputs.append(h4)
        h5 = F.relu(self.ln5(self.conv5(h4, edge_index) + h4))
        layer_outputs.append(h5)
        h6 = F.relu(self.ln6(self.conv6(h5, edge_index) + h5))
        layer_outputs.append(h6)

        pooled_representations = []
        for layer_out in layer_outputs:
            pooled_representations.extend([global_mean_pool(layer_out, batch), global_max_pool(layer_out, batch), global_add_pool(layer_out, batch)])
        h_final = torch.cat(pooled_representations, dim=1)
        graph_emb = F.normalize(self.readout_proj(h_final) + self.readout_skip(h_final), dim=1)
        
        node_emb = F.normalize(h6, dim=1)
        return graph_emb, node_emb


class RelationAwareSubgraphEncoder(torch.nn.Module):
    """RGCN graph encoder used for heterogeneous MAG experiments."""

    def __init__(
        self,
        in_neurons,
        hidden_neurons,
        output_neurons,
        num_relations,
        dropout=0.1,
        use_residual=True,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.dropout = dropout
        self.num_relations = max(1, int(num_relations))
        num_bases = min(8, self.num_relations)

        self.conv1 = RGCNConv(hidden_neurons, hidden_neurons, self.num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_neurons, hidden_neurons, self.num_relations, num_bases=num_bases)
        self.conv3 = RGCNConv(hidden_neurons, hidden_neurons, self.num_relations, num_bases=num_bases)
        self.conv4 = RGCNConv(hidden_neurons, hidden_neurons, self.num_relations, num_bases=num_bases)
        self.conv5 = RGCNConv(hidden_neurons, hidden_neurons, self.num_relations, num_bases=num_bases)
        self.conv6 = RGCNConv(hidden_neurons, hidden_neurons, self.num_relations, num_bases=num_bases)

        self.ln1 = LayerNorm(hidden_neurons); self.ln2 = LayerNorm(hidden_neurons); self.ln3 = LayerNorm(hidden_neurons)
        self.ln4 = LayerNorm(hidden_neurons); self.ln5 = LayerNorm(hidden_neurons); self.ln6 = LayerNorm(hidden_neurons)
        self.input_proj = Linear(in_neurons, hidden_neurons)
        readout_dim = hidden_neurons * 6 * 3

        self.readout_proj = Sequential(
            Linear(readout_dim, hidden_neurons * 2), ReLU(), Dropout(dropout),
            Linear(hidden_neurons * 2, hidden_neurons), ReLU(), Dropout(dropout),
            Linear(hidden_neurons, output_neurons)
        )
        self.readout_skip = Linear(readout_dim, output_neurons)

    def _edge_type(self, edge_index, edge_type):
        if edge_type is None or edge_type.numel() != edge_index.size(1):
            return torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
        edge_type = edge_type.to(edge_index.device).long()
        return edge_type.clamp(min=0, max=self.num_relations - 1)

    def forward(self, x, edge_index, batch, edge_type=None):
        layer_outputs = []
        feat = x.x if hasattr(x, 'x') else x
        rel = self._edge_type(edge_index, edge_type)

        feat = F.relu(self.input_proj(feat))
        x_res = feat

        h1 = F.relu(self.ln1(self.conv1(feat, edge_index, rel) + (x_res if self.use_residual else 0)))
        layer_outputs.append(h1)
        h2 = F.relu(self.ln2(self.conv2(h1, edge_index, rel) + (h1 if self.use_residual else 0)))
        layer_outputs.append(h2)
        h3 = F.relu(self.ln3(self.conv3(h2, edge_index, rel) + (h2 if self.use_residual else 0)))
        layer_outputs.append(h3)
        h4 = F.relu(self.ln4(self.conv4(h3, edge_index, rel) + (h3 if self.use_residual else 0)))
        layer_outputs.append(h4)
        h5 = F.relu(self.ln5(self.conv5(h4, edge_index, rel) + (h4 if self.use_residual else 0)))
        layer_outputs.append(h5)
        h6 = F.relu(self.ln6(self.conv6(h5, edge_index, rel) + (h5 if self.use_residual else 0)))
        layer_outputs.append(h6)

        pooled_representations = []
        for layer_out in layer_outputs:
            pooled_representations.extend([
                global_mean_pool(layer_out, batch),
                global_max_pool(layer_out, batch),
                global_add_pool(layer_out, batch),
            ])
        h_final = torch.cat(pooled_representations, dim=1)
        graph_emb = F.normalize(self.readout_proj(h_final) + self.readout_skip(h_final), dim=1)
        node_emb = F.normalize(h6, dim=1)
        return graph_emb, node_emb

def info_nce_loss(queries, positives, hard_negatives=None, temperature=0.1):
    # positives sim
    pos_sim = torch.sum(queries * positives, dim=1, keepdim=True) / temperature
    
    # in-batch negatives
    neg_sim = torch.matmul(queries, positives.T) / temperature
    mask = torch.eye(len(queries), device=queries.device, dtype=torch.bool)
    neg_sim[mask] = -float('inf') 
    
    all_sims = [pos_sim, neg_sim]
    
    if hard_negatives is not None:
        B, num_hn, D = hard_negatives.shape
        q_exp = queries.unsqueeze(1)
        hn_sim = torch.sum(q_exp * hard_negatives, dim=2) / temperature
        all_sims.append(hn_sim)
        
    logits = torch.cat(all_sims, dim=1)
    labels = torch.zeros(len(queries), dtype=torch.long, device=queries.device)
    return F.cross_entropy(logits, labels)

def node_alignment_loss(q_nodes, q_batch, p_nodes, p_batch, temperature=0.1):
    if not hasattr(q_batch, 'global_id') or not hasattr(p_batch, 'global_id'):
        return 0.0

    q_gids = q_batch.global_id
    p_gids = p_batch.global_id
    
    # Unique Q
    q_unique_gids, q_inv = torch.unique(q_gids, return_inverse=True)
    rev_idx_q = len(q_gids) - 1 - torch.arange(len(q_gids), device=q_gids.device)
    first_q_indices = torch.zeros(len(q_unique_gids), dtype=torch.long, device=q_gids.device)
    first_q_indices.scatter_(0, q_inv, rev_idx_q)
    first_q_indices = len(q_gids) - 1 - first_q_indices
    q_nodes_unique = q_nodes[first_q_indices]
    
    # Unique P
    p_unique_gids, p_inv = torch.unique(p_gids, return_inverse=True)
    rev_idx_p = len(p_gids) - 1 - torch.arange(len(p_gids), device=p_gids.device)
    first_p_indices = torch.zeros(len(p_unique_gids), dtype=torch.long, device=p_gids.device)
    first_p_indices.scatter_(0, p_inv, rev_idx_p)
    first_p_indices = len(p_gids) - 1 - first_p_indices
    p_nodes_unique = p_nodes[first_p_indices]

    p_max_gid = p_unique_gids.max().item() + 1
    q_max_gid = q_unique_gids.max().item() + 1
    max_gid = max(p_max_gid, q_max_gid)
    
    p_lookup = torch.full((max_gid,), -1, dtype=torch.long, device=p_gids.device)
    p_lookup[p_unique_gids] = torch.arange(len(p_unique_gids), device=p_gids.device)
    
    p_indices = p_lookup[q_unique_gids]
    match_mask = p_indices >= 0
    
    if match_mask.sum() < 2:
        return 0.0
    
    q_final_feats = q_nodes_unique[match_mask]
    p_final_feats = p_nodes_unique[p_indices[match_mask]]
    
    logits = torch.matmul(q_final_feats, p_final_feats.T) / temperature
    labels = torch.arange(len(q_final_feats), device=q_final_feats.device)
    return F.cross_entropy(logits, labels)

def hierarchical_info_nce_loss(zq, z_fine, z_coarse, q_node_emb, p_node_emb, q_batch, p_batch, hard_negatives=None, temperature=0.1, alpha=0.2, beta=0.0):
    loss_fine = info_nce_loss(zq, z_fine, temperature=temperature)
    loss_coarse = info_nce_loss(zq, z_coarse, hard_negatives=hard_negatives, temperature=temperature)
    loss_node = 0.0 if beta == 0.0 else node_alignment_loss(q_node_emb, q_batch, p_node_emb, p_batch, temperature=temperature)
    return (alpha * loss_fine) + ((1 - alpha - beta) * loss_coarse) + (beta * loss_node)

def partition_coverage_loss(zq, coarse_part_embs, query_coarse_ids, temperature=0.05):
    """
    Coverage loss for multi-partition queries.

    This treats every touched coarse partition as a required positive. The old
    logsumexp-positive objective could be satisfied by ranking only one of many
    true partitions highly, which is exactly the failure mode for k-hop recall.
    """
    B, D = zq.shape
    num_coarse = coarse_part_embs.shape[0]
    logits = torch.matmul(zq, coarse_part_embs.T) / temperature
    
    pos_mask = torch.zeros(B, num_coarse, dtype=torch.bool, device=zq.device)
    for i, part_ids in enumerate(query_coarse_ids):
        for pid in part_ids:
            if 0 <= pid < num_coarse:
                pos_mask[i, pid] = True
    
    valid = pos_mask.any(dim=1)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=zq.device)
    
    logits = logits[valid]
    pos_mask = pos_mask[valid]

    pos_counts = pos_mask.sum(dim=1).clamp_min(1).float()

    # Per-positive softmax CE: each true partition must receive probability mass.
    log_probs = F.log_softmax(logits, dim=1)
    per_positive_ce = -(log_probs * pos_mask.float()).sum(dim=1) / pos_counts

    # Hard-negative margin: every positive should beat the strongest negative.
    neg_logits = logits.masked_fill(pos_mask, float('-inf'))
    hardest_neg = neg_logits.max(dim=1).values
    has_negative = torch.isfinite(hardest_neg)
    margin_terms = torch.zeros_like(per_positive_ce)
    if has_negative.any():
        raw_margin = F.softplus(hardest_neg.unsqueeze(1) - logits)
        margin_terms = (raw_margin * pos_mask.float()).sum(dim=1) / pos_counts
        margin_terms = torch.where(has_negative, margin_terms, torch.zeros_like(margin_terms))

    return (per_positive_ce + 0.25 * margin_terms).mean()

def get_graph_embedding(data, encoder, device):
    """Get embedding for a single graph using the encoder."""
    was_training = encoder.training
    encoder.eval()
    data = data.to(device)
    batch_tensor = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    with torch.no_grad():
        # encoder returns (graph_emb, node_emb)
        edge_type = getattr(data, "edge_type", None)
        try:
            embedding, _ = encoder(data.x, data.edge_index, batch_tensor, edge_type)
        except TypeError:
            embedding, _ = encoder(data.x, data.edge_index, batch_tensor)
    if was_training:
        encoder.train()
    return embedding
