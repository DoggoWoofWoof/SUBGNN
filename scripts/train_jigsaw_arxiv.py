import modal
image = (
    modal.Image.debian_slim()
    .pip_install(
        # Core packages
        "numpy<2.0",
        "networkx==3.2.1",
        "pymetis==2022.1",
        "torch==2.2.1",
        "torch_geometric==2.5.2",
        "torch-scatter==2.1.2",
        "torch-sparse==0.6.18",
        # DGL is not used, but kept in image in case of future use.
        # If not needed, both dgl pip_installs can be removed.
        "dgl",
        # DGL's graphbolt dependencies / OGB dependencies
        "ogb>=1.3.6",
        "torchdata==0.7.1", # Pinned for stability
        "pandas",
        "PyYAML",
        "pydantic",
        # Add the two separate --find-links flags
        find_links="https://data.pyg.org/whl/torch-2.2.1+cu121.html",
    )
    # This second pip_install is needed for the second find-links URL.
    .pip_install(
        "dgl",
        find_links="https://data.dgl.ai/wheels/cu121/repo.html",
    )
    # Set the library path so C++ extensions can find Torch's CUDA libs.
    .env({"LD_LIBRARY_PATH": "/usr/local/lib/python3.11/site-packages/torch/lib"})
)

# It's good practice to rename the app to reflect the new dataset
app = modal.App("jigsaw-6_layer-arxiv-training", image=image)

@app.function(gpu="a100", timeout=32400)
def train(epochs, steps_per_epoch, batch_size, num_hierarchies=3):
    # --- REMOTE-ONLY IMPORTS and DEFINITIONS ---
    import itertools
    import random
    from collections import Counter, defaultdict

    import networkx as nx
    import pymetis
    import torch
    import torch.nn.functional as F
    #
    # FIX: Removed unused DGL import
    # from dgl.data import FlickrDataset
    #
    from torch.nn import Dropout, LeakyReLU, Linear, ReLU, Sequential, LayerNorm
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GINConv, global_mean_pool, GATConv, global_max_pool, global_add_pool
    from torch_geometric.utils import k_hop_subgraph, to_networkx, to_undirected

        # --- MODEL ARCHITECTURE ---
    class ImprovedSubgraphEncoder(torch.nn.Module):
        def __init__(self, in_neurons, hidden_neurons, output_neurons, dropout=0.1, use_residual=True, use_attention=False):
            super().__init__()
            self.use_residual = use_residual
            self.use_attention = use_attention
            self.dropout = dropout

            # Option 1: All GIN layers (original approach)
            if not use_attention:
                # Layer 1
                nn1 = Sequential(
                    Linear(in_neurons, hidden_neurons),
                    ReLU(),
                    Dropout(dropout),
                    Linear(hidden_neurons, hidden_neurons),
                )
                self.conv1 = GINConv(nn1)

                # Layer 2
                nn2 = Sequential(
                    Linear(hidden_neurons, hidden_neurons),
                    ReLU(),
                    Dropout(dropout),
                    Linear(hidden_neurons, hidden_neurons),
                )
                self.conv2 = GINConv(nn2)

                # Layer 3
                nn3 = Sequential(
                    Linear(hidden_neurons, hidden_neurons),
                    ReLU(),
                    Dropout(dropout),
                    Linear(hidden_neurons, hidden_neurons),
                )
                self.conv3 = GINConv(nn3)

                # Layer 4
                nn4 = Sequential(
                    Linear(hidden_neurons, hidden_neurons),
                    ReLU(),
                    Dropout(dropout),
                    Linear(hidden_neurons, hidden_neurons),
                )
                self.conv4 = GINConv(nn4)

                # Layer 5
                nn5 = Sequential(
                    Linear(hidden_neurons, hidden_neurons),
                    ReLU(),
                    Dropout(dropout),
                    Linear(hidden_neurons, hidden_neurons),
                )
                self.conv5 = GINConv(nn5)

                # Layer 6
                nn6 = Sequential(
                    Linear(hidden_neurons, hidden_neurons),
                    ReLU(),
                    Dropout(dropout),
                    Linear(hidden_neurons, hidden_neurons),
                )
                self.conv6 = GINConv(nn6)

            # Option 2: Mix of GIN and GAT layers
            else:
                # First 3 layers: GIN for local structure
                nn1 = Sequential(Linear(in_neurons, hidden_neurons), ReLU(), Dropout(dropout), Linear(hidden_neurons, hidden_neurons))
                self.conv1 = GINConv(nn1)

                nn2 = Sequential(Linear(hidden_neurons, hidden_neurons), ReLU(), Dropout(dropout), Linear(hidden_neurons, hidden_neurons))
                self.conv2 = GINConv(nn2)

                nn3 = Sequential(Linear(hidden_neurons, hidden_neurons), ReLU(), Dropout(dropout), Linear(hidden_neurons, hidden_neurons))
                self.conv3 = GINConv(nn3)

                # Last 3 layers: GAT for attention-based aggregation
                self.conv4 = GATConv(hidden_neurons, hidden_neurons // 8, heads=8, dropout=dropout, concat=True)
                self.conv5 = GATConv(hidden_neurons, hidden_neurons // 8, heads=8, dropout=dropout, concat=True)
                self.conv6 = GATConv(hidden_neurons, hidden_neurons, heads=1, dropout=dropout, concat=False)

            # Normalization layers (LayerNorm works better than BatchNorm for graphs)
            self.ln1 = LayerNorm(hidden_neurons)
            self.ln2 = LayerNorm(hidden_neurons)
            self.ln3 = LayerNorm(hidden_neurons)
            self.ln4 = LayerNorm(hidden_neurons)
            self.ln5 = LayerNorm(hidden_neurons)
            self.ln6 = LayerNorm(hidden_neurons)

            # Residual connection projections (if input/output dims don't match)
            self.input_proj = Linear(in_neurons, hidden_neurons) if in_neurons != hidden_neurons else None

            # Enhanced readout with multiple pooling strategies
            self.use_multi_pool = True
            if self.use_multi_pool:
                # Concatenate mean, max, and sum pooling from multiple layers
                readout_dim = hidden_neurons * 6 * 3  # 6 layers × 3 pooling methods
            else:
                # Original approach: just mean pooling from all layers
                readout_dim = hidden_neurons * 6

            # Final projection layers with residual connection
            self.readout_proj = Sequential(
                Linear(readout_dim, hidden_neurons * 2),
                ReLU(),
                Dropout(dropout),
                Linear(hidden_neurons * 2, hidden_neurons),
                ReLU(),
                Dropout(dropout),
                Linear(hidden_neurons, output_neurons)
            )

            # Skip connection for readout
            self.readout_skip = Linear(readout_dim, output_neurons)

        def forward(self, x, edge_index, batch):
            # Store all layer outputs for skip connections and readout
            layer_outputs = []

            # Initial projection if needed
            if self.input_proj is not None:
                x_res = self.input_proj(x)
            else:
                x_res = x

            # Layer 1
            h1 = self.conv1(x, edge_index)
            if self.use_residual and h1.shape == x_res.shape:
                h1 = h1 + x_res
            h1 = F.relu(self.ln1(h1))
            h1 = F.dropout(h1, p=self.dropout, training=self.training)
            layer_outputs.append(h1)

            # Layer 2
            h2 = self.conv2(h1, edge_index)
            if self.use_residual:
                h2 = h2 + h1
            h2 = F.relu(self.ln2(h2))
            h2 = F.dropout(h2, p=self.dropout, training=self.training)
            layer_outputs.append(h2)

            # Layer 3
            h3 = self.conv3(h2, edge_index)
            if self.use_residual:
                h3 = h3 + h2
            h3 = F.relu(self.ln3(h3))
            h3 = F.dropout(h3, p=self.dropout, training=self.training)
            layer_outputs.append(h3)

            # Layer 4
            h4 = self.conv4(h3, edge_index)
            if self.use_residual:
                h4 = h4 + h3
            h4 = F.relu(self.ln4(h4))
            h4 = F.dropout(h4, p=self.dropout, training=self.training)
            layer_outputs.append(h4)

            # Layer 5
            h5 = self.conv5(h4, edge_index)
            if self.use_residual:
                h5 = h5 + h4
            h5 = F.relu(self.ln5(h5))
            h5 = F.dropout(h5, p=self.dropout, training=self.training)
            layer_outputs.append(h5)

            # Layer 6
            h6 = self.conv6(h5, edge_index)
            if self.use_residual:
                h6 = h6 + h5
            h6 = F.relu(self.ln6(h6))
            h6 = F.dropout(h6, p=self.dropout, training=self.training)
            layer_outputs.append(h6)

            # Enhanced readout with multiple pooling strategies
            if self.use_multi_pool:
                pooled_representations = []
                for layer_out in layer_outputs:
                    pooled_representations.extend([
                        global_mean_pool(layer_out, batch),
                        global_max_pool(layer_out, batch),
                        global_add_pool(layer_out, batch)
                    ])
                h_final = torch.cat(pooled_representations, dim=1)
            else:
                # Original approach
                h_final = torch.cat([
                    global_mean_pool(layer_out, batch) for layer_out in layer_outputs
                ], dim=1)

            # Final projection with skip connection
            main_output = self.readout_proj(h_final)
            skip_output = self.readout_skip(h_final)
            final_output = main_output + skip_output

            return F.normalize(final_output, dim=1)

    # --- HIERARCHICAL LOSS ---
    def info_nce_loss(queries, positives, temperature=0.1):
        logits = torch.matmul(queries, positives.T) / temperature
        labels = torch.arange(len(queries), device=queries.device)
        return F.cross_entropy(logits, labels)

    def hierarchical_info_nce_loss(zq, z_fine, z_coarse, temperature=0.1, alpha=0.5):
        loss_fine = info_nce_loss(zq, z_fine, temperature)
        loss_coarse = info_nce_loss(zq, z_coarse, temperature)
        return (alpha * loss_fine) + ((1 - alpha) * loss_coarse)

    # --- DATA PARTITIONING AND HIERARCHY HELPERS ---
    def make_partitions(dataset, num_parts):
        if dataset.num_nodes < num_parts:
            num_parts = dataset.num_nodes
        if num_parts <= 1:
            return [dataset], {
                0: torch.arange(dataset.num_nodes, device=dataset.x.device)
            }
        adj = to_networkx(dataset, to_undirected=True).adjacency()
        adj_list = [list(neighbors) for _, neighbors in adj]
        _, membership = pymetis.part_graph(num_parts, adjacency=adj_list)
        part_graphs, part_nodes_map = [], {}
        for part_id in range(num_parts):
            node_indices = [i for i, p in enumerate(membership) if p == part_id]
            if node_indices:
                nodes_tensor = torch.tensor(
                    node_indices, dtype=torch.long, device=dataset.x.device
                )
                part_nodes_map[part_id] = nodes_tensor
                # WORKAROUND: Use boolean mask for subgraphing on GPU to avoid PyG bug
                mask = torch.zeros(
                    dataset.num_nodes, dtype=torch.bool, device=dataset.x.device
                )
                mask[nodes_tensor] = True
                part_graphs.append(dataset.subgraph(mask))
        return part_graphs, part_nodes_map

    def build_single_hierarchy(data, num_coarse, num_fine):
        """
        Builds a two-level graph hierarchy.
        1. Partitions the original graph into 'num_coarse' partitions.
        2. Further partitions each coarse partition into 'num_fine' smaller partitions.
        Includes robustness checks and filters for creating usable hierarchies.
        """
        print(
            f"\n  • Building hierarchy with {num_coarse} coarse and {num_coarse*num_fine} (target) fine partitions..."
        )

        # 1. Create the coarse-level partitions from the original graph
        coarse_graphs, coarse_part_nodes_map = make_partitions(data, num_coarse)

        # 2. Create a mapping from each original node to its coarse partition ID
        node_to_coarse_map = {}
        for coarse_id, nodes in coarse_part_nodes_map.items():
            for node_idx in nodes:
                node_to_coarse_map[node_idx.item()] = coarse_id

        # 3. Build the abstract "graph of partitions" (coarse_part_graph)
        coarse_part_graph = nx.Graph()
        for u, v in data.edge_index.t().tolist():
            c_u, c_v = node_to_coarse_map.get(u), node_to_coarse_map.get(v)
            if c_u is not None and c_v is not None and c_u != c_v:
                coarse_part_graph.add_edge(c_u, c_v)

        # 4. Create the fine-level partitions by sub-partitioning each coarse graph
        fine_graphs, fine_part_nodes_map, fine_to_coarse_map = [], {}, {}
        fine_global_idx = 0
        for coarse_idx, coarse_graph in enumerate(coarse_graphs):
            if coarse_idx not in coarse_part_nodes_map:
                continue

            global_nodes_of_this_coarse_part = coarse_part_nodes_map[coarse_idx]

            # Robustness Check: Don't try to sub-partition if the coarse graph is too small or has no edges
            if coarse_graph.num_nodes < (num_fine * 2) or coarse_graph.num_edges == 0:
                finer_partitions, finer_nodes_map_local = [coarse_graph], {
                    0: torch.arange(coarse_graph.num_nodes, device=data.x.device)
                }
            else:
                finer_partitions, finer_nodes_map_local = make_partitions(
                    coarse_graph, num_fine
                )

            # Process the results of the sub-partitioning
            for fine_local_idx, fine_part in enumerate(finer_partitions):
                if fine_local_idx not in finer_nodes_map_local:
                    continue

                local_indices_in_coarse = finer_nodes_map_local[fine_local_idx]
                global_indices_for_fine = global_nodes_of_this_coarse_part[
                    local_indices_in_coarse
                ]

                # Quality Filter: Only keep fine partitions that are non-trivial
                if fine_part.num_nodes > 10 and fine_part.num_edges > 0:
                    fine_graphs.append(fine_part)
                    fine_part_nodes_map[fine_global_idx] = global_indices_for_fine
                    fine_to_coarse_map[fine_global_idx] = coarse_idx
                    fine_global_idx += 1

        # --- MODIFIED FINAL PRINT STATEMENT ---
        # Calculate statistics for the actual coarse partition subgraphs
        if coarse_graphs:
            partition_sizes = [g.num_nodes for g in coarse_graphs]
            min_size, max_size, avg_size = (
                min(partition_sizes),
                max(partition_sizes),
                sum(partition_sizes) / len(partition_sizes),
            )
        else:
            min_size, max_size, avg_size = 0, 0, 0.0

        # The new log message combines all relevant information into one summary line
        print(
            f"    - Created {len(fine_graphs)} usable fine partitions from {len(coarse_graphs)} coarse partitions "
            f"(Node counts: Min={min_size}, Max={max_size}, Avg={avg_size:.2f}). "
            f"Coarse connectivity graph has {coarse_part_graph.number_of_edges()} edges."
        )

        return (
            coarse_graphs,
            fine_graphs,
            node_to_coarse_map,
            fine_to_coarse_map,
            fine_part_nodes_map,
            coarse_part_graph,
            coarse_part_nodes_map,
        )

    def build_multiple_hierarchies(data, n_hierarchies):
        print(
            f"[SETUP] Building {n_hierarchies} different hierarchies for Jigsaw training..."
        )
        hierarchies = []
        for i in range(n_hierarchies):
            num_coarse = random.randint(140, 200)
            num_fine = random.randint(5, 10)
            hierarchy_data = build_single_hierarchy(data, num_coarse, num_fine)
            hierarchies.append(hierarchy_data)
        return hierarchies

    # --- DATA GENERATION HELPERS ---
    def _extract_fragment(source_graph, target_size):
        if source_graph.num_nodes < target_size / 2:
            return None
        source_nx = to_networkx(source_graph, to_undirected=True)
        if not nx.is_connected(source_nx):
            try:
                largest_cc_nodes = max(nx.connected_components(source_nx), key=len)
            except ValueError:
                return None
            source_nx = source_nx.subgraph(largest_cc_nodes)
        if source_nx.number_of_nodes() == 0:
            return None
        start_node = random.choice(list(source_nx.nodes()))
        q_nodes = list(nx.bfs_tree(source_nx, start_node, depth_limit=5).nodes())[
            :target_size
        ]
        if len(q_nodes) < target_size / 2:
            return None
        return torch.tensor(q_nodes, dtype=torch.long)

    def _finalize_query_from_nodes(original_data, global_node_indices, min_nodes):
        if not global_node_indices:
            return None, None
        q_global_nodes = torch.tensor(
            list(set(global_node_indices)),
            dtype=torch.long,
            device=original_data.x.device,
        )
        if len(q_global_nodes) < min_nodes:
            return None, None
        # WORKAROUND: Use boolean mask for subgraphing on GPU to avoid PyG bug
        mask = torch.zeros(
            original_data.num_nodes, dtype=torch.bool, device=original_data.x.device
        )
        mask[q_global_nodes] = True
        return original_data.subgraph(mask), q_global_nodes

    def are_partitions_neighbors(G_nx, nodes1, nodes2):
        nodes2_set = set(nodes2.tolist())
        for node in nodes1:
            for neighbor in G_nx.neighbors(node.item()):
                if neighbor in nodes2_set:
                    return True
        return False

    # --- MODIFIED: FUNCTION WITH RANDOMIZED-ENTRY FALLBACK ---
    def generate_multi_coarse_partition_query(
        original_data,
        G_nx,
        coarse_part_graph,
        fine_graphs,
        fine_part_nodes_map,
        fine_to_coarse_map,
        min_nodes=80,
        max_nodes=100,
    ):
        if coarse_part_graph.number_of_edges() == 0:
            raise RuntimeError("Coarse graph has no edges.")
        configurations = [(4, 4), (4, 3), (3, 3), (4, 2), (3, 2), (2, 2)]
        start_index = random.randint(0, len(configurations) - 1)
        reordered_configs_to_try = (
            configurations[start_index:] + configurations[:start_index]
        )
        coarse_to_fine_map = defaultdict(list)
        for f_idx, c_idx in fine_to_coarse_map.items():
            coarse_to_fine_map[c_idx].append(f_idx)
        for num_frags, min_coarse_parts in reordered_configs_to_try:
            possible_start_edges = list(coarse_part_graph.edges())
            random.shuffle(possible_start_edges)
            for c_idx1, c_idx2 in possible_start_edges:
                fine_parts_in_c1 = coarse_to_fine_map.get(c_idx1, [])
                fine_parts_in_c2 = coarse_to_fine_map.get(c_idx2, [])
                all_fine_pairs = list(
                    itertools.product(fine_parts_in_c1, fine_parts_in_c2)
                )
                random.shuffle(all_fine_pairs)
                for f1, f2 in all_fine_pairs:
                    if not are_partitions_neighbors(
                        G_nx, fine_part_nodes_map[f1], fine_part_nodes_map[f2]
                    ):
                        continue
                    q_fine_indices, queue, visited = [f1, f2], [f1, f2], {f1, f2}
                    while queue and len(q_fine_indices) < num_frags:
                        current_fine_idx = queue.pop(0)
                        current_c_idx = fine_to_coarse_map[current_fine_idx]
                        coarse_neighbors_and_self = list(
                            coarse_part_graph.neighbors(current_c_idx)
                        ) + [current_c_idx]
                        potential_fine_neighbors = [
                            fn
                            for c_idx in coarse_neighbors_and_self
                            for fn in coarse_to_fine_map.get(c_idx, [])
                        ]
                        random.shuffle(potential_fine_neighbors)
                        for neighbor_idx in potential_fine_neighbors:
                            if (
                                neighbor_idx not in visited
                                and are_partitions_neighbors(
                                    G_nx,
                                    fine_part_nodes_map[current_fine_idx],
                                    fine_part_nodes_map[neighbor_idx],
                                )
                            ):
                                visited.add(neighbor_idx)
                                queue.append(neighbor_idx)
                                q_fine_indices.append(neighbor_idx)
                                if len(q_fine_indices) >= num_frags:
                                    break
                    if len(q_fine_indices) < num_frags:
                        continue
                    true_coarse_indices = {
                        fine_to_coarse_map[f_idx] for f_idx in q_fine_indices
                    }
                    if len(true_coarse_indices) < min_coarse_parts:
                        continue
                    nodes_per_frag = max_nodes // num_frags
                    all_query_nodes = []
                    for fine_idx in q_fine_indices:
                        local_nodes = _extract_fragment(
                            fine_graphs[fine_idx], nodes_per_frag
                        )
                        if local_nodes is not None:
                            all_query_nodes.extend(
                                fine_part_nodes_map[fine_idx][local_nodes].tolist()
                            )
                    Gq, _ = _finalize_query_from_nodes(
                        original_data, all_query_nodes, min_nodes
                    )
                    if Gq:
                        stitched_nodes = torch.cat(
                            [fine_part_nodes_map[idx] for idx in q_fine_indices]
                        )
                        # WORKAROUND: Use boolean mask for subgraphing on GPU to avoid PyG bug
                        mask = torch.zeros(
                            original_data.num_nodes,
                            dtype=torch.bool,
                            device=original_data.x.device,
                        )
                        mask[stitched_nodes] = True
                        G_stitched = original_data.subgraph(mask)
                        return Gq, G_stitched, true_coarse_indices
        raise RuntimeError(
            "Failed to generate multi-coarse-partition query after trying all valid configurations."
        )

    def generate_hierarchical_sample(
        original_data,
        G_nx,
        coarse_graphs,
        fine_graphs,
        node_to_coarse_map,
        fine_to_coarse_map,
        fine_part_nodes_map,
        coarse_part_nodes_map,
        coarse_part_graph,
        k=6,
        q_size_min=20,
        q_size_max=120,
        prob_k_hop=0.2,
        prob_single_part=0.2,
        prob_multi_coarse=0.4,
        max_gpos_nodes=4000,
    ):
        rand_choice = random.random()
        device = original_data.x.device
        Gq, Gpos, G_coarse_pos = None, None, None

        if rand_choice < prob_k_hop:  # View 1: K-hop subgraph
            anchor = random.randint(0, original_data.num_nodes - 1)
            subset_pos, _, _, _ = k_hop_subgraph(
                anchor, k, original_data.edge_index, relabel_nodes=False
            )
            if len(subset_pos) < q_size_min or len(subset_pos) > max_gpos_nodes:
                return None
            # WORKAROUND: Use a boolean mask for subgraphing on GPU to avoid PyG bug
            pos_mask = torch.zeros(
                original_data.num_nodes, dtype=torch.bool, device=device
            )
            pos_mask[subset_pos] = True
            Gpos = original_data.subgraph(pos_mask)
            parent_counts = Counter(
                node_to_coarse_map.get(node_id.item()) for node_id in subset_pos
            )
            if not parent_counts:
                return None
            coarse_parent_idx = parent_counts.most_common(1)[0][0]
            if coarse_parent_idx is None:
                return None

            # --- START FIX ---
            # The original `Gpos.subgraph(indices_tensor)` call caused the error due to a
            # device mismatch bug in this PyG version. The fix is to manually create a
            # boolean mask on the correct device (GPU) before calling subgraph.
            q_nodes_local_indices = torch.randperm(Gpos.num_nodes)[
                : random.randint(q_size_min, min(q_size_max, Gpos.num_nodes))
            ]
            q_mask = torch.zeros(Gpos.num_nodes, dtype=torch.bool, device=device)
            q_mask[q_nodes_local_indices.to(device)] = True
            Gq = Gpos.subgraph(q_mask)
            # --- END FIX ---

            G_coarse_pos = coarse_graphs[coarse_parent_idx]

        elif (
            rand_choice < prob_k_hop + prob_single_part
        ):  # View 2: Single fine partition
            if not fine_graphs:
                return None
            fine_idx = random.choice(list(fine_to_coarse_map.keys()))
            Gpos = fine_graphs[fine_idx]
            if Gpos.num_nodes > max_gpos_nodes:
                return None
            q_nodes_local = _extract_fragment(
                Gpos, random.randint(q_size_min, q_size_max)
            )
            if q_nodes_local is None:
                return None
            # WORKAROUND: Use boolean mask for subgraphing on GPU to avoid PyG bug
            q_mask = torch.zeros(Gpos.num_nodes, dtype=torch.bool, device=device)
            q_mask[q_nodes_local.to(device)] = True
            Gq = Gpos.subgraph(q_mask)
            coarse_parent_idx = fine_to_coarse_map.get(fine_idx)
            if coarse_parent_idx is None:
                return None
            G_coarse_pos = coarse_graphs[coarse_parent_idx]

        elif (
            rand_choice < prob_k_hop + prob_single_part + prob_multi_coarse
        ):  # View 3: Multi-coarse query
            try:
                Gq, Gpos, coarse_indices = generate_multi_coarse_partition_query(
                    original_data,
                    G_nx,
                    coarse_part_graph,
                    fine_graphs,
                    fine_part_nodes_map,
                    fine_to_coarse_map,
                    min_nodes=q_size_min,
                    max_nodes=q_size_max,
                )
                all_coarse_pos_nodes = torch.cat(
                    [coarse_part_nodes_map[c_idx] for c_idx in coarse_indices]
                )
                # WORKAROUND: Use boolean mask for subgraphing on GPU to avoid PyG bug
                coarse_mask = torch.zeros(
                    original_data.num_nodes, dtype=torch.bool, device=device
                )
                coarse_mask[all_coarse_pos_nodes] = True
                G_coarse_pos = original_data.subgraph(coarse_mask)
            except RuntimeError:
                return None

        else:  # View 4: Stitched from neighboring fine partitions within the SAME coarse parent
            if not fine_part_nodes_map or len(fine_part_nodes_map) < 2:
                return None
            num_frags = random.randint(2, 3)
            start_fine_idx = random.choice(list(fine_part_nodes_map.keys()))
            coarse_parent_idx = fine_to_coarse_map.get(start_fine_idx)
            if coarse_parent_idx is None:
                return None
            siblings = [
                idx
                for idx, c_idx in fine_to_coarse_map.items()
                if c_idx == coarse_parent_idx
            ]
            source_part_indices = {start_fine_idx}
            queue = [start_fine_idx]
            while queue and len(source_part_indices) < num_frags:
                curr_idx = queue.pop(0)
                random.shuffle(siblings)
                for neighbor_idx in siblings:
                    if (
                        neighbor_idx not in source_part_indices
                        and are_partitions_neighbors(
                            G_nx,
                            fine_part_nodes_map[curr_idx],
                            fine_part_nodes_map[neighbor_idx],
                        )
                    ):
                        source_part_indices.add(neighbor_idx)
                        queue.append(neighbor_idx)
                        if len(source_part_indices) >= num_frags:
                            break
            if len(source_part_indices) < num_frags:
                return None
            pos_nodes = torch.cat([fine_part_nodes_map[i] for i in source_part_indices])
            if len(pos_nodes) > max_gpos_nodes:
                return None
            # WORKAROUND: Use boolean mask for subgraphing on GPU to avoid PyG bug
            pos_mask = torch.zeros(
                original_data.num_nodes, dtype=torch.bool, device=device
            )
            pos_mask[pos_nodes] = True
            Gpos = original_data.subgraph(pos_mask)
            nodes_per_frag = (q_size_min + q_size_max) // (2 * num_frags)
            all_query_global_nodes = []
            for fine_idx in source_part_indices:
                local_indices = _extract_fragment(fine_graphs[fine_idx], nodes_per_frag)
                if local_indices is None:
                    return None
                all_query_global_nodes.extend(
                    fine_part_nodes_map[fine_idx][local_indices].tolist()
                )
            Gq, _ = _finalize_query_from_nodes(
                original_data, all_query_global_nodes, q_size_min
            )
            if Gq is None:
                return None
            G_coarse_pos = coarse_graphs[coarse_parent_idx]

        if Gq is None or Gpos is None or G_coarse_pos is None:
            return None
        return Gq, Gpos, G_coarse_pos

    # --- CORE TRAINING LOGIC ---
    device = torch.device("cuda")
    # --- DATASET LOADING (MODIFIED FOR OGBN-ARXIV) ---
    from ogb.nodeproppred import PygNodePropPredDataset
    import torch_geometric.transforms as T
    import torch # Need to import torch here for torch.save in main() to work

    print("[REMOTE INFO] Loading ogbn-arxiv dataset using OGB...")

    # Load the dataset directly into PyG format.
    # We use a transform to make the graph undirected upon loading, which is a best practice.
    # A root directory is specified to store data in Modal's ephemeral filesystem.
    dataset = PygNodePropPredDataset(
        name='ogbn-arxiv',
        root='/tmp/ogbn_arxiv_data', # A temporary directory for the dataset
        transform=T.ToUndirected()
    )
    # The 'data' object is the actual graph (a torch_geometric.data.Data instance)
    data = dataset[0]

    # Move the entire data object to the specified device (e.g., 'cuda')
    data = data.to(device)

    print(f"  - Dataset loaded directly in PyG format: {data}")
    print(f"  - Nodes: {data.num_nodes}, Edges: {data.num_edges}, Features: {data.num_features}")

    print(f"[REMOTE INFO] Using device: {device}")

    # The dataset object has a .num_features property, so this is correct.
    encoder = ImprovedSubgraphEncoder(dataset.num_features, 256, 128, use_attention=False, dropout=0.1, use_residual=True).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )
    encoder.train()

    #
    # --- FIX: Pass the 'data' object (the graph), not the 'dataset' object ---
    #
    hierarchies = build_multiple_hierarchies(data, num_hierarchies)
    G_nx = to_networkx(data, to_undirected=True)

    print("-" * 50)
    for epoch in range(epochs):
        total_loss = 0
        for step in range(steps_per_epoch):
            query_graphs, pos_graphs, coarse_pos_graphs = [], [], []

            chosen_hierarchy = random.choice(hierarchies)
            (
                coarse_graphs,
                fine_graphs,
                node_to_coarse_map,
                fine_to_coarse_map,
                fine_part_nodes_map,
                coarse_part_graph,
                coarse_part_nodes_map,
            ) = chosen_hierarchy

            while len(query_graphs) < batch_size:
                sample = generate_hierarchical_sample(
                    data, # FIX: Pass the 'data' object here too
                    G_nx,
                    coarse_graphs,
                    fine_graphs,
                    node_to_coarse_map,
                    fine_to_coarse_map,
                    fine_part_nodes_map,
                    coarse_part_nodes_map,
                    coarse_part_graph,
                )
                if sample:
                    gq, gpos, g_coarse = sample
                    if (
                        gq.num_nodes > 1
                        and gpos.num_nodes > 1
                        and g_coarse.num_nodes > 1
                    ):
                        query_graphs.append(gq)
                        pos_graphs.append(gpos)
                        coarse_pos_graphs.append(g_coarse)

            if not query_graphs:
                print(
                    f"Warning: No valid samples generated in step {step+1} of epoch {epoch+1}. Skipping batch."
                )
                continue

            query_batch = Batch.from_data_list(query_graphs)
            pos_batch = Batch.from_data_list(pos_graphs)
            coarse_pos_batch = Batch.from_data_list(coarse_pos_graphs)

            zq = encoder(query_batch.x, query_batch.edge_index, query_batch.batch)
            z_pos = encoder(pos_batch.x, pos_batch.edge_index, pos_batch.batch)
            z_coarse = encoder(
                coarse_pos_batch.x, coarse_pos_batch.edge_index, coarse_pos_batch.batch
            )

            loss = hierarchical_info_nce_loss(zq, z_pos, z_coarse)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / steps_per_epoch if steps_per_epoch > 0 else 0
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{epochs}: Avg Loss = {avg_loss:.6f}, LR = {current_lr:.1e}"
        )

    print("\n[REMOTE INFO] Training finished.")
    encoder.to("cpu")
    return encoder.state_dict()


# --- THE LOCAL ENTRYPOINT ---
@app.local_entrypoint()
def main():
    #
    # FIX: Need to import torch locally for torch.save()
    #
    import torch

    print("🚀 Starting Jigsaw GNN training on Modal...")
    model_state_dict = train.remote(
        epochs=150, steps_per_epoch=50, batch_size=64, num_hierarchies=5
    )
    #
    # FIX: Updated filename to match the dataset
    #
    file_path = "arxiv-6_layer-model-jigsaw.pth"
    torch.save(model_state_dict, file_path)
    print(f"✅ Model saved to '{file_path}'")