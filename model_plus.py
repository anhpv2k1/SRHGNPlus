import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.functional import edge_softmax


def normalize(x):
    """L2 normalization"""
    return x / (torch.max(torch.norm(x, dim=1, keepdim=True), torch.tensor(1e-9, device=x.device)))


class SRHGNLayer(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 node_dict, 
                 edge_dict, 
                 num_node_heads=4,
                 num_type_heads=4,
                 dropout=0.2, 
                 alpha=0.5, 
        ):
        super(SRHGNLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)

        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.num_node_heads = num_node_heads
        self.num_type_heads = num_type_heads

        # Node and edge transformations
        self.node_linear = nn.ModuleList()
        self.edge_linear = nn.ModuleList()
        
        # Attention modules
        self.src_attn = nn.ModuleList()
        self.dst_attn = nn.ModuleList()

        self.sem_attn_src = nn.ModuleList()
        self.sem_attn_dst = nn.ModuleList()
        self.rel_attn = nn.ModuleList()

        # Initialize linear transformations for each node type
        for _ in range(self.num_types):
            self.node_linear.append(nn.Linear(input_dim, output_dim))

        # Initialize transformations and attention for each relation type
        for _ in range(self.num_relations):
            self.edge_linear.append(nn.Linear(input_dim, output_dim))
            self.src_attn.append(nn.Linear(input_dim, num_node_heads))
            self.dst_attn.append(nn.Linear(input_dim, num_node_heads))

            self.sem_attn_src.append(nn.Linear(output_dim, num_type_heads))
            self.sem_attn_dst.append(nn.Linear(output_dim, num_type_heads))
            self.rel_attn.append(nn.Linear(output_dim, num_type_heads))

        # Learnable relation embeddings
        self.rel_emb = nn.Parameter(torch.randn(self.num_relations, output_dim), requires_grad=True)
        nn.init.xavier_normal_(self.rel_emb, gain=1.414)

        # Balance parameter between semantic and relation attention
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float), requires_grad=False)
        self.epsilon = nn.Parameter(torch.FloatTensor([1e-9]), requires_grad=False)

        self.drop = nn.Dropout(dropout)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict

            # Process each edge type in the heterogeneous graph
            for src, e, dst in G.canonical_etypes:
                # Extract subgraph for this specific relation
                sub_graph = G[src, e, dst]
                h_src = h[src]
                h_dst = h[dst]

                e_id = edge_dict[e]
                src_id = node_dict[src]
                dst_id = node_dict[dst]

                # Transform source nodes based on relation type
                h_src = self.drop(self.edge_linear[e_id](h_src))
                
                # Transform target nodes based on node type
                h_dst = self.drop(self.node_linear[dst_id](h_dst))

                # Calculate attention scores (similar to GAT)
                src_attn = self.drop(self.src_attn[e_id](h_src)).unsqueeze(-1)
                dst_attn = self.drop(self.dst_attn[e_id](h_dst)).unsqueeze(-1)

                # Combine attention scores on subgraph
                sub_graph.srcdata.update({'attn_src': src_attn})
                sub_graph.dstdata.update({'attn_dst': dst_attn})
                sub_graph.apply_edges(fn.u_add_v('attn_src', 'attn_dst', 'a'))
                a = F.leaky_relu(sub_graph.edata['a'])

                # Store node embeddings and normalized attention scores
                sub_graph.srcdata[f'v_{e_id}'] = h_src.view(
                    -1, self.num_node_heads, self.output_dim // self.num_node_heads)
                sub_graph.edata[f'a_{e_id}'] = self.drop(edge_softmax(sub_graph, a))

            # Aggregate embeddings across all relation types
            G.multi_update_all({etype: (fn.u_mul_e(f'v_{e_id}', f'a_{e_id}', 'm'), fn.sum('m', 'z')) \
                                for etype, e_id in edge_dict.items()}, cross_reducer='stack')

            z = {}
            attns = {}
            rel_idx_start = 0

            # Process each node type
            for ntype in G.ntypes:
                dst_id = node_dict[ntype]
                h_dst = h[ntype]

                # Check if aggregated data exists for this node type
                if 'z' not in G.nodes[ntype].data:
                    # If no incoming edges, use original node features
                    z_dst = self.drop(self.node_linear[dst_id](h_dst))
                    z[ntype] = normalize(F.gelu(z_dst + h[ntype]))
                    attns[ntype] = {'full': None, 'semantic': None, 'relation': None}
                    continue

                z_src = G.nodes[ntype].data['z']  # [N x R x H x (D // H)]
                num_nodes = z_src.shape[0]
                num_rel = z_src.shape[1]

                z_src = z_src.view(num_nodes, num_rel, self.output_dim)  # [N x R x D]
                z_dst = self.drop(self.node_linear[dst_id](h_dst))  # [N x D]

                # Initialize attention matrices
                sem_attn = torch.zeros(num_nodes, num_rel, self.num_type_heads, device=z_src.device)
                rel_attn = torch.zeros(num_nodes, num_rel, self.num_type_heads, device=z_src.device)

                # Compute semantic-aware and relation-aware attention scores
                for rel_idx in range(num_rel):
                    attn_idx = rel_idx_start + rel_idx
                    if attn_idx < len(self.sem_attn_src):  # Safety check
                        z_src_rel = z_src[:, rel_idx]

                        # Semantic attention based on node content
                        sem_attn_src = self.sem_attn_src[attn_idx](normalize(z_src_rel))
                        sem_attn_dst = self.sem_attn_dst[attn_idx](normalize(z_dst))
                        sem_attn[:, rel_idx] = sem_attn_src + sem_attn_dst

                        # Relation attention based on relation type
                        rel_attn[:, rel_idx] = self.rel_attn[attn_idx](
                            self.rel_emb[attn_idx].unsqueeze(0)).repeat(num_nodes, 1)

                rel_idx_start += num_rel

                # Normalize attention weights
                sem_attn = self.drop(F.softmax(F.leaky_relu(sem_attn), dim=1))
                rel_attn = self.drop(F.softmax(F.leaky_relu(rel_attn), dim=1))

                # Combine semantic and relation attention
                attn = self.alpha * sem_attn + (1 - self.alpha) * rel_attn

                # Apply multi-head attention to node embeddings
                z_dst = torch.mul(z_src.view(num_nodes, num_rel, self.num_type_heads, -1), 
                                attn.unsqueeze(-1))  # [N x R x H x (D // H)]
                
                # Concatenate all attention heads
                z_dst = z_dst.view(num_nodes, num_rel, self.output_dim)  # [N x R x D]

                # Aggregate across all relations and add residual connection
                z_dst = F.gelu(z_dst.sum(1) + h[ntype])

                z[ntype] = normalize(z_dst)

                # Store attention weights for analysis
                attns[ntype] = {
                    'full': attn.detach().cpu().numpy(),
                    'semantic': sem_attn.detach().cpu().numpy(),
                    'relation': rel_attn.detach().cpu().numpy()
                }
            
            return z, attns


class SRHGN(nn.Module):
    def __init__(self, 
                 G, 
                 node_dict, 
                 edge_dict, 
                 input_dims, 
                 hidden_dim, 
                 output_dim,
                 num_layers=2, 
                 num_node_heads=4, 
                 num_type_heads=4,
                 alpha=0.5
        ):
        super(SRHGN, self).__init__()

        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Pre-transformation layers for different node types
        self.pre_transform = nn.ModuleList()
        for ntype, idx in node_dict.items():
            self.pre_transform.append(nn.Linear(input_dims[ntype], hidden_dim))

        # SR-HGN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                SRHGNLayer(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    node_dict=node_dict,
                    edge_dict=edge_dict,
                    num_node_heads=num_node_heads,
                    num_type_heads=num_type_heads,
                    alpha=alpha
                ))

        # Output layer
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, G, target):
        h = {}
        attns = []

        # Pre-transformation for each node type
        for ntype in G.ntypes:
            n_id = self.node_dict[ntype]
            # Check if features exist
            if 'x' not in G.nodes[ntype].data:
                raise KeyError(f"Node type '{ntype}' missing features 'x' in graph data")
            h[ntype] = self.pre_transform[n_id](G.nodes[ntype].data['x'])
            h[ntype] = F.gelu(h[ntype])

        # Apply SR-HGN layers
        for conv in self.convs:
            h, attn = conv(G, h)
            attns.append(attn)

        # Generate final logits for target node type
        logits = self.out(h[target])

        return logits, h[target], attns


class LinkPredictor(nn.Module):
    """Link prediction decoder for heterogeneous graphs"""
    
    def __init__(self, hidden_dim, prediction_type="dot", dropout=0.2):
        super(LinkPredictor, self).__init__()
        self.prediction_type = prediction_type
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        
        if prediction_type == "mlp":
            self.predictor = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
        elif prediction_type == "bilinear":
            self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
            nn.init.xavier_uniform_(self.W)
        elif prediction_type == "complex":
            # Complex-valued embeddings for link prediction
            self.real_transform = nn.Linear(hidden_dim, hidden_dim // 2)
            self.imag_transform = nn.Linear(hidden_dim, hidden_dim // 2)
        elif prediction_type == "dist":
            # Distance-based prediction
            self.distance_metric = "cosine"  # or "euclidean"
    
    def forward(self, x_i, x_j):
        if self.prediction_type == "dot":
            # Dot product for link prediction
            return torch.sum(x_i * x_j, dim=1)
        elif self.prediction_type == "mlp":
            # Concatenation + MLP for link prediction
            x = torch.cat([x_i, x_j], dim=1)
            return self.predictor(x).squeeze(-1)
        elif self.prediction_type == "bilinear":
            # Bilinear transformation: x_i^T W x_j
            return torch.sum(x_i * torch.mm(x_j, self.W.t()), dim=1)
        elif self.prediction_type == "complex":
            # Complex embeddings
            real_i = self.real_transform(x_i)
            imag_i = self.imag_transform(x_i)
            real_j = self.real_transform(x_j)
            imag_j = self.imag_transform(x_j)
            
            # Complex multiplication
            real_part = torch.sum(real_i * real_j + imag_i * imag_j, dim=1)
            imag_part = torch.sum(real_i * imag_j - imag_i * real_j, dim=1)
            
            return torch.sqrt(real_part ** 2 + imag_part ** 2)
        elif self.prediction_type == "dist":
            # Distance-based similarity
            if self.distance_metric == "cosine":
                return F.cosine_similarity(x_i, x_j, dim=1)
            else:  # euclidean
                return -torch.norm(x_i - x_j, dim=1)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")


class LinkPredictionSRHGN(nn.Module):
    """SR-HGN model adapted for link prediction"""
    
    def __init__(self, 
                 G, 
                 node_dict, 
                 edge_dict, 
                 input_dims, 
                 hidden_dim,
                 num_layers=2, 
                 num_node_heads=4, 
                 num_type_heads=4,
                 alpha=0.5,
                 prediction_type="dot",
                 target_node_type="author",
                 dropout=0.2):
        super(LinkPredictionSRHGN, self).__init__()
        
        self.target_node_type = target_node_type
        
        # SR-HGN components for node embedding generation
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Pre-transformation layers
        self.pre_transform = nn.ModuleList()
        for ntype, idx in node_dict.items():
            self.pre_transform.append(nn.Linear(input_dims[ntype], hidden_dim))

        # SR-HGN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                SRHGNLayer(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    node_dict=node_dict,
                    edge_dict=edge_dict,
                    num_node_heads=num_node_heads,
                    num_type_heads=num_type_heads,
                    alpha=alpha,
                    dropout=dropout
                ))
        
        # Link prediction component
        self.link_predictor = LinkPredictor(hidden_dim, prediction_type, dropout)
    
    def generate_node_embeddings(self, G):
        """Generate node embeddings using SR-HGN architecture"""
        h = {}
        
        # Pre-transformation
        for ntype in G.ntypes:
            # Check if node type has features
            if 'x' not in G.nodes[ntype].data:
                raise KeyError(f"Node type '{ntype}' missing features 'x' in graph data. Available keys: {list(G.nodes[ntype].data.keys())}")
            
            n_id = self.node_dict[ntype]
            h[ntype] = self.pre_transform[n_id](G.nodes[ntype].data['x'])
            h[ntype] = F.gelu(h[ntype])

        # Apply SR-HGN layers
        for conv in self.convs:
            h, _ = conv(G, h)
        
        return h
    
    def forward(self, G, pos_graph=None, neg_graph=None):
        """
        Forward pass for link prediction
        
        Args:
            G: Full heterogeneous graph
            pos_graph: Subgraph with positive edges (optional)
            neg_graph: Subgraph with negative edges (optional)
        
        Returns:
            If pos_graph and neg_graph are provided: (pos_score, neg_score)
            Otherwise: node embeddings dict
        """
        # Generate node embeddings
        node_embeddings = self.generate_node_embeddings(G)
        
        if pos_graph is not None and neg_graph is not None:
            # Extract embeddings for target node type
            target_embeddings = node_embeddings[self.target_node_type]
            
            # Get positive predictions
            if pos_graph.num_edges() > 0:
                pos_src, pos_dst = pos_graph.edges()
                pos_src_embeds = target_embeddings[pos_src]
                pos_dst_embeds = target_embeddings[pos_dst]
                pos_score = self.link_predictor(pos_src_embeds, pos_dst_embeds)
            else:
                pos_score = torch.tensor([], device=target_embeddings.device)
            
            # Get negative predictions
            if neg_graph.num_edges() > 0:
                neg_src, neg_dst = neg_graph.edges()
                neg_src_embeds = target_embeddings[neg_src]
                neg_dst_embeds = target_embeddings[neg_dst]
                neg_score = self.link_predictor(neg_src_embeds, neg_dst_embeds)
            else:
                neg_score = torch.tensor([], device=target_embeddings.device)
            
            return pos_score, neg_score
        else:
            return node_embeddings


class HeterogeneousLinkPredictor(nn.Module):
    """
    Advanced link prediction model for multiple relation types in heterogeneous graphs
    """
    
    def __init__(self, 
                 G,
                 node_dict,
                 edge_dict,
                 input_dims,
                 hidden_dim,
                 relation_types,
                 num_layers=2,
                 num_node_heads=4,
                 num_type_heads=4,
                 alpha=0.5,
                 prediction_type="dot",
                 dropout=0.2):
        super(HeterogeneousLinkPredictor, self).__init__()
        
        self.relation_types = relation_types
        
        # Shared SR-HGN encoder
        self.encoder = LinkPredictionSRHGN(
            G, node_dict, edge_dict, input_dims, hidden_dim,
            num_layers, num_node_heads, num_type_heads, alpha,
            prediction_type="dot",  # Use simple dot product for encoder
            dropout=dropout
        )
        
        # Relation-specific link predictors
        self.relation_predictors = nn.ModuleDict()
        for rel_type in relation_types:
            rel_key = f"{rel_type[0]}_{rel_type[1]}_{rel_type[2]}"
            self.relation_predictors[rel_key] = LinkPredictor(hidden_dim, prediction_type, dropout)
    
    def forward(self, G, relation_type, pos_graph=None, neg_graph=None):
        """
        Forward pass for specific relation type
        
        Args:
            G: Full heterogeneous graph
            relation_type: Target relation type for prediction (src_type, edge_type, dst_type)
            pos_graph: Positive edges for this relation
            neg_graph: Negative edges for this relation
        """
        # Get node embeddings
        node_embeddings = self.encoder.generate_node_embeddings(G)
        
        # Get source and target node types for this relation
        src_type, edge_type, dst_type = relation_type
        rel_key = f"{src_type}_{edge_type}_{dst_type}"
        
        if pos_graph is not None and neg_graph is not None:
            # Extract relevant embeddings
            src_embeddings = node_embeddings[src_type]
            dst_embeddings = node_embeddings[dst_type]
            
            # Positive predictions
            if pos_graph.num_edges() > 0:
                pos_src, pos_dst = pos_graph.edges()
                pos_src_embeds = src_embeddings[pos_src]
                pos_dst_embeds = dst_embeddings[pos_dst]
                pos_score = self.relation_predictors[rel_key](pos_src_embeds, pos_dst_embeds)
            else:
                pos_score = torch.tensor([], device=src_embeddings.device)
            
            # Negative predictions
            if neg_graph.num_edges() > 0:
                neg_src, neg_dst = neg_graph.edges()
                neg_src_embeds = src_embeddings[neg_src]
                neg_dst_embeds = dst_embeddings[neg_dst]
                neg_score = self.relation_predictors[rel_key](neg_src_embeds, neg_dst_embeds)
            else:
                neg_score = torch.tensor([], device=src_embeddings.device)
            
            return pos_score, neg_score
        else:
            return node_embeddings


class MultiScaleSRHGN(nn.Module):
    """
    Multi-scale SR-HGN for handling graphs of different scales
    """
    def __init__(self, G, node_dict, edge_dict, input_dims, hidden_dim,
                 num_layers=2, num_node_heads=4, num_type_heads=4, alpha=0.5,
                 use_sampling=False, sampling_method='node'):
        super(MultiScaleSRHGN, self).__init__()
        
        self.use_sampling = use_sampling
        if use_sampling:
            self.sampler = GraphSAINT(G, sampling_method)
        
        # Base SR-HGN model
        self.srhgn = SRHGN(G, node_dict, edge_dict, input_dims, hidden_dim, hidden_dim,
                          num_layers, num_node_heads, num_type_heads, alpha)
    
    def forward(self, G, target, use_full_graph=True):
        """
        Forward pass with optional subgraph sampling
        """
        if self.use_sampling and not use_full_graph:
            # Use sampled subgraph for training
            subgraph = self.sampler.sample_subgraph()
            # Copy node features to subgraph
            for ntype in subgraph.ntypes:
                if 'x' in G.nodes[ntype].data:
                    subgraph.nodes[ntype].data['x'] = G.nodes[ntype].data['x'][subgraph.nodes[ntype].data[dgl.NID]]
            return self.srhgn(subgraph, target)
        else:
            # Use full graph
            return self.srhgn(G, target)


class GraphSAINT(nn.Module):
    """
    GraphSAINT-style sampling for large heterogeneous graphs
    """
    def __init__(self, G, sampling_method='node', num_walks=50, walk_length=2):
        super(GraphSAINT, self).__init__()
        self.G = G
        self.sampling_method = sampling_method
        self.num_walks = num_walks
        self.walk_length = walk_length
    
    def sample_subgraph(self):
        """Sample a subgraph for training"""
        if self.sampling_method == 'node':
            return self._node_sampling()
        elif self.sampling_method == 'edge':
            return self._edge_sampling()
        elif self.sampling_method == 'rw':
            return self._random_walk_sampling()
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")
    
    def _node_sampling(self):
        """Node-based sampling"""
        # Sample nodes for each type
        sampled_nodes = {}
        for ntype in self.G.ntypes:
            num_nodes = self.G.num_nodes(ntype)
            sample_size = min(num_nodes // 4, 1000)  # Sample 1/4 of nodes or max 1000
            sampled_nodes[ntype] = torch.randperm(num_nodes)[:sample_size]
        
        # Create subgraph
        subgraph = dgl.node_subgraph(self.G, sampled_nodes)
        return subgraph
    
    def _edge_sampling(self):
        """Edge-based sampling"""
        # Sample edges for each type
        edge_dict = {}
        for etype in self.G.canonical_etypes:
            num_edges = self.G.num_edges(etype)
            sample_size = min(num_edges // 4, 5000)  # Sample 1/4 of edges or max 5000
            edge_ids = torch.randperm(num_edges)[:sample_size]
            src, dst = self.G.find_edges(edge_ids, etype=etype)
            edge_dict[etype] = (src, dst)
        
        # Create subgraph
        subgraph = dgl.heterograph(edge_dict, 
                                  num_nodes_dict={ntype: self.G.num_nodes(ntype) 
                                                 for ntype in self.G.ntypes})
        return subgraph
    
    def _random_walk_sampling(self):
        """Random walk-based sampling"""
        # This is a simplified version
        return self._node_sampling()
