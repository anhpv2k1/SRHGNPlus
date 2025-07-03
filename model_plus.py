import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.functional import edge_softmax

def normalize(x):
    """L2 normalization"""
    return x / (torch.max(torch.norm(x, dim=1, keepdim=True), torch.tensor(1e-9, device=x.device)))

class SRHGNLayerPlus(nn.Module):
    def __init__(self, input_dim, output_dim, node_dict, edge_dict, num_node_heads=4, num_type_heads=4, dropout=0.2, alpha=0.5, num_metapaths=16, metapath_length=4):
        super(SRHGNLayerPlus, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.num_node_heads = num_node_heads
        self.num_type_heads = num_type_heads
        self.num_metapaths = num_metapaths
        self.metapath_length = metapath_length

        self.node_linear = nn.ModuleList()
        self.edge_linear = nn.ModuleList()
        self.src_attn = nn.ModuleList()
        self.dst_attn = nn.ModuleList()
        self.sem_attn_src = nn.ModuleList()
        self.sem_attn_dst = nn.ModuleList()
        self.rel_attn = nn.ModuleList()

        for _ in range(self.num_types):
            self.node_linear.append(nn.Linear(input_dim, output_dim))

        for _ in range(self.num_relations):
            self.edge_linear.append(nn.Linear(input_dim, output_dim))
            self.src_attn.append(nn.Linear(input_dim, num_node_heads))
            self.dst_attn.append(nn.Linear(input_dim, num_node_heads))
            self.sem_attn_src.append(nn.Linear(output_dim, num_type_heads))
            self.sem_attn_dst.append(nn.Linear(output_dim, num_type_heads))
            self.rel_attn.append(nn.Linear(output_dim, num_type_heads))

        self.rel_emb = nn.Parameter(torch.randn(self.num_relations, output_dim), requires_grad=True)
        nn.init.xavier_normal_(self.rel_emb, gain=1.414)

        self.film_mlp = nn.ModuleList()
        for _ in range(self.num_relations):
            self.film_mlp.append(nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.LeakyReLU(),
                nn.Linear(output_dim, output_dim * 2)
            ))

        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float), requires_grad=False)
        self.epsilon = nn.Parameter(torch.FloatTensor([1e-9]), requires_grad=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, G, h, metapath_dict=None):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict

            for src, e, dst in G.canonical_etypes:
                sub_graph = G[src, e, dst]
                h_src = h[src]
                h_dst = h[dst]
                e_id = edge_dict[e]
                src_id = node_dict[src]
                dst_id = node_dict[dst]

                h_src = self.drop(self.edge_linear[e_id](h_src))
                h_dst = self.drop(self.node_linear[dst_id](h_dst))

                src_attn = self.drop(self.src_attn[e_id](h_src)).unsqueeze(-1)
                dst_attn = self.drop(self.dst_attn[e_id](h_dst)).unsqueeze(-1)
                sub_graph.srcdata.update({'attn_src': src_attn})
                sub_graph.dstdata.update({'attn_dst': dst_attn})
                sub_graph.apply_edges(fn.u_add_v('attn_src', 'attn_dst', 'a'))
                a = F.leaky_relu(sub_graph.edata['a'])
                sub_graph.srcdata[f'v_{e_id}'] = h_src.view(-1, self.num_node_heads, self.output_dim // self.num_node_heads)
                sub_graph.edata[f'a_{e_id}'] = self.drop(edge_softmax(sub_graph, a))

            G.multi_update_all({etype: (fn.u_mul_e(f'v_{e_id}', f'a_{e_id}', 'm'), fn.sum('m', 'z')) for etype, e_id in edge_dict.items()}, cross_reducer='stack')

            z = {}
            attns = {}
            rel_idx_start = 0

            for ntype in G.ntypes:
                dst_id = node_dict[ntype]
                h_dst = h[ntype]

                if 'z' not in G.nodes[ntype].data:
                    z_dst = self.drop(self.node_linear[dst_id](h_dst))
                    z[ntype] = normalize(F.gelu(z_dst + h[ntype]))
                    attns[ntype] = {'full': None, 'semantic': None, 'relation': None}
                    continue

                z_src = G.nodes[ntype].data['z']
                num_nodes = z_src.shape[0]
                num_rel = z_src.shape[1]
                z_src = z_src.view(num_nodes, num_rel, self.output_dim)
                z_dst = self.drop(self.node_linear[dst_id](h_dst))

                sem_attn = torch.zeros(num_nodes, num_rel, self.num_type_heads, device=z_src.device)
                rel_attn = torch.zeros(num_nodes, num_rel, self.num_type_heads, device=z_src.device)

                if metapath_dict and ntype in metapath_dict:
                    for rel_idx in range(num_rel):
                        z_src_rel = z_src[:, rel_idx]
                        metapath_emb = torch.mean(z_src_rel, dim=0, keepdim=True)
                        film_params = self.film_mlp[rel_idx](metapath_emb)
                        gamma, beta = film_params[:, :self.output_dim], film_params[:, self.output_dim:]
                        z_src_rel = (gamma + 1) * z_src_rel + beta

                        attn_idx = rel_idx_start + rel_idx
                        sem_attn_src = self.sem_attn_src[attn_idx](normalize(z_src_rel))
                        sem_attn_dst = self.sem_attn_dst[attn_idx](normalize(z_dst))
                        sem_attn[:, rel_idx] = sem_attn_src + sem_attn_dst
                        rel_attn[:, rel_idx] = self.rel_attn[attn_idx](self.rel_emb[attn_idx].unsqueeze(0)).repeat(num_nodes, 1)
                else:
                    for rel_idx in range(num_rel):
                        attn_idx = rel_idx_start + rel_idx
                        z_src_rel = z_src[:, rel_idx]
                        sem_attn_src = self.sem_attn_src[attn_idx](normalize(z_src_rel))
                        sem_attn_dst = self.sem_attn_dst[attn_idx](normalize(z_dst))
                        sem_attn[:, rel_idx] = sem_attn_src + sem_attn_dst
                        rel_attn[:, rel_idx] = self.rel_attn[attn_idx](self.rel_emb[attn_idx].unsqueeze(0)).repeat(num_nodes, 1)

                rel_idx_start += num_rel

                sem_attn = self.drop(F.softmax(F.leaky_relu(sem_attn), dim=1))
                rel_attn = self.drop(F.softmax(F.leaky_relu(rel_attn), dim=1))
                attn = self.alpha * sem_attn + (1 - self.alpha) * rel_attn

                z_dst = torch.mul(z_src.view(num_nodes, num_rel, self.num_type_heads, -1), attn.unsqueeze(-1))
                z_dst = z_dst.view(num_nodes, num_rel, self.output_dim)
                z_dst = F.gelu(z_dst.sum(1) + h[ntype])
                z[ntype] = normalize(z_dst)

                attns[ntype] = {
                    'full': attn.detach().cpu().numpy(),
                    'semantic': sem_attn.detach().cpu().numpy(),
                    'relation': rel_attn.detach().cpu().numpy()
                }

            return z, attns

class LinkPredictionSRHGNPlus(nn.Module):
    def __init__(self, G, node_dict, edge_dict, input_dims, hidden_dim, num_layers=2, num_node_heads=4, num_type_heads=4, alpha=0.5, prediction_type="dot", target_node_type="author", dropout=0.2, global_gnn="graphsage"):
        super(LinkPredictionSRHGNPlus, self).__init__()
        self.target_node_type = target_node_type
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.global_gnn_type = global_gnn

        self.pre_transform = nn.ModuleList()
        for ntype, idx in node_dict.items():
            self.pre_transform.append(nn.Linear(input_dims[ntype], hidden_dim))

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SRHGNLayerPlus(
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                node_dict=node_dict,
                edge_dict=edge_dict,
                num_node_heads=num_node_heads,
                num_type_heads=num_type_heads,
                alpha=alpha,
                dropout=dropout
            ))

        if global_gnn == "graphsage":
            self.global_gnn = dgl.nn.SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean')
        else:
            self.global_gnn = dgl.nn.RelGraphConv(hidden_dim, hidden_dim, len(edge_dict), regularizer='basis', num_bases=4)

        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.link_predictor = LinkPredictor(hidden_dim, prediction_type, dropout)

    def generate_global_embeddings(self, G, h):
        if self.global_gnn_type == "graphsage":
            # Convert heterogeneous features to homogeneous
            h_homogeneous = []
            node_mapping = {}
            offset = 0
            for ntype in G.ntypes:
                num_nodes = h[ntype].shape[0]
                node_mapping[ntype] = (offset, offset + num_nodes)
                offset += num_nodes
                h_homogeneous.append(h[ntype])
            h_homogeneous = torch.cat(h_homogeneous, dim=0)

            # Create homogeneous graph
            src_list, dst_list = [], []
            for etype in G.canonical_etypes:
                src, dst = G.edges(etype=etype)
                src_ntype, _, dst_ntype = etype
                src = src + node_mapping[src_ntype][0]
                dst = dst + node_mapping[dst_ntype][0]
                src_list.append(src)
                dst_list.append(dst)
            if src_list and dst_list:
                src = torch.cat(src_list)
                dst = torch.cat(dst_list)
                g_homogeneous = dgl.graph((src, dst), num_nodes=h_homogeneous.shape[0])
            else:
                g_homogeneous = dgl.graph(([], []), num_nodes=h_homogeneous.shape[0])

            # Apply GraphSAGE
            h_global = self.global_gnn(g_homogeneous, h_homogeneous)
            
            # Convert back to heterogeneous dictionary
            h_global_dict = {}
            for ntype in G.ntypes:
                start, end = node_mapping[ntype]
                h_global_dict[ntype] = h_global[start:end]
            return h_global_dict
        else:
            # R-GCN handles heterogeneous graph directly
            edge_type = G.edata['id'] if 'id' in G.edata else torch.zeros(G.num_edges(), dtype=torch.long, device=G.device)
            h_global = self.global_gnn(G, h, edge_type)
            return h_global

    def generate_node_embeddings(self, G, metapath_dict=None):
        h = {}
        for ntype in G.ntypes:
            n_id = self.node_dict[ntype]
            if 'x' not in G.nodes[ntype].data:
                raise KeyError(f"Node type '{ntype}' missing features 'x' in graph data")
            h[ntype] = self.pre_transform[n_id](G.nodes[ntype].data['x'])
            h[ntype] = F.gelu(h[ntype])

        for conv in self.convs:
            h, _ = conv(G, h, metapath_dict)

        # Local-Global Fusion
        h_global = self.generate_global_embeddings(G, h)
        for ntype in G.ntypes:
            h_local = h[ntype]
            h_global_ntype = h_global[ntype] if isinstance(h_global, dict) else h_global[G.nodes[ntype].data[dgl.NID]]
            h_fused = torch.cat([h_local, h_global_ntype], dim=1)
            h_fused = self.fusion(h_fused)
            h_fused = F.relu(h_fused)
            h_fused = self.drop(h_fused)  # Apply dropout to tensor
            h[ntype] = self.layer_norm(h_fused)

        return h

    def forward(self, G, pos_graph=None, neg_graph=None, metapath_dict=None):
        node_embeddings = self.generate_node_embeddings(G, metapath_dict)
        if pos_graph is not None and neg_graph is not None:
            target_embeddings = node_embeddings[self.target_node_type]
            if pos_graph.num_edges() > 0:
                pos_src, pos_dst = pos_graph.edges()
                pos_src_embeds = target_embeddings[pos_src]
                pos_dst_embeds = target_embeddings[pos_dst]
                pos_score = self.link_predictor(pos_src_embeds, pos_dst_embeds)
            else:
                pos_score = torch.tensor([], device=target_embeddings.device)
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

class LinkPredictor(nn.Module):
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
            self.real_transform = nn.Linear(hidden_dim, hidden_dim // 2)
            self.imag_transform = nn.Linear(hidden_dim, hidden_dim // 2)
        elif prediction_type == "dist":
            self.distance_metric = "cosine"

    def forward(self, x_i, x_j):
        if self.prediction_type == "dot":
            return torch.sum(x_i * x_j, dim=1)
        elif self.prediction_type == "mlp":
            x = torch.cat([x_i, x_j], dim=1)
            return self.predictor(x).squeeze(-1)
        elif self.prediction_type == "bilinear":
            return torch.sum(x_i * torch.mm(x_j, self.W.t()), dim=1)
        elif self.prediction_type == "complex":
            real_i = self.real_transform(x_i)
            imag_i = self.imag_transform(x_i)
            real_j = self.real_transform(x_j)
            imag_j = self.imag_transform(x_j)
            real_part = torch.sum(real_i * real_j + imag_i * imag_j, dim=1)
            imag_part = torch.sum(real_i * imag_j - imag_i * real_j, dim=1)
            return torch.sqrt(real_part ** 2 + imag_part ** 2)
        elif self.prediction_type == "dist":
            if self.distance_metric == "cosine":
                return F.cosine_similarity(x_i, x_j, dim=1)
            else:
                return -torch.norm(x_i - x_j, dim=1)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
