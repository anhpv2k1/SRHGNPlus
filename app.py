import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import dgl
import time
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
import networkx as nx
import plotly.graph_objects as go
from itertools import combinations

warnings.filterwarnings('ignore')

# ============================================================================  
# IMPORT MODEL FILES  
# ============================================================================  
try:
    from model_plus import LinkPredictionSRHGNPlus
    from utils import (
        load_data, set_random_seed, create_train_val_test_splits,
        extract_coauthorship_network, get_target_node_type,
        create_negative_samples, generate_metapaths
    )
    MODEL_FILES_AVAILABLE = True
except ImportError as e:
    MODEL_FILES_AVAILABLE = False
    st.error(f"❌ Lỗi import model files: {e}")
    st.error("Hãy đảm bảo files model_plus.py và utils.py có trong thư mục!")
    st.stop()

# ============================================================================  
# PAGE CONFIGURATION  
# ============================================================================  
st.set_page_config(
    page_title="SR-HGN Plus Link Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================  
# LOGGING SETUP  
# ============================================================================  
def setup_logging():
    """Thiết lập logging system"""
    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"sr_hgn_plus_streamlit_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('SR-HGN-Plus')
    return logger, log_file

# ============================================================================  
# CUSTOM CSS  
# ============================================================================  
st.markdown("""
<style>
    .stApp { background-color: #0e1117 !important; color: #ffffff !important; }
    .main .block-container { background-color: #0e1117 !important; color: #ffffff !important; }
    .main-title { font-size: 2.5rem; font-weight: bold; text-align: center; color: #1f77b4 !important; margin-bottom: 1rem; }
    .sub-title { font-size: 1.2rem; text-align: center; color: #cccccc !important; margin-bottom: 2rem; }
    .metric-container { background: #1f77b4 !important; padding: 1rem; border-radius: 10px; color: black !important; text-align: center; margin: 0.5rem 0; }
    .info-box { background: #1a2332 !important; padding: 1rem; border-radius: 8px; border-left: 4px solid #1f77b4; margin: 1rem 0; color: #ffffff !important; font-weight: 500; }
    .success-box { background: #1a2e1a !important; padding: 1rem; border-radius: 8px; border-left: 4px solid #28a745; margin: 1rem 0; color: #ffffff !important; font-weight: 500; }
    .error-box { background: #2e1a1a !important; padding: 1rem; border-radius: 8px; border-left: 4px solid #dc3545; margin: 1rem 0; color: #ffffff !important; font-weight: 500; }
    .dataset-info { background: #1a2332 !important; padding: 1.5rem; border-radius: 10px; border: 2px solid #1f77b4; margin: 1rem 0; font-family: 'Courier New', monospace; color: #ffffff !important; font-size: 0.95rem; line-height: 1.6; }
    .training-log { background: #1a1a1a !important; padding: 1rem; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 0.85rem; max-height: 400px; overflow-y: auto; border: 2px solid #333333; color: #ffffff !important; white-space: pre-wrap; line-height: 1.4; }
    .results-popup { background: #1a2332 !important; border: 3px solid #1f77b4; border-radius: 15px; padding: 2rem; margin: 1.5rem 0; text-align: center; font-family: 'Courier New', monospace; font-size: 1.1rem; box-shadow: 0 8px 16px rgba(31,119,180,0.3); color: #ffffff !important; font-weight: bold; }
    .prediction-result { background: #1f77b4 !important; border-radius: 15px; padding: 2rem; margin: 1.5rem 0; text-align: center; color: black !important; font-size: 1.2rem; box-shadow: 0 8px 16px rgba(31,119,180,0.3); }
    h1, h2, h3, h4, h5, h6 { color: #1f77b4 !important; }
    .stMarkdown { color: #ffffff !important; }
    .stTextInput > div > div > input { color: #ffffff !important; background-color: #262730 !important; border: 1px solid #333333 !important; }
    .stSelectbox > div > div > div { color: #ffffff !important; background-color: #262730 !important; border: 1px solid #333333 !important; }
    .stSlider > div > div > div > div { color: #ffffff !important; }
    .css-1d391kg { background-color: #1a1a1a !important; }
    .css-1d391kg .stMarkdown { color: #ffffff !important; }
    .stButton > button { color: #ffffff !important; background-color: #1f77b4 !important; border: none; border-radius: 5px; font-weight: bold; border: 1px solid #1f77b4 !important; }
    .stButton > button:hover { background-color: #1565c0 !important; border: 1px solid #1565c0 !important; }
    .stApp > div > div > div > div { color: #ffffff !important; }
    .stProgress > div > div > div > div { background-color: #1f77b4 !important; }
    [data-testid="metric-container"] { background-color: #1a2332 !important; border: 1px solid #333333 !important; border-radius: 8px; color: #ffffff !important; }
    .metric-container h2, .metric-container h4 { color: black !important; margin: 0.25rem 0; }
</style>
""", unsafe_allow_html=True)

# ============================================================================  
# SESSION STATE INITIALIZATION  
# ============================================================================  
def init_session_state():
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'graph' not in st.session_state:
        st.session_state.graph = None
    if 'node_dict' not in st.session_state:
        st.session_state.node_dict = None
    if 'edge_dict' not in st.session_state:
        st.session_state.edge_dict = None
    if 'features' not in st.session_state:
        st.session_state.features = None
    if 'target_node_type' not in st.session_state:
        st.session_state.target_node_type = None
    if 'dataset_name' not in st.session_state:
        st.session_state.dataset_name = None
    if 'dataset_loaded' not in st.session_state:
        st.session_state.dataset_loaded = False
    if 'training_history' not in st.session_state:
        st.session_state.training_history = {'epoch': [], 'loss': [], 'val_auc': [], 'val_ap': [], 'val_f1': []}
    if 'training_logs' not in st.session_state:
        st.session_state.training_logs = []
    if 'is_training' not in st.session_state:
        st.session_state.is_training = False
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None
    if 'logger' not in st.session_state:
        st.session_state.logger, st.session_state.log_file = setup_logging()

# ============================================================================  
# UTILITY FUNCTIONS  
# ============================================================================  
def show_message(message, type="info"):
    if type == "success":
        st.markdown(f'<div class="success-box">✅ {message}</div>', unsafe_allow_html=True)
    elif type == "error":
        st.markdown(f'<div class="error-box">❌ {message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="info-box">ℹ️ {message}</div>', unsafe_allow_html=True)

def create_directories():
    for directory in ['data', 'log', 'checkpoint', 'results']:
        Path(directory).mkdir(exist_ok=True)

def add_training_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - INFO - {message}"
    st.session_state.training_logs.append(log_entry)
    st.session_state.logger.info(message)

def visualize_subgraph(collab_network, target_node, max_nodes):
    """Visualize subgraph of 1st and 2nd degree neighbors using Plotly"""
    try:
        neighbors_1 = set(collab_network.neighbors(target_node)) if target_node in collab_network.nodes else set()
        neighbors_2 = set()
        for n1 in neighbors_1:
            neighbors_2.update(collab_network.neighbors(n1))
        neighbors_2.discard(target_node)
        neighbors_2.difference_update(neighbors_1)
        max_display_nodes = min(50, max_nodes)
        if len(neighbors_1) + len(neighbors_2) > max_display_nodes:
            neighbors_1 = set(list(neighbors_1)[:max_display_nodes//2])
            neighbors_2 = set(list(neighbors_2)[:max_display_nodes//2])
        nodes = [target_node] + list(neighbors_1) + list(neighbors_2)
        subgraph = nx.subgraph(collab_network, nodes)
        edge_x = []
        edge_y = []
        edge_text = []
        pos = nx.spring_layout(subgraph, seed=42)
        for edge in subgraph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(f"Weight: {edge[2].get('weight', 1)}")
        node_x = [pos[node][0] for node in subgraph.nodes]
        node_y = [pos[node][1] for node in subgraph.nodes]
        node_text = [f"Node {node}" for node in subgraph.nodes]
        node_colors = []
        for node in subgraph.nodes:
            if node == target_node:
                node_colors.append('#ff6b6b')  # Target node in red
            elif node in neighbors_1:
                node_colors.append('#4ecdc4')  # 1st degree in cyan
            else:
                node_colors.append('#f9ca24')  # 2nd degree in yellow
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color='#888888'),
            hoverinfo='text',
            text=edge_text,
            mode='lines'
        )
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition='top center',
            marker=dict(
                showscale=False,
                color=node_colors,
                size=15,
                line_width=2
            )
        )
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=dict(
                                text=f'Subgraph of Node {target_node} (1st & 2nd Degree Neighbors)',
                                font=dict(size=16, color='#1f77b4'),
                                x=0.5,
                                xanchor='center'
                            ),
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            plot_bgcolor='#0e1117',
                            paper_bgcolor='#0e1117',
                            font=dict(color='#ffffff'),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))
        return fig
    except Exception as e:
        add_training_log(f"Error in visualizing subgraph: {e}")
        return None

# ============================================================================  
# DATA LOADING FUNCTIONS  
# ============================================================================  
def load_dataset(dataset_name):
    try:
        if not MODEL_FILES_AVAILABLE:
            raise ImportError("Model files not available")
        set_random_seed(42)
        add_training_log(f"Training SR-HGN Plus for link prediction on {dataset_name}")
        g, node_dict, edge_dict, features, labels, num_classes, \
        train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, target = \
            load_data(dataset_name, 0.8, 0.1, feat=1)
        st.session_state.graph = g
        st.session_state.node_dict = node_dict
        st.session_state.edge_dict = edge_dict
        st.session_state.features = features
        real_target = get_target_node_type(dataset_name)
        st.session_state.target_node_type = real_target
        add_training_log(f"Target node type set to: {real_target}")
        st.session_state.dataset_name = dataset_name
        st.session_state.dataset_loaded = True
        metapath_dict = generate_metapaths(g, num_metapaths=16, metapath_length=4)
        st.session_state.metapath_dict = metapath_dict
        add_training_log(f"Generated metapath dictionary: {metapath_dict}")
        add_training_log(f"Loaded {dataset_name} dataset successfully (target: {target})")
        num_nodes_dict = {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
        num_edges_dict = {etype: g.num_edges(etype) for etype in g.canonical_etypes}
        graph_info = (
            f"Graph(num_nodes={num_nodes_dict}, "
            f"num_edges={num_edges_dict}, metagraph={list(g.canonical_etypes)})"
        )
        add_training_log(f"Graph info: {graph_info}")
        add_training_log(f"Node types: {g.ntypes}")
        add_training_log(f"Edge types: {g.etypes}")
        add_training_log(f"Initial features keys: {list(features.keys())}")
        return True, "Dataset loaded successfully!"
    except FileNotFoundError as e:
        return False, f"Không tìm thấy file dữ liệu: {e}"
    except Exception as e:
        return False, f"Lỗi load dataset: {e}"

def display_dataset_info(g, features, dataset_name):
    num_nodes_dict = {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
    num_edges_dict = {etype: g.num_edges(etype) for etype in g.canonical_etypes}
    dataset_info = f"""
<span style="font-size: 1.2rem; color: #1f77b4; font-weight: bold;">{dataset_name.upper()} Graph Info</span>

<span style="color: #4ecdc4; font-weight: bold;">Graph:</span> <span style="color: #ffffff;">Graph(num_nodes={num_nodes_dict}, num_edges={num_edges_dict}, metagraph={list(g.canonical_etypes)})</span>

<span style="color: #4ecdc4; font-weight: bold;">Node types:</span> <span style="color: #ffffff;">{g.ntypes}</span>

<span style="color: #4ecdc4; font-weight: bold;">Edge types:</span> <span style="color: #ffffff;">{g.etypes}</span>

<span style="color: #4ecdc4; font-weight: bold;">Initial features keys:</span> <span style="color: #ffffff;">{list(features.keys())}</span>

<span style="color: #4ecdc4; font-weight: bold;">Total nodes:</span> <span style="color: #f9ca24; font-weight: bold;">{g.num_nodes():,}</span>

<span style="color: #4ecdc4; font-weight: bold;">Total edges:</span> <span style="color: #45b7d1; font-weight: bold;">{g.num_edges():,}</span>
"""
    st.markdown(f'<div class="dataset-info">{dataset_info}</div>', unsafe_allow_html=True)

def prepare_features(g, features, dataset_name):
    input_dims = {}
    for ntype in g.ntypes:
        if ntype in features:
            input_dims[ntype] = features[ntype].shape[1]
            g.nodes[ntype].data['x'] = features[ntype]
            add_training_log(f"{ntype} has features with dim {input_dims[ntype]}")
        else:
            if dataset_name == 'acm':
                if ntype == 'author':
                    feat_dim = 128
                elif ntype == 'subject':
                    feat_dim = 64
                else:
                    feat_dim = 100
            elif dataset_name == 'dblp':
                feat_dim = 128
            elif dataset_name == 'imdb':
                feat_dim = 128
            else:
                feat_dim = 128
            g.nodes[ntype].data['x'] = torch.randn(g.num_nodes(ntype), feat_dim)
            input_dims[ntype] = feat_dim
            add_training_log(f"{ntype} missing features, using default dim {feat_dim}")
    add_training_log(f"Final input dimensions: {input_dims}")
    for ntype in g.ntypes:
        if 'x' in g.nodes[ntype].data:
            add_training_log(f"Verified: {ntype} has features shape {g.nodes[ntype].data['x'].shape}")
    return input_dims

def create_realtime_plot():
    history = st.session_state.training_history
    if not history['epoch']:
        return None
    filtered_epochs = []
    filtered_loss = []
    filtered_val_auc = []
    filtered_val_ap = []
    filtered_val_f1 = []
    for i, epoch in enumerate(history['epoch']):
        if epoch % 10 == 0:
            filtered_epochs.append(epoch)
            filtered_loss.append(history['loss'][i])
            filtered_val_auc.append(history['val_auc'][i])
            filtered_val_ap.append(history['val_ap'][i])
            filtered_val_f1.append(history['val_f1'][i])
    if not filtered_epochs:
        return None
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Training Loss', 'Validation AUC', 'Validation AP', 'Validation F1'),
        horizontal_spacing=0.15, vertical_spacing=0.15
    )
    fig.add_trace(
        go.Scatter(x=filtered_epochs, y=filtered_loss, mode='lines+markers', name='Loss',
                   line=dict(color='#ff6b6b', width=3), marker=dict(size=8, color='#ff6b6b', symbol='circle')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=filtered_epochs, y=filtered_val_auc, mode='lines+markers', name='Val AUC',
                   line=dict(color='#4ecdc4', width=3), marker=dict(size=8, color='#4ecdc4', symbol='circle')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=filtered_epochs, y=filtered_val_ap, mode='lines+markers', name='Val AP',
                   line=dict(color='#45b7d1', width=3), marker=dict(size=8, color='#45b7d1', symbol='circle')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=filtered_epochs, y=filtered_val_f1, mode='lines+markers', name='Val F1',
                   line=dict(color='#f9ca24', width=3), marker=dict(size=8, color='#f9ca24', symbol='circle')),
        row=2, col=2
    )
    fig.update_layout(
        height=500, 
        title=dict(
            text="🔄 Training Progress (Every 10 Epochs)",
            font=dict(size=20, color='#1f77b4'),
            x=0.5,
            xanchor='center'
        ),
        showlegend=False, 
        plot_bgcolor='#0e1117', 
        paper_bgcolor='#0e1117', 
        font=dict(color='#ffffff'),
        margin=dict(t=60, b=40, l=40, r=40)
    )
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color='#1f77b4', size=14)
    for axis in fig.layout:
        if 'xaxis' in axis or 'yaxis' in axis:
            fig.layout[axis].update(title_font=dict(color='#ffffff'), tickfont=dict(color='#ffffff'), gridcolor='#333333')
    return fig

# ============================================================================  
# MODEL TRAINING FUNCTIONS  
# ============================================================================  
def evaluate_link_prediction(model, g, pos_edges, neg_edges, device, node_type='author', metapath_dict=None):
    model.eval()
    with torch.no_grad():
        try:
            node_embeddings = model.generate_node_embeddings(g, metapath_dict)
            if node_type not in node_embeddings:
                return {'auc': 0.0, 'ap': 0.0, 'f1': 0.0, 'mrr': 0.0, 'hit@10': 0.0}
            target_embeddings = node_embeddings[node_type]
            pos_scores, neg_scores = [], []
            if pos_edges:
                for src, dst in pos_edges:
                    if src < target_embeddings.shape[0] and dst < target_embeddings.shape[0]:
                        src_emb = target_embeddings[src].unsqueeze(0)
                        dst_emb = target_embeddings[dst].unsqueeze(0)
                        score = model.link_predictor(src_emb, dst_emb)
                        pos_scores.append(score.item())
            if neg_edges:
                for src, dst in neg_edges:
                    if src < target_embeddings.shape[0] and dst < target_embeddings.shape[0]:
                        src_emb = target_embeddings[src].unsqueeze(0)
                        dst_emb = target_embeddings[dst].unsqueeze(0)
                        score = model.link_predictor(src_emb, dst_emb)
                        neg_scores.append(score.item())
            if pos_scores and neg_scores:
                from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
                all_scores = pos_scores + neg_scores
                all_labels = [1]*len(pos_scores) + [0]*len(neg_scores)
                auc_score = roc_auc_score(all_labels, all_scores)
                ap_score = average_precision_score(all_labels, all_scores)
                precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                best_f1 = np.max(f1_scores)
                pos_arr = np.array(pos_scores)
                neg_arr = np.array(neg_scores)
                reciprocal_ranks = []
                for ps in pos_scores:
                    rank = 1 + np.sum(neg_arr > ps)
                    reciprocal_ranks.append(1.0/rank)
                mrr = np.mean(reciprocal_ranks)
                hit_10 = 0
                for ps in pos_scores:
                    rank = 1 + np.sum(neg_arr > ps)
                    if rank <= 10:
                        hit_10 += 1
                hit_10 = hit_10/len(pos_scores) if pos_scores else 0
                return {'auc': auc_score, 'ap': ap_score, 'f1': best_f1, 'mrr': mrr, 'hit@10': hit_10}
            else:
                return {'auc': 0.0, 'ap': 0.0, 'f1': 0.0, 'mrr': 0.0, 'hit@10': 0.0}
        except Exception as e:
            add_training_log(f"Error in evaluation: {e}")
            return {'auc': 0.0, 'ap': 0.0, 'f1': 0.0, 'mrr': 0.0, 'hit@10': 0.0}

def train_model(config):
    try:
        st.session_state.training_history = {'epoch': [], 'loss': [], 'val_auc': [], 'val_ap': [], 'val_f1': []}
        st.session_state.training_logs = []
        if st.session_state.graph is None:
            return False, "Vui lòng load dataset trước!"
        g = st.session_state.graph
        node_dict = st.session_state.node_dict
        edge_dict = st.session_state.edge_dict
        features = st.session_state.features
        dataset_name = st.session_state.dataset_name
        target_node_type = st.session_state.target_node_type
        metapath_dict = st.session_state.metapath_dict
        input_dims = prepare_features(g, features, dataset_name)
        add_training_log(f"Creating collaboration network from {dataset_name} data...")
        collab_network = extract_coauthorship_network(g, dataset_name)
        if collab_network.number_of_edges() == 0:
            return False, "Không tìm thấy collaboration edges!"
        add_training_log(f"Collaboration network: {collab_network.number_of_nodes()} nodes, {collab_network.number_of_edges()} edges")
        train_edges, val_edges, test_edges, train_neg, val_neg, test_neg = create_train_val_test_splits(
            collab_network,
            test_ratio=config['test_ratio'],
            val_ratio=config['val_ratio'],
            negative_sampling=config['neg_sampling'],
            neg_sample_ratio=config['neg_ratio']
        )
        add_training_log(f"Train edges: {len(train_edges)}, Val edges: {len(val_edges)}, Test edges: {len(test_edges)}")
        add_training_log(f"Train neg: {len(train_neg)}, Val neg: {len(val_neg)}, Test neg: {len(test_neg)}")
        add_training_log(f"Target node type: {target_node_type}")
        model = LinkPredictionSRHGNPlus(
            G=g,
            node_dict=node_dict,
            edge_dict=edge_dict,
            input_dims=input_dims,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_node_heads=config['num_node_heads'],
            num_type_heads=config['num_type_heads'],
            alpha=config['alpha'],
            prediction_type=config['prediction_type'],
            target_node_type=target_node_type,
            dropout=config['dropout'],
            global_gnn=config['global_gnn']
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        g = g.to(device)
        add_training_log(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        add_training_log(f"Using device: {device}")
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        if config['scheduler'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
        elif config['scheduler'] == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_container = st.empty()
        st.session_state.is_training = True
        best_val_auc, best_val_ap, early_stop_counter, best_epoch = 0, 0, 0, 0
        for epoch in range(config['num_epochs']):
            if not st.session_state.is_training:
                break
            model.train()
            pos_graph = dgl.graph(([], []), num_nodes=collab_network.number_of_nodes()).to(device)
            neg_graph = dgl.graph(([], []), num_nodes=collab_network.number_of_nodes()).to(device)
            if train_edges:
                src, dst = zip(*train_edges)
                pos_graph = dgl.graph((src, dst), num_nodes=collab_network.number_of_nodes()).to(device)
            if train_neg:
                srcn, dstn = zip(*train_neg)
                neg_graph = dgl.graph((srcn, dstn), num_nodes=collab_network.number_of_nodes()).to(device)
            try:
                pos_scores, neg_scores = model(g, pos_graph, neg_graph, metapath_dict)
            except Exception as e:
                st.session_state.is_training = False
                add_training_log(f"Error during forward pass: {str(e)}")
                return False, f"Lỗi training: {str(e)}"
            pos_labels = torch.ones(len(pos_scores), device=device) if len(pos_scores)>0 else torch.tensor([], device=device)
            neg_labels = torch.zeros(len(neg_scores), device=device) if len(neg_scores)>0 else torch.tensor([], device=device)
            if pos_scores.numel()>0 and neg_scores.numel()>0:
                all_scores = torch.cat([pos_scores, neg_scores])
                all_labels = torch.cat([pos_labels, neg_labels])
                loss = F.binary_cross_entropy_with_logits(all_scores, all_labels)
                optimizer.zero_grad()
                loss.backward()
                if config['clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
                optimizer.step()
                if config['scheduler'] in ['cosine', 'step']:
                    scheduler.step()
            else:
                loss = torch.tensor(0.0)
            st.session_state.training_history['epoch'].append(epoch)
            st.session_state.training_history['loss'].append(loss.item())
            if epoch % config['eval_every'] == 0:
                val_metrics = evaluate_link_prediction(model, g, val_edges, val_neg, device, target_node_type, metapath_dict)
                st.session_state.training_history['val_auc'].append(val_metrics['auc'])
                st.session_state.training_history['val_ap'].append(val_metrics['ap'])
                st.session_state.training_history['val_f1'].append(val_metrics['f1'])
                add_training_log(f"Epoch {epoch:04d} | Loss: {loss.item():.4f} | Val AUC: {val_metrics['auc']:.4f} | Val AP: {val_metrics['ap']:.4f} | Val F1: {val_metrics['f1']:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
                current_score = val_metrics['auc'] + val_metrics['ap']
                best_score = best_val_auc + best_val_ap
                if current_score > best_score:
                    best_val_auc = val_metrics['auc']
                    best_val_ap = val_metrics['ap']
                    early_stop_counter = 0
                    best_epoch = epoch
                    add_training_log(f"New best model at epoch {epoch}")
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= config['early_stop']:
                        add_training_log(f"Early stopping after {config['early_stop']} epochs without improvement")
                        break
            else:
                if st.session_state.training_history['val_auc']:
                    st.session_state.training_history['val_auc'].append(st.session_state.training_history['val_auc'][-1])
                    st.session_state.training_history['val_ap'].append(st.session_state.training_history['val_ap'][-1])
                    st.session_state.training_history['val_f1'].append(st.session_state.training_history['val_f1'][-1])
                else:
                    st.session_state.training_history['val_auc'].append(0.0)
                    st.session_state.training_history['val_ap'].append(0.0)
                    st.session_state.training_history['val_f1'].append(0.0)
            progress = (epoch + 1) / config['num_epochs']
            progress_bar.progress(progress)
            current_val_auc = st.session_state.training_history['val_auc'][-1]
            status_text.text(f'Epoch {epoch+1}/{config["num_epochs"]} - Loss: {loss.item():.4f} - Val AUC: {current_val_auc:.4f} - Progress: {progress*100:.1f}%')
            if epoch % 10 == 0:
                fig = create_realtime_plot()
                if fig:
                    chart_container.plotly_chart(fig, use_container_width=True)
        st.session_state.model = model
        st.session_state.test_edges = test_edges
        st.session_state.test_neg = test_neg
        st.session_state.is_training = False
        add_training_log(f"Training completed up to epoch {best_epoch}")
        return True, "Training completed successfully!"
    except Exception as e:
        st.session_state.is_training = False
        add_training_log(f"Error during training: {str(e)}")
        return False, f"Lỗi training: {str(e)}"

def evaluate_model():
    try:
        if st.session_state.model is None:
            return None
        model = st.session_state.model
        g = st.session_state.graph.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        test_edges = st.session_state.test_edges
        test_neg = st.session_state.test_neg
        target_node_type = st.session_state.target_node_type
        metapath_dict = st.session_state.metapath_dict
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_metrics = evaluate_link_prediction(model, g, test_edges, test_neg, device, target_node_type, metapath_dict)
        add_training_log("="*50)
        add_training_log("Final Test Results:")
        add_training_log(f"  AUC: {test_metrics['auc']:.4f}")
        add_training_log(f"  AP: {test_metrics['ap']:.4f}")
        add_training_log(f"  F1: {test_metrics['f1']:.4f}")
        add_training_log(f"  MRR: {test_metrics['mrr']:.4f}")
        add_training_log(f"  Hit@10: {test_metrics['hit@10']:.4f}")
        add_training_log("="*50)
        return test_metrics
    except Exception as e:
        add_training_log(f"Error in evaluation: {e}")
        return None

def show_results_popup(results):
    results_text = f"""
**KẾT QUẢ CUỐI CÙNG**

AUC: {results['auc']:.4f}
AP: {results['ap']:.4f}
F1: {results['f1']:.4f}
MRR: {results['mrr']:.4f}
Hit@10: {results['hit@10']:.4f}
"""
    st.markdown(f'<div class="results-popup">{results_text}</div>', unsafe_allow_html=True)

def predict_link_probability(node1_id, node2_id):
    try:
        if st.session_state.model is None:
            return None, "Model chưa được training!"
        model = st.session_state.model
        g = st.session_state.graph.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        target_node_type = st.session_state.target_node_type
        metapath_dict = st.session_state.metapath_dict
        max_nodes = g.num_nodes(target_node_type)
        if node1_id < 0 or node1_id >= max_nodes or node2_id < 0 or node2_id >= max_nodes:
            return None, f"ID nút phải trong khoảng 0-{max_nodes-1}"
        if node1_id == node2_id:
            return None, "Hai node phải khác nhau!"
        model.eval()
        with torch.no_grad():
            node_embeddings = model.generate_node_embeddings(g, metapath_dict)
            target_embeddings = node_embeddings[target_node_type]
            src_emb = target_embeddings[node1_id].unsqueeze(0)
            dst_emb = target_embeddings[node2_id].unsqueeze(0)
            score = model.link_predictor(src_emb, dst_emb)
            probability = torch.sigmoid(score).item()
            return probability, None
    except Exception as e:
        return None, f"Lỗi dự đoán: {e}"

def predict_multiple_nodes(node_ids):
    try:
        if st.session_state.model is None:
            return None, "Model chưa được training!"
        model = st.session_state.model
        g = st.session_state.graph.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        target_node_type = st.session_state.target_node_type
        metapath_dict = st.session_state.metapath_dict
        max_nodes = g.num_nodes(target_node_type)
        node_ids = [id for id in node_ids if 0 <= id < max_nodes]
        if len(node_ids) < 2:
            return None, "Cần ít nhất 2 ID nút hợp lệ!"
        if len(set(node_ids)) != len(node_ids):
            return None, "Các ID nút phải khác nhau!"
        model.eval()
        results = []
        with torch.no_grad():
            node_embeddings = model.generate_node_embeddings(g, metapath_dict)
            H = node_embeddings[target_node_type]
            for u, v in combinations(node_ids, 2):
                src_emb = H[u].unsqueeze(0)
                dst_emb = H[v].unsqueeze(0)
                score = model.link_predictor(src_emb, dst_emb)
                prob = torch.sigmoid(score).item()
                prediction = "Có liên kết" if prob > 0.5 else "Không"
                results.append((u, v, prob, prediction))
        return results, None
    except Exception as e:
        return None, f"Lỗi dự đoán: {e}"

# ============================================================================  
# MAIN APPLICATION  
# ============================================================================  
def main():
    init_session_state()
    create_directories()
    st.markdown('<h1 class="main-title">🧠 SR-HGN Plus Link Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Semantic-and Relation-aware Heterogeneous Graph Neural Network with FiLM and Local-Global Fusion</p>', unsafe_allow_html=True)
    required_files = ['model_plus.py', 'utils.py']
    missing_files = [f for f in required_files if not Path(f).exists()]
    if missing_files:
        st.error(f"❌ Thiếu files: {', '.join(missing_files)}")
        st.info("📁 Hãy copy files model_plus.py và utils.py vào thư mục hiện tại")
        st.stop()
    if not MODEL_FILES_AVAILABLE:
        st.error("❌ Không thể import model files. Hãy kiểm tra lại!")
        st.stop()
    with st.sidebar:
        st.header("🔧 Cấu hình")
        dataset = st.selectbox("📊 Dataset", ["acm", "dblp", "imdb"], index=0, help="Chọn dataset để thử nghiệm")
        st.session_state.dataset = dataset
        st.subheader("🧠 Tham số Model")
        hidden_dim = st.slider("Hidden Dimension", 32, 256, 64, 32)
        st.session_state.hidden_dim = hidden_dim
        num_layers = st.slider("Number of Layers", 1, 4, 2)
        st.session_state.num_layers = num_layers
        num_node_heads = st.slider("Node Attention Heads", 1, 8, 4)
        st.session_state.num_node_heads = num_node_heads
        num_type_heads = st.slider("Type Attention Heads", 1, 8, 4)
        st.session_state.num_type_heads = num_type_heads
        alpha = st.slider("Alpha (Semantic/Relation Balance)", 0.0, 1.0, 0.5, 0.1)
        st.session_state.alpha = alpha
        dropout = st.slider("Dropout", 0.0, 0.5, 0.2, 0.1)
        st.session_state.dropout = dropout
        prediction_type = st.selectbox("Prediction Type", ["dot", "mlp", "bilinear", "dist", "complex"], index=1)
        st.session_state.prediction_type = prediction_type
        global_gnn = st.selectbox("Global GNN", ["graphsage", "rgcn"], index=0)
        st.session_state.global_gnn = global_gnn
        st.subheader("🚀 Tham số Training")
        learning_rate = st.select_slider("Learning Rate", options=[0.001, 0.005, 0.01, 0.05], value=0.005)
        st.session_state.learning_rate = learning_rate
        num_epochs = st.slider("Number of Epochs", 10, 200, 200, 10)
        st.session_state.num_epochs = num_epochs
        weight_decay = st.select_slider("Weight Decay", options=[1e-5, 1e-4, 1e-3], value=1e-4)
        st.session_state.weight_decay = weight_decay
        clip = st.slider("Gradient Clipping", 0.0, 2.0, 1.0, 0.1)
        st.session_state.clip = clip
        early_stop = st.slider("Early Stop Patience", 5, 50, 20, 5)
        st.session_state.early_stop = early_stop
        eval_every = st.slider("Evaluation Frequency", 1, 20, 10, 1)
        st.session_state.eval_every = eval_every
        scheduler = st.selectbox("Learning Rate Scheduler", ["cosine", "step", "plateau"], index=0)
        st.session_state.scheduler = scheduler
        neg_sampling = st.selectbox("Negative Sampling", ["uniform", "degree_aware", "structure_aware", "mixed"], index=3)
        st.session_state.neg_sampling = neg_sampling
        neg_ratio = st.slider("Negative Sample Ratio", 1, 5, 3)
        st.session_state.neg_ratio = neg_ratio
        test_ratio = st.slider("Test Ratio", 0.1, 0.3, 0.1, 0.05)
        st.session_state.test_ratio = test_ratio
        val_ratio = st.slider("Validation Ratio", 0.1, 0.3, 0.1, 0.05)
        st.session_state.val_ratio = val_ratio
    col1, col2 = st.columns([3, 2])
    with col1:
        st.header("📊 Dataset & Training")
        if st.button("📂 Load Dataset", type="primary"):
            with st.spinner("Loading dataset..."):
                success, message = load_dataset(dataset)
                if success:
                    show_message(message, "success")
                    g = st.session_state.graph
                    features = st.session_state.features
                    display_dataset_info(g, features, dataset)
                else:
                    show_message(message, "error")
        if st.session_state.dataset_loaded:
            st.markdown("### 🚀 Model Training")
            col_train, col_stop = st.columns(2)
            with col_train:
                if st.button("🎯 Start Training", type="primary", disabled=st.session_state.is_training):
                    config = {
                        'hidden_dim': hidden_dim,
                        'num_layers': num_layers,
                        'num_node_heads': num_node_heads,
                        'num_type_heads': num_type_heads,
                        'alpha': alpha,
                        'dropout': dropout,
                        'prediction_type': prediction_type,
                        'global_gnn': global_gnn,
                        'learning_rate': learning_rate,
                        'num_epochs': num_epochs,
                        'weight_decay': weight_decay,
                        'clip': clip,
                        'early_stop': early_stop,
                        'eval_every': eval_every,
                        'scheduler': scheduler,
                        'neg_sampling': neg_sampling,
                        'neg_ratio': neg_ratio,
                        'test_ratio': test_ratio,
                        'val_ratio': val_ratio
                    }
                    success, message = train_model(config)
                    if success:
                        show_message(message, "success")
                    else:
                        show_message(message, "error")
            with col_stop:
                if st.button("⏹️ Stop Training", disabled=not st.session_state.is_training):
                    st.session_state.is_training = False
                    add_training_log("Training stopped by user")
                    show_message("Training stopped bởi user", "info")
            if st.session_state.model is not None and not st.session_state.is_training:
                st.markdown("### 🎯 Final Results")
                if st.button("🔍 Evaluate Model", type="primary"):
                    with st.spinner("Đang đánh giá model..."):
                        results = evaluate_model()
                        if results:
                            st.session_state.current_results = results
                            show_results_popup(results)
                if st.session_state.current_results:
                    results = st.session_state.current_results
                    metrics_cols = st.columns(5)
                    with metrics_cols[0]:
                        st.markdown(
                            f'<div class="metric-container"><h4>AUC</h4><h2>{results["auc"]:.4f}</h2></div>',
                            unsafe_allow_html=True
                        )
                    with metrics_cols[1]:
                        st.markdown(
                            f'<div class="metric-container"><h4>AP</h4><h2>{results["ap"]:.4f}</h2></div>',
                            unsafe_allow_html=True
                        )
                    with metrics_cols[2]:
                        st.markdown(
                            f'<div class="metric-container"><h4>F1</h4><h2>{results["f1"]:.4f}</h2></div>',
                            unsafe_allow_html=True
                        )
                    with metrics_cols[3]:
                        st.markdown(
                            f'<div class="metric-container"><h4>MRR</h4><h2>{results["mrr"]:.4f}</h2></div>',
                            unsafe_allow_html=True
                        )
                    with metrics_cols[4]:
                        st.markdown(
                            f'<div class="metric-container"><h4>Hit@10</h4><h2>{results["hit@10"]:.4f}</h2></div>',
                            unsafe_allow_html=True
                        )
    if st.session_state.model is not None and st.session_state.dataset_loaded:
        target_node_type = st.session_state.target_node_type
        max_nodes = st.session_state.graph.num_nodes(target_node_type)
        st.markdown("### 🔎 Dự Đoán Liên Kết")
        st.markdown("#### Dự Đoán Liên Kết Giữa Hai Node")
        st.markdown(f"""
        <div class="info-box">
        <strong>📝 Hướng dẫn:</strong><br>
        • Nhập ID của 2 nút {target_node_type}, cách nhau bằng dấu cách (ví dụ: 0 1)<br>
        • ID hợp lệ: từ 0 đến {max_nodes-1:,}<br>
        • Kết quả sẽ hiển thị xác suất liên kết và dự đoán (ngưỡng 50%)<br>
        </div>
        """, unsafe_allow_html=True)
        two_nodes_input = st.text_input(f"Nhập ID của 2 nút {target_node_type} (cách nhau bằng dấu cách)", key="two_nodes_input")
        if st.button("🔍 Dự Đoán Hai Node", key="predict_two_nodes"):
            try:
                node_ids = [int(x) for x in two_nodes_input.strip().split()]
                if len(node_ids) != 2:
                    show_message("Vui lòng nhập đúng 2 ID nút, cách nhau bằng dấu cách!", "error")
                else:
                    node1_id, node2_id = node_ids
                    probability, error = predict_link_probability(node1_id, node2_id)
                    if error:
                        show_message(error, "error")
                    else:
                        prediction = "Có liên kết" if probability > 0.5 else "Không"
                        df_two = pd.DataFrame([(node1_id, node2_id, probability, prediction)],
                                              columns=["Node 1", "Node 2", "Probability", "Prediction"])
                        st.write(f"**Kết quả dự đoán liên kết giữa nút {node1_id} và {node2_id}:**")
                        st.dataframe(df_two, use_container_width=True)
            except ValueError:
                show_message("Vui lòng nhập ID nút là số nguyên, cách nhau bằng dấu cách!", "error")
        st.markdown("#### Dự Đoán Liên Kết Giữa Hàng Loạt Node")
        st.markdown(f"""
        <div class="info-box">
        <strong>📝 Hướng dẫn:</strong><br>
        • Nhập danh sách ID của các nút {target_node_type}, cách nhau bằng dấu cách (ví dụ: 0 1 2 3)<br>
        • ID hợp lệ: từ 0 đến {max_nodes-1:,}<br>
        • Mô hình sẽ dự đoán liên kết giữa tất cả các cặp nút trong danh sách<br>
        • Kết quả hiển thị xác suất liên kết và dự đoán (ngưỡng 50%)<br>
        </div>
        """, unsafe_allow_html=True)
        multiple_nodes_input = st.text_input(f"Nhập danh sách ID {target_node_type} (cách nhau bằng dấu cách)", key="multiple_nodes_input")
        if st.button("🔍 Dự Đoán Hàng Loạt Node", key="predict_multiple_nodes"):
            try:
                node_ids = [int(x) for x in multiple_nodes_input.strip().split()]
                results, error = predict_multiple_nodes(node_ids)
                if error:
                    show_message(error, "error")
                else:
                    df_multiple = pd.DataFrame(results, columns=["Node 1", "Node 2", "Probability", "Prediction"])
                    df_multiple.sort_values(by="Probability", ascending=False, inplace=True)
                    st.write(f"**Kết quả dự đoán liên kết giữa {len(node_ids)} nút:**")
                    st.dataframe(df_multiple, use_container_width=True, height=600)
            except ValueError:
                show_message("Vui lòng nhập ID nút là số nguyên, cách nhau bằng dấu cách!", "error")
        st.markdown("#### Dự Đoán Tất Cả Liên Kết Từ Node Mục Tiêu")
        st.markdown(f"""
        <div class="info-box">
        <strong>📝 Hướng dẫn:</strong><br>
        • Nhập ID của 1 nút {target_node_type} để dự đoán khả năng có liên kết<br>
        • ID hợp lệ: từ 0 đến {max_nodes-1:,}<br>
        • Mô hình sẽ in ra các node đã có liên kết với node được cho, và dự đoán liên kết với tất cả các node còn lại<br>
        • Ngưỡng: 50% để quyết định có hay không có liên kết<br>
        • Sub-graph hiển thị hàng xóm cấp 1 và cấp 2 của node mục tiêu
        </div>
        """, unsafe_allow_html=True)
        single_node_id = st.number_input(
            f"Nhập ID {target_node_type} (0 ~ {max_nodes-1})",
            min_value=0, max_value=max_nodes-1, value=0, step=1, key="single_node_input"
        )
        if st.button("🔍 Dự Đoán Tất Cả Từ Node", key="predict_all_from_node"):
            u = int(single_node_id)
            collab_network = extract_coauthorship_network(st.session_state.graph, st.session_state.dataset_name)
            existing_neighbors = list(collab_network.neighbors(u)) if u in collab_network.nodes else []
            if existing_neighbors:
                st.markdown(f"**Các nút đã liên kết với nút {u} (hàng xóm cấp 1):** {existing_neighbors}")
            else:
                st.markdown(f"**Nút {u} chưa có liên kết hợp tác nào trước đó.**")
            st.markdown(f"**Sub-graph của nút {u} (Hàng xóm cấp 1 và cấp 2):**")
            fig = visualize_subgraph(collab_network, u, max_nodes)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("**Không thể hiển thị sub-graph do lỗi dữ liệu.**")
            model = st.session_state.model
            g = st.session_state.graph.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            metapath_dict = st.session_state.metapath_dict
            model.eval()
            results = []
            with torch.no_grad():
                node_embeddings = model.generate_node_embeddings(g, metapath_dict)
                H = node_embeddings[target_node_type]
                for v in range(max_nodes):
                    if v == u or v in existing_neighbors:
                        continue
                    src_emb = H[u].unsqueeze(0)
                    dst_emb = H[v].unsqueeze(0)
                    score = model.link_predictor(src_emb, dst_emb)
                    prob = torch.sigmoid(score).item()
                    if prob > 0.5:
                        results.append((u, v, prob, "Có liên kết"))
            results.sort(key=lambda x: x[2], reverse=True)
            df_single = pd.DataFrame(results, columns=["Node 1", "Node 2", "Probability", "Prediction"])
            num_pred = len(df_single)
            possible = max_nodes - len(existing_neighbors)
            ratio = num_pred / possible if possible > 0 else 0
            st.write(f"• Tổng số dự đoán từ node {u}: {num_pred:,}")
            st.write(f"• Tỷ lệ dự đoán liên kết: {num_pred:,}/{possible:,} = {ratio:.2%}")
            st.dataframe(df_single, use_container_width=True, height=600)
    if st.session_state.training_logs:
        st.header("📋 Training Logs")
        log_text = "\n".join(st.session_state.training_logs[-20:])
        st.markdown(f'<div class="training-log">{log_text}</div>', unsafe_allow_html=True)
        if st.button("📄 Download Log File", key="download_log"):
            log_content = "\n".join(st.session_state.training_logs)
            st.download_button(
                label="Download Complete Log",
                data=log_content,
                file_name=f"sr_hgn_plus_training_{st.session_state.dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}.log",
                mime="text/plain",
                key="download_log_btn"
            )
    if st.session_state.current_results:
        st.header("💾 Export Results")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Download Results (JSON)", key="download_json"):
                results_json = json.dumps(st.session_state.current_results, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=results_json,
                    file_name=f"srhgn_plus_results_{st.session_state.dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_json_btn"
                )
        with col2:
            if st.button("📊 Download Training History (CSV)", key="download_csv"):
                if st.session_state.training_history['epoch']:
                    history_df = pd.DataFrame(st.session_state.training_history)
                    csv = history_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"srhgn_plus_history_{st.session_state.dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_csv_btn"
                    )
    st.markdown("---")
    st.markdown(f"""
    ### 📚 About SR-HGN Plus

    **SR-HGN Plus** is an advanced GNN model designed for link prediction in heterogeneous graphs, enhanced with FiLM and Local-Global Fusion.

    **Tính năng chính:**
    - 🔗 Dự đoán xác suất liên kết giữa 2 node bất kỳ
    - 🔄 Dự đoán liên kết giữa hàng loạt node
    - 🔎 Dự đoán tất cả các liên kết từ một node mục tiêu với sub-graph hiển thị hàng xóm cấp 1 và cấp 2
    - 🎯 Đánh giá chi tiết với nhiều metrics
    - 📄 Export results ra CSV/JSON cho phân tích sau
    - 🧠 Tích hợp FiLM và Local-Global Fusion cho hiệu suất cao hơn

    **Log file lưu tại:** `{st.session_state.log_file if 'log_file' in st.session_state else 'log/'}`
    """)

if __name__ == "__main__":
    main()