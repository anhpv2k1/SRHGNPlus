import datetime
import numpy as np
import os
import os.path as osp
import random
import logging
from pathlib import Path
import networkx as nx
from collections import defaultdict
from scipy import sparse
from scipy import io as sio
from itertools import product
import torch
import dgl

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_date_postfix():
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(dt.date(), dt.hour, dt.minute, dt.second)
    return post_fix

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size, dtype=torch.bool)
    mask[indices] = 1
    return mask.byte()

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

def generate_metapaths(G, num_metapaths=16, metapath_length=4):
    """Generate random meta-paths based on network schema."""
    metapaths = {}
    for ntype in G.ntypes:
        metapaths[ntype] = []
        for _ in range(num_metapaths):
            path = [ntype]
            current_type = ntype
            for _ in range(metapath_length - 1):
                # Get possible edge types from current node type
                possible_etypes = [(src, e, dst) for src, e, dst in G.canonical_etypes if src == current_type]
                if not possible_etypes:
                    break
                # Randomly select an edge type
                etype = random.choice(possible_etypes)
                path.append(etype[2])  # Add destination node type
                current_type = etype[2]
            metapaths[ntype].append(path)
    return metapaths

def load_acm_raw(train_split=0.2, val_split=0.3, feat=1):
    data_folder = './data/acm/'
    data_path = osp.join(data_folder, 'ACM.mat')
    
    if not osp.exists(data_path):
        raise FileNotFoundError(f"ACM.mat not found at {data_path}")
    
    data = sio.loadmat(data_path)
    target = 'paper'

    p_vs_l = data['PvsL']
    p_vs_a = data['PvsA']
    p_vs_t = data['PvsT']
    p_vs_p = data['PvsP']
    p_vs_c = data['PvsC']

    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    p_vs_c = p_vs_c[p_selected]
    p_vs_p = p_vs_p[p_selected].T[p_selected]
    a_selected = (p_vs_a[p_selected].sum(0) != 0).A1.nonzero()[0]
    p_vs_a = p_vs_a[p_selected].T[a_selected].T
    l_selected = (p_vs_l[p_selected].sum(0) != 0).A1.nonzero()[0]
    p_vs_l = p_vs_l[p_selected].T[l_selected].T
    t_selected = (p_vs_t[p_selected].sum(0) != 0).A1.nonzero()[0]
    p_vs_t = p_vs_t[p_selected].T[t_selected].T

    if feat == 1 or feat == 3:
        hg = dgl.heterograph({
            ('paper', 'paper_paper_cite', 'paper'): p_vs_p.nonzero(),
            ('paper', 'paper_paper_ref', 'paper'): p_vs_p.transpose().nonzero(),
            ('paper', 'paper_author', 'author'): p_vs_a.nonzero(),
            ('author', 'author_paper', 'paper'): p_vs_a.transpose().nonzero(),
            ('paper', 'paper_subject', 'subject'): p_vs_l.nonzero(),
            ('subject', 'subject_paper', 'paper'): p_vs_l.transpose().nonzero(),
        })
        paper_feats = torch.FloatTensor(p_vs_t.toarray())
        features = {'paper': paper_feats}
    elif feat == 2:
        hg = dgl.heterograph({
            ('paper', 'paper_paper_cite', 'paper'): p_vs_p.nonzero(),
            ('paper', 'paper_paper_ref', 'paper'): p_vs_p.transpose().nonzero(),
            ('paper', 'paper_author', 'author'): p_vs_a.nonzero(),
            ('author', 'author_paper', 'paper'): p_vs_a.transpose().nonzero(),
            ('paper', 'paper_subject', 'subject'): p_vs_l.nonzero(),
            ('subject', 'subject_paper', 'paper'): p_vs_l.transpose().nonzero(),
            ('paper', 'paper_term', 'term'): p_vs_t.nonzero(),
            ('term', 'term_paper', 'paper'): p_vs_t.transpose().nonzero()
        })
        features = {}
    elif feat == 4:
        hg = dgl.heterograph({
            ('paper', 'paper_paper_cite', 'paper'): p_vs_p.nonzero(),
            ('paper', 'paper_paper_ref', 'paper'): p_vs_p.transpose().nonzero(),
            ('paper', 'paper_author', 'author'): p_vs_a.nonzero(),
            ('author', 'author_paper', 'paper'): p_vs_a.transpose().nonzero(),
            ('paper', 'paper_subject', 'subject'): p_vs_l.nonzero(),
            ('subject', 'subject_paper', 'paper'): p_vs_l.transpose().nonzero(),
        })
        features = {}
        
    print(f"ACM Graph: {hg}")

    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    labels = torch.LongTensor(labels)

    num_classes = 3
    split_file = osp.join(data_folder, f'train_val_test_idx_{int(train_split * 100)}_2021.npz')
    if osp.exists(split_file):
        split = np.load(split_file)
        train_idx = split['train_idx']
        val_idx = split['val_idx']
        test_idx = split['test_idx']
    else:
        num_nodes = len(p_selected)
        indices = np.random.permutation(num_nodes)
        train_size = int(num_nodes * train_split)
        val_size = int(num_nodes * val_split)
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        np.savez(split_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    num_nodes = hg.number_of_nodes('paper')
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    node_dict = {ntype: i for i, ntype in enumerate(hg.ntypes)}
    edge_dict = {etype: i for i, etype in enumerate(hg.etypes)}
    for etype in hg.etypes:
        hg.edges[etype].data['id'] = torch.ones(hg.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]

    return hg, node_dict, edge_dict, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, target

def load_dblp_processed(train_split=0.2, val_split=0.3, feat=1):
    raw_dir = './data/dblp/'
    required_files = ['features_0.npz', 'features_1.npz', 'features_2.npy', 'labels.npy', 'node_types.npy', 'adjM.npz']
    
    for file in required_files:
        if not osp.exists(osp.join(raw_dir, file)):
            raise FileNotFoundError(f"DBLP file {file} not found at {raw_dir}")

    author_feats = sparse.load_npz(osp.join(raw_dir, 'features_0.npz'))
    paper_feats = sparse.load_npz(osp.join(raw_dir, 'features_1.npz'))
    term_feats = np.load(osp.join(raw_dir, 'features_2.npy'))
    node_type_idx = np.load(osp.join(raw_dir, 'node_types.npy'))
    target = 'author'

    if feat == 1:
        author_feats = torch.from_numpy(author_feats.todense()).to(torch.float)
        paper_feats = torch.from_numpy(paper_feats.todense()).to(torch.float)
        term_feats = torch.from_numpy(term_feats).to(torch.float)
        features = {'author': author_feats, 'paper': paper_feats, 'term': term_feats}
    elif feat == 2 or feat == 4:
        features = {}
    elif feat == 3:
        author_feats = torch.from_numpy(author_feats.todense()).to(torch.float)
        features = {'author': author_feats}
        
    node_type_idx = torch.from_numpy(node_type_idx).to(torch.long)
    num_confs = int((node_type_idx == 3).sum())
    labels = np.load(osp.join(raw_dir, 'labels.npy'))
    labels = torch.from_numpy(labels).to(torch.long)

    s = {}
    N_a = author_feats.shape[0]
    N_p = paper_feats.shape[0]
    N_t = term_feats.shape[0]
    N_c = num_confs
    s['author'] = (0, N_a)
    s['paper'] = (N_a, N_a + N_p)
    s['term'] = (N_a + N_p, N_a + N_p + N_t)
    s['conference'] = (N_a + N_p + N_t, N_a + N_p + N_t + N_c)

    node_types = ['author', 'paper', 'term', 'conference']
    A = sparse.load_npz(osp.join(raw_dir, 'adjM.npz'))
    hg_data = dict()

    for src, dst in product(node_types, node_types):
        A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]]
        if A_sub.nnz > 0:
            hg_data[src, src + '_' + dst, dst] = A_sub.nonzero()

    if feat == 1 or feat == 3 or feat == 4:
        pass
    elif feat == 2:
        hg_data['author', 'author_keyword', 'keyword'] = author_feats.nonzero()
        hg_data['keyword', 'keyword_author', 'author'] = author_feats.transpose().nonzero()
        hg_data['paper', 'paper_title', 'title-word'] = paper_feats.nonzero()
        hg_data['title-word', 'title_paper', 'paper'] = paper_feats.transpose().nonzero()
        
    hg = dgl.heterograph(hg_data)
    print(f"DBLP Graph: {hg}")

    split_file = osp.join(raw_dir, f'train_val_test_idx_{int(train_split * 100)}_2021.npz')
    if osp.exists(split_file):
        split = np.load(split_file)
        train_idx = split['train_idx']
        val_idx = split['val_idx']
        test_idx = split['test_idx']
    else:
        num_nodes = len(labels)
        indices = np.random.permutation(num_nodes)
        train_size = int(num_nodes * train_split)
        val_size = int(num_nodes * val_split)
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        np.savez(split_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    train_mask = get_binary_mask(len(labels), train_idx)
    val_mask = get_binary_mask(len(labels), val_idx)
    test_mask = get_binary_mask(len(labels), test_idx)

    num_classes = torch.unique(labels).shape[0]
    node_dict = {ntype: i for i, ntype in enumerate(hg.ntypes)}
    edge_dict = {etype: i for i, etype in enumerate(hg.etypes)}
    for etype in hg.etypes:
        hg.edges[etype].data['id'] = torch.ones(hg.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]

    return hg, node_dict, edge_dict, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, target

def load_imdb_processed(train_split=0.2, val_split=0.3, feat=1):
    raw_dir = './data/imdb/'
    required_files = ['features_0.npz', 'features_1.npz', 'features_2.npz', 'labels.npy', 'adjM.npz']
    
    for file in required_files:
        if not osp.exists(osp.join(raw_dir, file)):
            raise FileNotFoundError(f"IMDB file {file} not found at {raw_dir}")

    node_types = ['movie', 'director', 'actor']
    movie_feats = sparse.load_npz(osp.join(raw_dir, 'features_0.npz'))
    director_feats = sparse.load_npz(osp.join(raw_dir, 'features_1.npz'))
    actor_feats = sparse.load_npz(osp.join(raw_dir, 'features_2.npz'))
    target = 'movie'

    if feat == 1:
        movie_feats = torch.from_numpy(movie_feats.todense()).to(torch.float)
        director_feats = torch.from_numpy(director_feats.todense()).to(torch.float)
        actor_feats = torch.from_numpy(actor_feats.todense()).to(torch.float)
        features = {'movie': movie_feats, 'director': director_feats, 'actor': actor_feats}
    elif feat == 2 or feat == 4:
        features = {}
    elif feat == 3:
        movie_feats = torch.from_numpy(movie_feats.todense()).to(torch.float)
        features = {'movie': movie_feats}

    labels = np.load(osp.join(raw_dir, 'labels.npy'))
    labels = torch.from_numpy(labels).to(torch.long)

    s = {}
    N_m = movie_feats.shape[0]
    N_d = director_feats.shape[0]
    N_a = actor_feats.shape[0]
    s['movie'] = (0, N_m)
    s['director'] = (N_m, N_m + N_d)
    s['actor'] = (N_m + N_d, N_m + N_d + N_a)
    
    A = sparse.load_npz(osp.join(raw_dir, 'adjM.npz'))
    hg_data = dict()

    for src, dst in product(node_types, node_types):
        A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]]
        if A_sub.nnz > 0:
            hg_data[src, src + '_' + dst, dst] = A_sub.nonzero()

    if feat == 1 or feat == 3 or feat == 4:
        pass
    elif feat == 2:
        hg_data['movie', 'movie_keyword', 'keyword'] = movie_feats.nonzero()
        hg_data['keyword', 'keyword_movie', 'movie'] = movie_feats.transpose().nonzero()

    hg = dgl.heterograph(hg_data)
    print(f"IMDB Graph: {hg}")

    split_file = osp.join(raw_dir, f'train_val_test_idx_{int(train_split * 100)}_2021.npz')
    if osp.exists(split_file):
        split = np.load(split_file)
        train_idx = split['train_idx']
        val_idx = split['val_idx']
        test_idx = split['test_idx']
    else:
        num_nodes = len(labels)
        indices = np.random.permutation(num_nodes)
        train_size = int(num_nodes * train_split)
        val_size = int(num_nodes * val_split)
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        np.savez(split_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    train_mask = get_binary_mask(len(labels), train_idx)
    val_mask = get_binary_mask(len(labels), val_idx)
    test_mask = get_binary_mask(len(labels), test_idx)

    num_classes = torch.unique(labels).shape[0]
    node_dict = {ntype: i for i, ntype in enumerate(hg.ntypes)}
    edge_dict = {etype: i for i, etype in enumerate(hg.etypes)}
    for etype in hg.etypes:
        hg.edges[etype].data['id'] = torch.ones(hg.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]

    return hg, node_dict, edge_dict, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, target

def load_data(dataset, train_split, val_split, feat=1):
    if dataset == 'acm':
        return load_acm_raw(train_split, val_split, feat=feat)
    elif dataset == 'dblp':
        return load_dblp_processed(train_split, val_split, feat=feat)
    elif dataset == 'imdb':
        return load_imdb_processed(train_split, val_split, feat=feat)
    else:
        raise NotImplementedError('Unsupported dataset {}'.format(dataset))

def create_negative_samples(positive_edges, num_nodes, n=None, strategy='uniform', degrees=None):
    if n is None:
        n = len(positive_edges)
    
    positive_set = set(tuple(sorted(edge)) for edge in positive_edges)
    negative_samples = []
    
    if strategy == 'degree_aware' and degrees is not None:
        total_degree = sum(degrees.values())
        node_probs = np.array([degrees.get(i, 0) / total_degree for i in range(num_nodes)])
        node_probs = node_probs / node_probs.sum()
    else:
        node_probs = None
    
    max_attempts = n * 20
    attempts = 0
    
    while len(negative_samples) < n and attempts < max_attempts:
        if strategy == 'degree_aware' and node_probs is not None:
            i = np.random.choice(num_nodes, p=node_probs)
            j = np.random.choice(num_nodes, p=node_probs)
        elif strategy == 'structure_aware':
            if positive_edges:
                ref_edge = positive_edges[np.random.randint(len(positive_edges))]
                if np.random.random() < 0.5:
                    i = ref_edge[0]
                    j = np.random.randint(0, num_nodes)
                else:
                    i = np.random.randint(0, num_nodes)
                    j = ref_edge[1]
            else:
                i = np.random.randint(0, num_nodes)
                j = np.random.randint(0, num_nodes)
        elif strategy == 'mixed':
            if np.random.random() < 0.7:
                i = np.random.randint(0, num_nodes)
                j = np.random.randint(0, num_nodes)
            else:
                if node_probs is not None:
                    i = np.random.choice(num_nodes, p=node_probs)
                    j = np.random.choice(num_nodes, p=node_probs)
                else:
                    i = np.random.randint(0, num_nodes)
                    j = np.random.randint(0, num_nodes)
        else:
            i = np.random.randint(0, num_nodes)
            j = np.random.randint(0, num_nodes)
        
        if i == j:
            attempts += 1
            continue
        
        edge = tuple(sorted([i, j]))
        if edge not in positive_set and edge not in negative_samples:
            negative_samples.append(edge)
        
        attempts += 1
    
    return negative_samples[:n]

def create_train_val_test_splits(graph, test_ratio=0.1, val_ratio=0.1, negative_sampling='uniform', neg_sample_ratio=1):
    edges = list(graph.edges())
    num_edges = len(edges)
    num_test = max(1, int(num_edges * test_ratio))
    num_val = max(1, int(num_edges * val_ratio))
    
    np.random.shuffle(edges)
    train_edges = edges[:-num_test-num_val]
    val_edges = edges[-num_test-num_val:-num_test]
    test_edges = edges[-num_test:]
    
    if len(train_edges) == 0:
        train_edges = edges[:max(1, len(edges) - 2)]
        val_edges = edges[-2:-1] if len(edges) > 1 else []
        test_edges = edges[-1:] if len(edges) > 0 else []
    
    degrees = dict(graph.degree()) if negative_sampling in ['degree_aware', 'mixed'] else None
    num_nodes = graph.number_of_nodes()
    
    train_neg = create_negative_samples(train_edges, num_nodes, n=len(train_edges) * neg_sample_ratio, strategy=negative_sampling, degrees=degrees)
    all_positive_edges = set(tuple(sorted(edge)) for edge in edges)
    
    val_neg = create_negative_samples(edges, num_nodes, n=len(val_edges) * neg_sample_ratio, strategy=negative_sampling, degrees=degrees) if val_edges else []
    test_neg = create_negative_samples(edges, num_nodes, n=len(test_edges) * neg_sample_ratio, strategy=negative_sampling, degrees=degrees) if test_edges else []
    
    return train_edges, val_edges, test_edges, train_neg, val_neg, test_neg

def extract_coauthorship_network(g, dataset='acm'):
    collab_network = nx.Graph()
    
    if dataset == 'acm':
        author_paper_edges = []
        for etype in g.canonical_etypes:
            if etype == ('author', 'author_paper', 'paper'):
                src, dst = g.edges(etype=etype)
                author_paper_edges = list(zip(src.numpy(), dst.numpy()))
                break
            elif etype == ('paper', 'paper_author', 'author'):
                src, dst = g.edges(etype=etype)
                author_paper_edges = list(zip(dst.numpy(), src.numpy()))
                break
        
        if 'author' in g.ntypes:
            num_authors = g.num_nodes('author')
            collab_network.add_nodes_from(range(num_authors))
        
        paper_authors = defaultdict(list)
        for author, paper in author_paper_edges:
            paper_authors[paper].append(author)
        
        for paper, authors in paper_authors.items():
            for i in range(len(authors)):
                for j in range(i+1, len(authors)):
                    a1, a2 = authors[i], authors[j]
                    if collab_network.has_edge(a1, a2):
                        collab_network[a1][a2]['weight'] += 1
                    else:
                        collab_network.add_edge(a1, a2, weight=1)
    
    elif dataset == 'dblp':
        author_paper_edges = []
        for etype in g.canonical_etypes:
            src_type, edge_type, dst_type = etype
            if (src_type == 'author' and dst_type == 'paper') or 'author_paper' in edge_type:
                src, dst = g.edges(etype=etype)
                author_paper_edges = list(zip(src.numpy(), dst.numpy()))
                break
            elif (src_type == 'paper' and dst_type == 'author') or 'paper_author' in edge_type:
                src, dst = g.edges(etype=etype)
                author_paper_edges = list(zip(dst.numpy(), src.numpy()))
                break
        
        if 'author' in g.ntypes:
            num_authors = g.num_nodes('author')
            collab_network.add_nodes_from(range(num_authors))
        
        paper_authors = defaultdict(list)
        for author, paper in author_paper_edges:
            paper_authors[paper].append(author)
        
        for paper, authors in paper_authors.items():
            for i in range(len(authors)):
                for j in range(i+1, len(authors)):
                    a1, a2 = authors[i], authors[j]
                    if collab_network.has_edge(a1, a2):
                        collab_network[a1][a2]['weight'] += 1
                    else:
                        collab_network.add_edge(a1, a2, weight=1)
    
    elif dataset == 'imdb':
        actor_movie_edges = []
        for etype in g.canonical_etypes:
            src_type, edge_type, dst_type = etype
            if (src_type == 'actor' and dst_type == 'movie') or 'actor_movie' in edge_type:
                src, dst = g.edges(etype=etype)
                actor_movie_edges = list(zip(src.numpy(), dst.numpy()))
                break
            elif (src_type == 'movie' and dst_type == 'actor') or 'movie_actor' in edge_type:
                src, dst = g.edges(etype=etype)
                actor_movie_edges = list(zip(dst.numpy(), src.numpy()))
                break
        
        if 'actor' in g.ntypes:
            num_actors = g.num_nodes('actor')
            collab_network.add_nodes_from(range(num_actors))
        
        movie_actors = defaultdict(list)
        for actor, movie in actor_movie_edges:
            movie_actors[movie].append(actor)
        
        for movie, actors in movie_actors.items():
            for i in range(len(actors)):
                for j in range(i+1, len(actors)):
                    a1, a2 = actors[i], actors[j]
                    if collab_network.has_edge(a1, a2):
                        collab_network[a1][a2]['weight'] += 1
                    else:
                        collab_network.add_edge(a1, a2, weight=1)
    
    return collab_network

def get_target_node_type(dataset):
    target_mapping = {'acm': 'author', 'dblp': 'author', 'imdb': 'actor'}
    return target_mapping.get(dataset, 'author')

def set_logger(my_str):
    task_time = get_date_postfix()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    Path(f"log/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(f"log/{my_str}_{task_time}.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def get_checkpoint_path(args, epoch):
    Path('checkpoint').mkdir(parents=True, exist_ok=True)
    checkpoint_path = './checkpoint/{}_{}_{}_{}_{}_{}_{}_{}_{}.pth'.format(
        args.prefix, args.dataset, args.feat, args.train_split, args.seed,
        args.n_hid, args.n_layers, epoch, args.max_lr)
    return checkpoint_path

def create_features_for_missing_nodes(g, features, input_dims, dataset):
    for ntype in g.ntypes:
        if ntype not in features:
            num_nodes = g.num_nodes(ntype)
            if dataset == 'acm':
                if ntype == 'author':
                    features[ntype] = torch.randn(num_nodes, 128)
                    input_dims[ntype] = 128
                elif ntype == 'subject':
                    features[ntype] = torch.randn(num_nodes, 64)
                    input_dims[ntype] = 64
                else:
                    features[ntype] = torch.randn(num_nodes, 100)
                    input_dims[ntype] = 100
            elif dataset == 'dblp':
                if ntype == 'conference':
                    features[ntype] = torch.randn(num_nodes, 64)
                    input_dims[ntype] = 64
                else:
                    features[ntype] = torch.randn(num_nodes, 128)
                    input_dims[ntype] = 128
            elif dataset == 'imdb':
                features[ntype] = torch.randn(num_nodes, 128)
                input_dims[ntype] = 128
            else:
                features[ntype] = torch.randn(num_nodes, 128)
                input_dims[ntype] = 128
    
    for ntype, feat in features.items():
        if ntype in g.ntypes:
            g.nodes[ntype].data['x'] = feat
            print(f"Added features for {ntype}: shape {feat.shape}")
    
    return features, input_dims
