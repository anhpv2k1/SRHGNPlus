import argparse
import datetime
import torch
import torch.nn.functional as F
import numpy as np
import random
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import dgl
import warnings
import os
import networkx as nx
from collections import defaultdict
warnings.filterwarnings('ignore')

from model_plus import LinkPredictionSRHGNPlus
from utils import (
    set_random_seed, load_data, set_logger, get_checkpoint_path,
    create_train_val_test_splits, extract_coauthorship_network,
    get_target_node_type, generate_metapaths, create_features_for_missing_nodes
)

def evaluate_link_prediction(model, g, pos_edges, neg_edges, device, node_type='author', metapath_dict=None):
    model.eval()
    with torch.no_grad():
        try:
            node_embeddings = model.generate_node_embeddings(g, metapath_dict)
            if node_type not in node_embeddings:
                print(f"Warning: Node type '{node_type}' not found in embeddings. Available types: {list(node_embeddings.keys())}")
                return {
                    'auc': 0.0,
                    'ap': 0.0,
                    'f1': 0.0,
                    'mrr': 0.0,
                    'hit@10': 0.0
                }
            
            target_embeddings = node_embeddings[node_type]
            pos_scores = []
            if pos_edges:
                for src, dst in pos_edges:
                    if src < target_embeddings.shape[0] and dst < target_embeddings.shape[0]:
                        src_emb = target_embeddings[src].unsqueeze(0)
                        dst_emb = target_embeddings[dst].unsqueeze(0)
                        score = model.link_predictor(src_emb, dst_emb)
                        pos_scores.append(score.item())
            
            neg_scores = []
            if neg_edges:
                for src, dst in neg_edges:
                    if src < target_embeddings.shape[0] and dst < target_embeddings.shape[0]:
                        src_emb = target_embeddings[src].unsqueeze(0)
                        dst_emb = target_embeddings[dst].unsqueeze(0)
                        score = model.link_predictor(src_emb, dst_emb)
                        neg_scores.append(score.item())
            
            if pos_scores and neg_scores:
                all_scores = pos_scores + neg_scores
                all_labels = [1] * len(pos_scores) + [0] * len(neg_scores)
                try:
                    auc_score = roc_auc_score(all_labels, all_scores)
                    ap_score = average_precision_score(all_labels, all_scores)
                    precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
                    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                    best_f1 = np.max(f1_scores)
                    
                    pos_scores_array = np.array(pos_scores)
                    neg_scores_array = np.array(neg_scores)
                    reciprocal_ranks = []
                    for pos_score in pos_scores:
                        rank = 1 + np.sum(neg_scores_array > pos_score)
                        reciprocal_ranks.append(1.0 / rank)
                    mrr = np.mean(reciprocal_ranks)
                    
                    hit_10 = 0
                    for pos_score in pos_scores:
                        rank = 1 + np.sum(neg_scores_array > pos_score)
                        if rank <= 10:
                            hit_10 += 1
                    hit_10 = hit_10 / len(pos_scores) if pos_scores else 0
                    
                    return {
                        'auc': auc_score,
                        'ap': ap_score,
                        'f1': best_f1,
                        'mrr': mrr,
                        'hit@10': hit_10
                    }
                except Exception as e:
                    print(f"Error calculating metrics: {e}")
                    return {
                        'auc': 0.0,
                        'ap': 0.0,
                        'f1': 0.0,
                        'mrr': 0.0,
                        'hit@10': 0.0
                    }
            else:
                return {
                    'auc': 0.0,
                    'ap': 0.0,
                    'f1': 0.0,
                    'mrr': 0.0,
                    'hit@10': 0.0
                }
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return {
                'auc': 0.0,
                'ap': 0.0,
                'f1': 0.0,
                'mrr': 0.0,
                'hit@10': 0.0
            }

def train_link_prediction(args):
    logger = set_logger(f'sr_hgn_plus_link_pred_{args.dataset}')
    logger.info(f'Training SR-HGN Plus for link prediction on {args.dataset}')
    logger.info(f'Arguments: {args}')
    
    set_random_seed(args.seed)
    
    try:
        g, node_dict, edge_dict, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, target = load_data(
            args.dataset, args.train_split, args.val_split, feat=args.feat
        )
        
        logger.info(f'Loaded {args.dataset} dataset successfully')
        logger.info(f'Graph info: {g}')
        logger.info(f'Node types: {g.ntypes}')
        logger.info(f'Edge types: {g.etypes}')
        
        if not features:
            features = {}
            logger.info('No features provided, will create random features')
        
        input_dims = {}
        for ntype in g.ntypes:
            if ntype in features:
                input_dims[ntype] = features[ntype].shape[1]
                logger.info(f'{ntype} has features with dim {input_dims[ntype]}')
            else:
                input_dims[ntype] = 128
                logger.info(f'{ntype} missing features, will use default dim {input_dims[ntype]}')
        
        features, input_dims = create_features_for_missing_nodes(g, features, input_dims, args.dataset)
        logger.info(f'Final input dimensions: {input_dims}')
        
        for ntype in g.ntypes:
            if 'x' not in g.nodes[ntype].data:
                logger.error(f'Node type {ntype} missing features in graph!')
                return None
            else:
                logger.info(f'Verified: {ntype} has features shape {g.nodes[ntype].data["x"].shape}')
        
        collab_network = extract_coauthorship_network(g, args.dataset)
        if collab_network.number_of_edges() == 0:
            logger.error('No collaboration edges found! Check the graph structure.')
            return None
        
        logger.info(f'Collaboration network created with {collab_network.number_of_nodes()} nodes and {collab_network.number_of_edges()} edges')
        
        train_edges, val_edges, test_edges, train_neg, val_neg, test_neg = create_train_val_test_splits(
            collab_network, test_ratio=args.test_ratio, val_ratio=args.val_ratio,
            negative_sampling=args.neg_sampling, neg_sample_ratio=args.neg_ratio
        )
        
        logger.info(f'Train edges: {len(train_edges)}, Val edges: {len(val_edges)}, Test edges: {len(test_edges)}')
        logger.info(f'Train neg: {len(train_neg)}, Val neg: {len(val_neg)}, Test neg: {len(test_neg)}')
        
        target_node_type = get_target_node_type(args.dataset)
        if target_node_type not in g.ntypes:
            logger.error(f'Target node type "{target_node_type}" not found in graph. Available types: {g.ntypes}')
            target_node_type = g.ntypes[0]
            logger.info(f'Using fallback target node type: {target_node_type}')
        
        # Generate metapath dictionary
        metapath_dict = generate_metapaths(g, num_metapaths=16, metapath_length=4)
        logger.info(f'Generated metapath dictionary: {metapath_dict}')
        
        model = LinkPredictionSRHGNPlus(
            G=g,
            node_dict=node_dict,
            edge_dict=edge_dict,
            input_dims=input_dims,
            hidden_dim=args.n_hid,
            num_layers=args.n_layers,
            num_node_heads=args.num_node_heads,
            num_type_heads=args.num_type_heads,
            alpha=args.alpha,
            prediction_type=args.prediction_type,
            target_node_type=target_node_type,
            dropout=args.dropout,
            global_gnn=args.global_gnn
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
        model = model.to(device)
        g = g.to(device)
        
        logger.info(f'Model initialized with {sum(p.numel() for p in model.parameters())} parameters')
        logger.info(f'Using device: {device}')
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)
        
        if args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
        elif args.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        best_val_auc = 0
        best_val_ap = 0
        early_stop_counter = 0
        best_epoch = 0
        training_losses = []
        
        logger.info('Starting training...')
        
        for epoch in range(args.num_epochs):
            model.train()
            
            if args.use_edge_sampling and len(train_edges) > args.max_edges_per_epoch:
                sampled_indices = np.random.choice(len(train_edges), args.max_edges_per_epoch, replace=False)
                epoch_train_edges = [train_edges[i] for i in sampled_indices]
                epoch_train_neg = [train_neg[i] for i in sampled_indices[:len(train_neg)]]
            else:
                epoch_train_edges = train_edges
                epoch_train_neg = train_neg
            
            pos_graph = dgl.graph(([], []), num_nodes=g.num_nodes(target_node_type))
            neg_graph = dgl.graph(([], []), num_nodes=g.num_nodes(target_node_type))
            
            if epoch_train_edges:
                train_src, train_dst = zip(*epoch_train_edges)
                pos_graph = dgl.graph((train_src, train_dst), num_nodes=g.num_nodes(target_node_type))
            
            if epoch_train_neg:
                train_neg_src, train_neg_dst = zip(*epoch_train_neg)
                neg_graph = dgl.graph((train_neg_src, train_neg_dst), num_nodes=g.num_nodes(target_node_type))
            
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)
            
            try:
                pos_scores, neg_scores = model(g, pos_graph, neg_graph, metapath_dict)
            except KeyError as e:
                logger.error(f'KeyError during forward pass: {e}')
                logger.error('Checking graph node data...')
                for ntype in g.ntypes:
                    logger.error(f'{ntype} data keys: {list(g.nodes[ntype].data.keys())}')
                return None
            except Exception as e:
                logger.error(f'Error during forward pass: {e}')
                return None
            
            if len(pos_scores) > 0 and len(neg_scores) > 0:
                pos_labels = torch.ones(len(pos_scores), device=device)
                neg_labels = torch.zeros(len(neg_scores), device=device)
                all_scores = torch.cat([pos_scores, neg_scores])
                all_labels = torch.cat([pos_labels, neg_labels])
                
                if args.loss_function == 'bce':
                    loss = F.binary_cross_entropy_with_logits(all_scores, all_labels)
                elif args.loss_function == 'margin':
                    pos_loss = F.relu(1 - pos_scores).mean()
                    neg_loss = F.relu(1 + neg_scores).mean()
                    loss = pos_loss + neg_loss
                else:
                    loss = F.binary_cross_entropy_with_logits(all_scores, all_labels)
                
                if args.l2_reg > 0:
                    l2_reg = torch.tensor(0., device=device)
                    for param in model.parameters():
                        l2_reg += torch.norm(param)
                    loss += args.l2_reg * l2_reg
                
                optimizer.zero_grad()
                loss.backward()
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                
                if args.scheduler in ['cosine', 'step']:
                    scheduler.step()
                
                training_losses.append(loss.item())
            else:
                loss = torch.tensor(0.0)
                training_losses.append(0.0)
            
            if epoch % args.eval_every == 0 or epoch == args.num_epochs - 1:
                val_metrics = evaluate_link_prediction(model, g, val_edges, val_neg, device, target_node_type, metapath_dict)
                if args.scheduler == 'plateau':
                    scheduler.step(val_metrics['auc'])
                
                logger.info(f'Epoch {epoch:04d} | Loss: {loss.item():.4f} | '
                           f'Val AUC: {val_metrics["auc"]:.4f} | '
                           f'Val AP: {val_metrics["ap"]:.4f} | '
                           f'Val F1: {val_metrics["f1"]:.4f} | '
                           f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
                
                current_score = val_metrics['auc'] + val_metrics['ap']
                best_score = best_val_auc + best_val_ap
                
                if current_score > best_score:
                    best_val_auc = val_metrics['auc']
                    best_val_ap = val_metrics['ap']
                    early_stop_counter = 0
                    best_epoch = epoch
                    checkpoint_path = get_checkpoint_path(args, epoch)
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch,
                        'val_auc': val_metrics['auc'],
                        'val_ap': val_metrics['ap'],
                        'args': args
                    }, checkpoint_path)
                    logger.info(f'New best model saved at epoch {epoch}')
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= args.early_stop:
                        logger.info(f'Early stopping after {args.early_stop} epochs without improvement')
                        break
        
        checkpoint_path = get_checkpoint_path(args, best_epoch)
        if os.path.exists(checkpoint_path):
            if torch.cuda.is_available() and args.cuda:
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f'Loaded best model from epoch {best_epoch}')
        else:
            logger.warning(f"Checkpoint not found at {checkpoint_path}, using current model")
        
        test_metrics = evaluate_link_prediction(model, g, test_edges, test_neg, device, target_node_type, metapath_dict)
        logger.info('='*50)
        logger.info('Final Test Results:')
        logger.info(f'  AUC: {test_metrics["auc"]:.4f}')
        logger.info(f'  AP: {test_metrics["ap"]:.4f}')
        logger.info(f'  F1: {test_metrics["f1"]:.4f}')
        logger.info(f'  MRR: {test_metrics["mrr"]:.4f}')
        logger.info(f'  Hit@10: {test_metrics["hit@10"]:.4f}')
        logger.info('='*50)
        
        return test_metrics
        
    except Exception as e:
        logger.error(f'Error during training: {str(e)}')
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    parser = argparse.ArgumentParser(description='SR-HGN Plus Link Prediction')
    
    parser.add_argument('--dataset', type=str, default='acm', choices=['acm', 'dblp', 'imdb'])
    parser.add_argument('--feat', type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--model_type', type=str, default='plus', choices=['standard', 'plus'])
    parser.add_argument('--n_hid', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--num_node_heads', type=int, default=4)
    parser.add_argument('--num_type_heads', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--prediction_type', type=str, default='mlp', choices=['dot', 'mlp', 'bilinear', 'dist', 'complex'])
    parser.add_argument('--global_gnn', type=str, default='graphsage', choices=['graphsage', 'rgcn'])
    parser.add_argument('--neg_sampling', type=str, default='mixed', choices=['uniform', 'degree_aware', 'structure_aware', 'mixed'])
    parser.add_argument('--neg_ratio', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--max_lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'plateau'])
    parser.add_argument('--loss_function', type=str, default='bce', choices=['bce', 'margin'])
    parser.add_argument('--l2_reg', type=float, default=0.0)
    parser.add_argument('--use_edge_sampling', action='store_true', default=False)
    parser.add_argument('--max_edges_per_epoch', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--prefix', type=str, default='sr_hgn_plus_link_pred')
    
    args = parser.parse_args()
    
    if args.test_ratio + args.val_ratio >= 1.0:
        raise ValueError("test_ratio + val_ratio should be less than 1.0")
    
    if args.neg_ratio <= 0:
        raise ValueError("neg_ratio should be positive")
    
    results = train_link_prediction(args)
    
    if results is not None:
        print("\n" + "="*50)
        print("FINAL RESULTS")
        print("="*50)
        print(f"Dataset: {args.dataset}")
        print(f"Model: SR-HGN Plus")
        print(f"Prediction Type: {args.prediction_type}")
        print(f"Negative Sampling: {args.neg_sampling}")
        print("-"*50)
        print(f"AUC:     {results['auc']:.4f}")
        print(f"AP:      {results['ap']:.4f}")
        print(f"F1:      {results['f1']:.4f}")
        print(f"MRR:     {results['mrr']:.4f}")
        print(f"Hit@10:  {results['hit@10']:.4f}")
        print("="*50)
    else:
        print("Training failed!")

if __name__ == '__main__':
    main()
