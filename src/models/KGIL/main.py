import random
import torch
import numpy as np
from time import time
from prettytable import PrettyTable
from utils.parser import parse_args
from utils.data_loader import load_data
from utils.evaluate_kgil import test
from utils.helper import early_stopping
from KGIL import EnvGenerator, Recommender
import networkx as nx


n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0


def get_feed_dict(train_entity_pairs, start, end, train_user_set):

    def negative_sampling(user_item, train_user_set, neg_ratio=5):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            for _ in range(neg_ratio):  # Sample neg_ratio negatives per positive
                while True:
                    neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
                    if neg_item not in train_user_set[user]:
                        break
                neg_items.append(neg_item)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs,
                                                                train_user_set, args.neg_ratio)).to(device)
    return feed_dict


def build_graph(train_cf, kg_dict):
    print("\nDetailed graph building process:")
    
    graph = nx.MultiDiGraph()
    print(f"Initial graph state: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
    
    # Add user-item interactions
    for u, i in train_cf:
        if not graph.has_node(u):
            graph.add_node(u, type='user')
        if not graph.has_node(i):
            graph.add_node(i, type='item')
        graph.add_edge(u, i, type='interact')
    
    print(f"After adding interactions: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
    
    # Add knowledge graph triples
    for h, r, t in kg_dict:
        if not graph.has_node(h):
            graph.add_node(h, type='entity')
        if not graph.has_node(t):
            graph.add_node(t, type='entity')
        graph.add_edge(h, t, type=r)
    
    print(f"After adding KG: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
    
    # Verify node types
    node_types = {}
    for node in graph.nodes():
        node_type = graph.nodes[node].get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    print("\nNode type distribution:", node_types)
    
    return graph


def train_one_epoch(model, train_cf, user_dict, args):
    model.train()
    
    # Adjust batch size
    batch_size = min(args.batch_size, len(train_cf))
    n_batch = len(train_cf) // batch_size + 1
    
    total_loss = 0
    for batch_idx in range(n_batch):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(train_cf))
        
        if start >= end:
            break
            
        batch = get_feed_dict(train_cf, start, end, user_dict['train_user_set'])
        
        # Print batch statistics
        print(f"\nBatch {batch_idx + 1}/{n_batch}:")
        print(f"Users: {batch['users'].shape}, range: [{batch['users'].min()}, {batch['users'].max()}]")
        print(f"Pos items: {batch['pos_items'].shape}, range: [{batch['pos_items'].min()}, {batch['pos_items'].max()}]")
        print(f"Neg items: {batch['neg_items'].shape}, range: [{batch['neg_items'].min()}, {batch['neg_items'].max()}]")
        
        rec_loss, inv_mean, inv_var = model(batch)
        outer_loss = rec_loss + args.lamda * (inv_mean + inv_var)
        
        # Print loss components
        print(f"Loss components:")
        print(f"Rec loss: {rec_loss.item():.6f}")
        print(f"Inv mean: {inv_mean.item():.6f}")
        print(f"Inv var: {inv_var.item():.6f}")
        print(f"Total loss: {outer_loss.item():.6f}")
        
        if outer_loss.requires_grad:
            model.zero_grad()
            outer_loss.backward()
            
            # Print gradient norms
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Gradient norm: {total_norm:.6f}")
            
            model.optimizer.step()
            
        total_loss += outer_loss.item()
    
    return total_loss / n_batch


if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() and args.cuda else "cpu")

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list

    print(f"Dataset statistics:")
    print(f"Number of training interactions: {len(train_cf)}")
    print(f"Number of test interactions: {len(test_cf)}")
    print(f"Number of users: {n_params['n_users']}")
    print(f"Number of items: {n_params['n_items']}")
    print(f"Batch size: {args.batch_size}")

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']
    
    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))

    """define model"""
    augmenter = EnvGenerator(args.K, args.dim, args.dim, device)
    model = Recommender(n_params, args, graph, mean_mat_list[0], device).to(device)
    
    """define optimizer"""
    rec_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    aug_optimizer = torch.optim.Adam(augmenter.parameters(), lr=args.lr)
    model.augmenter = augmenter

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False
    total_iter = max(len(train_cf) // args.batch_size, 1)
    get_print = max(total_iter // 20, 1)
    
    print("start training ...")
    for epoch in range(args.epoch):
        # Add epoch start print
        print(f"\nEpoch {epoch+1}/{args.epoch}")
        print(f"Number of iterations expected: {total_iter}")
        
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        model.train()

        train_cf_pairs = train_cf_pairs[index]
        total_outer_loss, total_inner_loss, s = 0, 0, 0
        train_s_t = time()
        iteration = 0
        
        # Add batch processing print
        print(f"Processing {len(train_cf)} samples in batches of {args.batch_size}")
        
        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf_pairs, s, s + args.batch_size, user_dict['train_user_set'])
            
            # Debug print for first batch of first epoch
            if epoch == 0 and iteration == 0:
                print("\nFirst batch statistics:")
                print(f"Users shape: {batch['users'].shape}")
                print(f"Positive items shape: {batch['pos_items'].shape}")
                print(f"Negative items shape: {batch['neg_items'].shape}")
            
            rec_loss, inv_mean, inv_var = model(batch)
            outer_loss = rec_loss + args.lamda * (inv_mean + inv_var)
            
            # Debug print losses
            if iteration % 100 == 0:
                print(f"\nIteration {iteration} losses:")
                print(f"rec_loss: {rec_loss.item():.6f}")
                print(f"inv_mean: {inv_mean.item():.6f}")
                print(f"inv_var: {inv_var.item():.6f}")
                print(f"outer_loss: {outer_loss.item():.6f}")
            
            rec_optimizer.zero_grad()
            outer_loss.backward()
            
            # Check gradients periodically
            if iteration % 10 == 0:
                print("\nGradient norms:")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm(2).item()
                        print(f"{name}: {grad_norm:.6f}")
            
            rec_optimizer.step()
            total_outer_loss += outer_loss.item()

            if epoch % args.epup == 0 and iteration > 10 and iteration % args.itup == 0:
                # Debug augmenter update
                rec_loss, inv_mean, inv_var = model(batch)
                inner_loss = - inv_var
                
                if iteration == 12:  # First augmenter update
                    print(f"\nAugmenter update:")
                    print(f"inner_loss: {inner_loss.item():.6f}")
                    print(f"inv_var: {inv_var.item():.6f}")
                
                aug_optimizer.zero_grad()
                inner_loss.backward()
                
                # Debug augmenter gradients
                if iteration == 12:
                    print("\nAugmenter gradient norms:")
                    for name, param in augmenter.named_parameters():
                        if param.grad is not None:
                            print(f"{name}: {param.grad.data.norm(2).item():.6f}")
                
                aug_optimizer.step()
                total_inner_loss += inner_loss.item()

            s += args.batch_size
            iteration += 1
            if iteration % get_print == 0:
                print("Train epoch:[{}/{}], iter:[{}/{}], outer_loss:[{:.6f}], inner_loss:[{:.6f}], time:[{:.2f}] min"
                .format(epoch + 1, args.epoch, 
                        iteration, total_iter, 
                        outer_loss.item(), 
                        inner_loss.item(),
                        (time() - train_s_t) / 60))

        train_e_t = time()
        print("-" * 100)
        print("start evluation ...")
        print("-" * 100)
        test_s_t = time()
        ret = test(model, user_dict, n_params)
        test_e_t = time()
        epoch_outer_loss = total_outer_loss  / total_iter
        epoch_inner_loss = total_inner_loss  / total_iter
        train_time = (train_e_t - train_s_t) / 60
        test_time = (test_e_t - test_s_t) / 60
        
        # Create a pretty table for metrics
        metrics_table = PrettyTable()
        metrics_table.field_names = ["Metric", "Value"]
        
        # Add all available metrics
        for k_idx, k in enumerate(eval(args.Ks)):
            metrics_table.add_row([f"Recall@{k}", f"{ret['recall'][k_idx]:.4f}"])
            metrics_table.add_row([f"NDCG@{k}", f"{ret['ndcg'][k_idx]:.4f}"])
            metrics_table.add_row([f"Precision@{k}", f"{ret['precision'][k_idx]:.4f}"])
            metrics_table.add_row([f"Hit_Ratio@{k}", f"{ret['hit_ratio'][k_idx]:.4f}"])
        
        metrics_table.add_row(["AUC", f"{ret['auc']:.4f}"])
        metrics_table.add_row(["Train Time (min)", f"{train_time:.2f}"])
        metrics_table.add_row(["Test Time (min)", f"{test_time:.2f}"])
        metrics_table.add_row(["Outer Loss", f"{epoch_outer_loss:.6f}"])
        metrics_table.add_row(["Inner Loss", f"{epoch_inner_loss:.6f}"])
        
        print("\nEvaluation Metrics:")
        print(metrics_table)
        print("-" * 100)
        
        # Keep the original print for consistency
        recall_20 = ret['recall'][0]
        ndcg_20 = ret['ndcg'][0]
        print("Test epoch:[{}/{}], loss:[out:{:.6f}, in:{:.6f}], recall:[{:.4f}], ndcg:[{:.4f}] | train time:[{:.2f}] min, test time:[{:.2f}] min"
            .format(epoch + 1, args.epoch, 
                    epoch_outer_loss, epoch_inner_loss,
                    recall_20, ndcg_20,
                    train_time, test_time))
        print("-" * 100)
        