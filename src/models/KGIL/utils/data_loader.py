import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp
import os # Import the os module for path joining
import random
from time import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_nodes = 0
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)


def read_cf(file_name):
    # Check if file exists before opening
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Interaction file not found: {file_name}")
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        # Split by tab instead of comma and take only user_id and item_id
        parts = tmps.split()
        u_id = int(parts[0])
        i_id = int(parts[1])
        inter_mat.append([u_id, i_id])

    return np.array(inter_mat)


def remap_item(train_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))


def read_triplets(file_name):
    # Check if file exists before opening
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"KG file not found: {file_name}")
    global n_entities, n_relations, n_nodes

    # Handle different file formats based on extension
    file_ext = os.path.splitext(file_name)[1].lower()
    
    if file_ext == '.csv':
        # Load from CSV format (header expected)
        try:
            import pandas as pd
            df = pd.read_csv(file_name)
            # Ensure we have the right columns (head, relation, tail)
            if not all(col in df.columns for col in ['head', 'relation', 'tail']):
                # Try alternative names
                if all(col in df.columns for col in ['hypernym', 'relation', 'hyponym']):
                    df = df.rename(columns={'hypernym': 'head', 'hyponym': 'tail'})
                else:
                    raise ValueError(f"CSV file {file_name} doesn't have expected columns")
            
            # Convert string values to integers if needed
            can_triplets_list = []
            # Create entity and relation mappings
            entity_map = {}
            relation_map = {}
            entity_counter = 0
            relation_counter = 0
            
            for _, row in df.iterrows():
                # Map entities and relations to integers
                head = row['head']
                relation = row['relation']
                tail = row['tail']
                
                # Skip rows with missing values
                if pd.isna(head) or pd.isna(relation) or pd.isna(tail):
                    continue
                
                # Convert to string just in case
                head = str(head)
                relation = str(relation)
                tail = str(tail)
                
                # Map to integers
                if head not in entity_map:
                    entity_map[head] = entity_counter
                    entity_counter += 1
                if tail not in entity_map:
                    entity_map[tail] = entity_counter
                    entity_counter += 1
                if relation not in relation_map:
                    relation_map[relation] = relation_counter
                    relation_counter += 1
                    
                # Add to triplets
                can_triplets_list.append([entity_map[head], relation_map[relation], entity_map[tail]])
            
            can_triplets_np = np.array(can_triplets_list, dtype=np.int32)
            print(f"Loaded {len(can_triplets_np)} triplets from CSV file {file_name}")
            print(f"Found {entity_counter} entities and {relation_counter} relations")
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            raise
    else:
        # Default to txt format (typical kg_final.txt format)
        can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
        
    can_triplets_np = np.unique(can_triplets_np, axis=0)

    if args.inverse_r:
        # get triplets with inverse direction like <entity, is-aspect-of, item>
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        # consider two additional relations --- 'interact' and 'be interacted'
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
        # get full version of knowledge graph
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
    else:
        # consider two additional relations --- 'interact'.
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        triplets = can_triplets_np.copy()

    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1  # including items + users
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets


def build_graph(train_data, triplets):
    ckg_graph = nx.MultiDiGraph()
    print("Begin to build graph ...", ckg_graph)
    rd = defaultdict(list)

    print("Begin to load interaction triples ...")
    for u_id, i_id in tqdm(train_data, ascii=True):
        rd[0].append([u_id, i_id])

    print("\nBegin to load knowledge graph triples ...")
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):
        ckg_graph.add_edge(h_id, t_id, key=r_id)
        rd[r_id].append([h_id, t_id])

    print("End to build graph ...", ckg_graph)

    return ckg_graph, rd

def build_sparse_relational_graph(relation_dict):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    adj_mat_list = []
    print("Begin to build sparse relation matrix ...")
    for r_id in tqdm(relation_dict.keys()):
        np_mat = np.array(relation_dict[r_id])
        if r_id == 0:
            cf = np_mat.copy()
            cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
            vals = [1.] * len(cf)
            adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))
        else:
            vals = [1.] * len(np_mat)
            adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(n_nodes, n_nodes))
        adj_mat_list.append(adj)

    norm_mat_list = [_bi_norm_lap(mat) for mat in adj_mat_list]
    mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]
    # interaction: user->item, [n_users, n_entities]
    norm_mat_list[0] = norm_mat_list[0].tocsr()[:n_users, n_users:].tocoo()
    mean_mat_list[0] = mean_mat_list[0].tocsr()[:n_users, n_users:].tocoo()
    return adj_mat_list, norm_mat_list, mean_mat_list

def load_data(model_args):
    global args
    args = model_args
    
    # Construct paths to training and testing interaction files
    train_file = os.path.join(args.data_path, 'train_adj.txt')
    test_file = os.path.join(args.data_path, 'test_adj.txt')
    
    # Default KG file is in the same directory as train/test files
    kg_file = os.path.join(args.data_path, 'kg_final.txt')
    
    # Try to use model_technique-specific relations if available
    if hasattr(args, 'model_technique'):
        # Replace slashes with underscores for filesystem paths
        model_technique = args.model_technique.replace('/', '_')
        
        # Try different potential KG file locations and formats
        # First try the taxonomy directory with model-specific file
        taxonomy_dir = os.path.join("data", "taxonomy", args.dataset)
        potential_kg_files = [
            # Try model-specific relations file in taxonomy dir
            os.path.join(taxonomy_dir, f"{model_technique}_relations.csv"),
            # Try hearst patterns as fallback in taxonomy dir
            os.path.join(taxonomy_dir, "hearst_patterns_relations.csv"),
            # Try baseline relations as fallback in taxonomy dir
            os.path.join(taxonomy_dir, "baseline_relations.csv"),
            # Try in experiments directory with hearst patterns
            os.path.join(args.data_path, "hearst_patterns_relations.csv"),
            # Default location in the experiment directory
            os.path.join(args.data_path, "kg_final.txt")
        ]
        
        # Use the first file that exists
        for potential_file in potential_kg_files:
            if os.path.exists(potential_file):
                kg_file = potential_file
                break
    
    print(f'Train file: {train_file}')
    print(f'Test file: {test_file}')
    print(f'KG file: {kg_file}')
    
    if not os.path.exists(kg_file):
        raise FileNotFoundError(f"No KG file found at any of the expected locations. Last tried: {kg_file}")

    print('reading train and test user-item set ...')
    train_cf = read_cf(train_file)
    test_cf = read_cf(test_file)
    remap_item(train_cf, test_cf)

    print('combinating train_cf and kg data ...')
    triplets = read_triplets(kg_file)

    print('building the graph ...')
    graph, relation_dict = build_graph(train_cf, triplets)

    print('building the adj mat ...')
    adj_mat_list, norm_mat_list, mean_mat_list = build_sparse_relational_graph(relation_dict)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations)
    }
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set
    }

    return train_cf, test_cf, user_dict, n_params, graph, \
           [adj_mat_list, norm_mat_list, mean_mat_list]
