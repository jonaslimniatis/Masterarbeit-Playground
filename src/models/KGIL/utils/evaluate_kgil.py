from .metrics import *
from .parser import parse_args

import torch
import numpy as np
import multiprocessing
import heapq
from time import time
import pdb

cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)
device = torch.device("cuda:" + str(args.gpu_id)) if (args.cuda and torch.cuda.is_available()) else torch.device("cpu")
BATCH_SIZE = args.test_batch_size
batch_test_flag = args.batch_test_flag


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = AUC(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x):
    # Unpack the input tuple to include the necessary sets
    rating, u, train_user_set, test_user_set, n_items = x
    
    # user u's ratings for user u
    rating = rating
    # uid
    u = u
    # user u's items in the training set
    try:
        training_items = train_user_set[u]
    except Exception:
        training_items = []
    # user u's items in the test set
    user_pos_test = test_user_set[u]

    all_items = set(range(0, n_items))

    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


from .metrics import *
from .parser import parse_args

import torch
import numpy as np
import multiprocessing
import heapq
import pandas as pd
from time import time
import uuid

cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)
device = torch.device("cuda:" + str(args.gpu_id)) if (args.cuda and torch.cuda.is_available()) else torch.device("cpu")
BATCH_SIZE = args.test_batch_size
batch_test_flag = args.batch_test_flag

# … (keep all your helper funcs: ranklist_by_heapq, get_auc, etc.) …

def test(model, user_dict, n_params):
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'auc': 0.}
    
    # NEW: prepare list for collecting predictions
    predictions = []
    
    n_items = n_params['n_items']
    train_user_set = user_dict['train_user_set']
    test_user_set  = user_dict['test_user_set']

    pool = multiprocessing.Pool(cores)
    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)
    n_user_batches = n_test_users // BATCH_SIZE + 1

    # get all embeddings once
    entity_gcn_emb, user_gcn_emb = model.generate()

    for batch_id in range(n_user_batches):
        start = batch_id * BATCH_SIZE
        end   = min((batch_id + 1) * BATCH_SIZE, n_test_users)
        user_list_batch = test_users[start:end]
        user_tensor = torch.LongTensor(user_list_batch).to(device)
        u_emb_batch = user_gcn_emb[user_tensor]

        # compute the score matrix [batch_size × n_items]
        if batch_test_flag:
            # batched over items
            rate_batch = np.zeros((len(user_list_batch), n_items))
            for i_start in range(0, n_items, BATCH_SIZE):
                i_end = min(i_start + BATCH_SIZE, n_items)
                item_tensor = torch.arange(i_start, i_end, dtype=torch.long).to(device)
                i_emb = entity_gcn_emb[item_tensor]
                sub_scores = model.rating(u_emb_batch, i_emb).detach().cpu().numpy()
                rate_batch[:, i_start:i_end] = sub_scores
        else:
            # all-items at once
            item_tensor = torch.arange(0, n_items, dtype=torch.long).to(device)
            i_emb = entity_gcn_emb[item_tensor]
            rate_batch = model.rating(u_emb_batch, i_emb).detach().cpu().numpy()

        # Added by Jonas Limniatis
        # collect per-(user,item) predictions + actual
        for row_idx, uid in enumerate(user_list_batch):
            actual_set = test_user_set[uid]
            for item_id, pred in enumerate(rate_batch[row_idx]):
                actual = 1 if item_id in actual_set else 0
                predictions.append((uid, item_id, float(pred), actual))

        # now do your normal per-user metric aggregation via multiprocessing
        user_args = [
            (rate_batch[row_idx], uid, train_user_set, test_user_set, n_items)
            for row_idx, uid in enumerate(user_list_batch)
        ]
        batch_results = pool.map(test_one_user, user_args)

        for res in batch_results:
            result['precision']  += res['precision']  / n_test_users
            result['recall']     += res['recall']     / n_test_users
            result['ndcg']       += res['ndcg']       / n_test_users
            result['hit_ratio']  += res['hit_ratio']  / n_test_users
            result['auc']        += res['auc']        / n_test_users

    pool.close()

    # Added by Jonas Limniatis
    # convert to DataFrame and save
    df_preds = pd.DataFrame(predictions,
                            columns=['user_id','item_id','prediction','actual'])
    #df_preds.to_csv("all_predictions.csv", index=False)
    #print("Saved predictions to all_predictions.csv")
    
    # optionally return the DataFrame as well
    return result, df_preds
