import numpy as np
from sklearn.metrics import roc_auc_score
import Levenshtein
from sentence_transformers import SentenceTransformer
import heapq
import pandas as pd

def recall(rank, ground_truth, N):
    return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def average_precision(r,cut):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Returns:
        Average precision
    """
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(cut) if r[k]]
    if not out:
        return 0.
    return np.sum(out)/float(min(cut, np.sum(r)))


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, ground_truth, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain

        Low but correct defination
    """
    GT = set(ground_truth)
    if len(GT) > k :
        sent_list = [1.0] * k
    else:
        sent_list = [1.0]*len(GT) + [0.0]*(k-len(GT))
    dcg_max = dcg_at_k(sent_list, k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def recall_at_k(r, k, all_pos_num):
    r = np.asarray(r)[:k]
    return np.sum(r) / all_pos_num


def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.

def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.

def AUC(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res

#---Added by Jonas Limniatis---
def collect_unique_values(df):
    unique_values = []
    for column in ['head', 'tail']:  # Only consider head and tail columns where entities are stored
        if column in df.columns:
            # Convert to string and filter out nans
            values = df[column].dropna().astype(str).unique().tolist()
            unique_values.extend(values)
    
    # Filter parent ASINs (product IDs that start with 'B')
    unique_values = [value for value in unique_values if isinstance(value, str) and not value.startswith('B')]
    
    # Only unique values
    unique_values = list(set(unique_values))
    return unique_values

def levenshtein_distance(df1, df2):
    """Calculate normalized Levenshtein distance between two sets of taxonomy terms"""
    
    # Get unique values from each DataFrame (excluding product IDs)
    unique_values_df1 = collect_unique_values(df1)
    unique_values_df2 = collect_unique_values(df2)
    
    
    # Calculate Levenshtein ratio for each pair of terms
    total_similarity = 0
    valid_comparisons = 0
    
    for term1 in unique_values_df1:
        for term2 in unique_values_df2:
            # Skip if either term is not a valid string
            if not isinstance(term1, str) or not isinstance(term2, str):
                continue
                
            lev_ratio = Levenshtein.ratio(term1.lower(), term2.lower())
            total_similarity += lev_ratio
            valid_comparisons += 1
    
    # Avoid division by zero
    if valid_comparisons == 0:
        return 0
        
    avg_lev_ratio = total_similarity / valid_comparisons
    return avg_lev_ratio

#---Added by Jonas Limniatis---
# Fix the cosine_similarity function to handle matrix inputs correctly
def cosine_similarity_matrices(matrix1, matrix2):
    # Transpose the second matrix to align dimensions for dot product
    matrix2_transposed = matrix2.T
    # Calculate dot product between matrices
    dot_product = np.dot(matrix1, matrix2_transposed)
    
    # Calculate norms for each row in both matrices
    norm_matrix1 = np.linalg.norm(matrix1, axis=1, keepdims=True)
    norm_matrix2 = np.linalg.norm(matrix2, axis=1, keepdims=True)
    
    # Calculate cosine similarity
    cosine_sim = dot_product / (np.dot(norm_matrix1, norm_matrix2.T))
    
    return cosine_sim

# Update the calc_cosine_similarity function to use the correct similarity function
def calc_cosine_similarity_fixed(df1, df2, embedding_model):
    def collect_unique_values(df):
        """Collect unique non-ASIN values from head and tail columns."""
        unique_values = []
        for column in ['head', 'tail']:
            if column in df.columns:
                # Drop NaN values, convert to strings, then get unique values
                values = df[column].dropna().astype(str).unique().tolist()
                # Filter out ASINs (product IDs starting with 'B')
                values = [val for val in values if not val.startswith('B')]
                unique_values.extend(values)
        
        # Remove duplicates
        unique_values = list(set(unique_values))
        return unique_values
    
    # Get unique values from each dataframe
    unique_values_df1 = collect_unique_values(df1)
    unique_values_df2 = collect_unique_values(df2)
    
    
    # Skip if either list is empty
    if not unique_values_df1 or not unique_values_df2:
        print("Warning: One or both term lists are empty, returning 0")
        return 0

    # Embed unique values
    unique_values_df1_embed = embedding_model.encode(unique_values_df1)
    unique_values_df2_embed = embedding_model.encode(unique_values_df2)

    # Use the fixed cosine similarity function for matrices
    cosine_similarity_matrix = cosine_similarity_matrices(unique_values_df1_embed, unique_values_df2_embed).mean()

    return cosine_similarity_matrix

def gini_coefficient(x, name):
    sorted_x = np.sort(x)
    n = len(x)
    cumulative_x = np.cumsum(sorted_x)
    gini = (n + 1 - 2 * np.sum(cumulative_x) / cumulative_x[-1]) / n
    print(f"Gini score {name}: {gini:.2f}")
    return gini
def evaluate_random_baseline(predictions_df: pd.DataFrame, k_value: int = 20) -> dict:
    """
    Evaluates a random baseline based on coin toss predictions.

    Args:
        predictions_df (pd.DataFrame): DataFrame with columns
                                       ['user_id', 'item_id', 'actual', 'random_value'].
                                       'actual' is 1 if true interaction, 0 otherwise.
                                       'random_value' is 0 or 1 from coin_toss, used for ranking.
        k_value (int): The K for top-K metrics (e.g., 20).

    Returns:
        dict: A dictionary containing aggregated metrics for the specified K:
              {'recall@K': float, 'precision@K': float,
               'ndcg@K': float, 'hit_ratio@K': float}
    """
    
    required_columns = ['user_id', 'item_id', 'actual', 'random_value']
    if not all(col in predictions_df.columns for col in required_columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")

    aggregated_metrics = {
        f'recall@{k_value}': 0.0,
        f'precision@{k_value}': 0.0,
        f'ndcg@{k_value}': 0.0,
        f'hit_ratio@{k_value}': 0.0
    }

    test_users = predictions_df['user_id'].unique()
    n_test_users = len(test_users)

    if n_test_users == 0:
        print("No test users found in the predictions DataFrame.")
        return aggregated_metrics

    for user_id in test_users:
        user_data = predictions_df[predictions_df['user_id'] == user_id].copy() # Use .copy() to avoid SettingWithCopyWarning
        
        # Actual positive items for the user
        user_pos_test_items = set(user_data[user_data['actual'] == 1]['item_id'])
        num_user_pos_test_items = len(user_pos_test_items)
        
        # Create item_score dictionary: item_id -> random_value
        # Items with random_value=1 will be ranked higher.
        # Add a very small random number for tie-breaking among items with the same random_value (0 or 1)
        # This ensures a consistent, though still random, ranking for tie-broken items.
        user_data['tie_breaker_score'] = user_data['random_value'] + np.random.uniform(0, 0.00001, size=len(user_data))
        item_scores = pd.Series(user_data['tie_breaker_score'].values, index=user_data['item_id']).to_dict()

        if not item_scores:
            continue # Should not happen if user_data is not empty

        # --- Ranking based on random_value (with tie-breaking) ---
        # heapq.nlargest will rank items with score 1 (plus small random) before items with score 0 (plus small random).
        num_items_to_rank = min(k_value, len(item_scores))
        ranked_item_ids = heapq.nlargest(num_items_to_rank,
                                         item_scores, 
                                         key=item_scores.get)
        
        # Create the 'r' list for metrics: 1 if recommended item is in positive set, 0 otherwise
        r_list_for_metrics = []
        for item_id_in_ranking in ranked_item_ids:
            if item_id_in_ranking in user_pos_test_items:
                r_list_for_metrics.append(1)
            else:
                r_list_for_metrics.append(0)
        
        # Pad r_list_for_metrics with 0s if fewer than k_value items were ranked
        # (e.g., if the user had fewer than k_value items associated with them in total)
        while len(r_list_for_metrics) < k_value:
            r_list_for_metrics.append(0)

        # --- Calculate performance for this user ---
        aggregated_metrics[f'precision@{k_value}'] += precision_at_k(r_list_for_metrics, k_value)
        aggregated_metrics[f'recall@{k_value}'] += recall_at_k(r_list_for_metrics, k_value, num_user_pos_test_items)
        aggregated_metrics[f'ndcg@{k_value}'] += ndcg_at_k(r_list_for_metrics, k_value, num_user_pos_test_items)
        aggregated_metrics[f'hit_ratio@{k_value}'] += hit_at_k(r_list_for_metrics, k_value)
            
    # Average the results over all users
    if n_test_users > 0:
        for key in aggregated_metrics:
            aggregated_metrics[key] /= n_test_users
    
    return aggregated_metrics