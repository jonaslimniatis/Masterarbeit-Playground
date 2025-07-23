from datasets import load_dataset
import re
import pandas as pd
import numpy as np
import os
import json # For saving maps if needed
import networkx as nx
import itertools
from collections import defaultdict  # Add this import at the top
import matplotlib.pyplot as plt

class Preprocess_Amazon_Data:
    def __init__(self, dataset_name, metadataset_name, core_setting):
        self.dataset_name = dataset_name
        self.metadataset_name = metadataset_name
        self.core_setting = core_setting
    
    def load_data(self):

        metadata_df = load_dataset("McAuley-Lab/Amazon-Reviews-2023", self.metadataset_name)
        dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", self.dataset_name)

        train_df = pd.DataFrame(dataset["train"])
        test_df = pd.DataFrame(dataset["test"])
        meta_df = pd.DataFrame(metadata_df["full"])

        # drop duplicates 
        meta_df = meta_df.drop_duplicates(subset=['parent_asin'])
        
        train_test_df = pd.concat([train_df, test_df], ignore_index=True)
        user_value_count = train_test_df['user_id'].value_counts()
        item_value_count = train_test_df['parent_asin'].value_counts()
        train_test_df = train_test_df[train_test_df['user_id'].isin(user_value_count[user_value_count >= self.core_setting].index)]

        print("Filtered out % of users: ", (len(user_value_count.user_id.unique()) - len(train_test_df['user_id'].unique()))/len(user_value_count.user_id.unique())*100, "%")
        print("Filtered out % of items: ", (len(item_value_count) - len(train_test_df['parent_asin'].unique()))/len(item_value_count)*100, "%")
        
        train_df = train_df[train_df.parent_asin.isin(train_test_df.parent_asin)]
        test_df = test_df[test_df.parent_asin.isin(train_test_df.parent_asin)]
        
        # filter out users with less than 20 ratings
        train_df = train_df[train_df['user_id'].isin(user_value_count[user_value_count >= self.core_setting].index)]
        test_df = test_df[test_df['user_id'].isin(user_value_count[user_value_count >= self.core_setting].index)]

        # filter metadata parent_asin which is in train_df or test_df
        metadata_filtered_df = meta_df[meta_df.parent_asin.isin(train_df.parent_asin) | meta_df.parent_asin.isin(test_df.parent_asin)]

        

        print("The meta dataset is reduced by ",(len(metadata_filtered_df)-len(meta_df))/len(meta_df)*100,"%", " reviews, because of 5core rating")

        return train_df, test_df, metadata_filtered_df
    
    def clean_string(self, string):
        string = re.sub(r'\[', '', string)
        string = re.sub(r'\]', '', string)
        string = re.sub(r'"', '', string)
        string = re.sub(r'\s+', ' ', string)
        string = re.sub("{", "", string)
        string = re.sub("}", "", string)
        return string
    
    def clean_column(self, df, column_name):
        def safe_clean(value):
            if isinstance(value, list):
                return self.clean_string(str(value))
            elif pd.isna(value) or value is None:
                return ""
            else:
                return self.clean_string(str(value))
        return df[column_name].apply(safe_clean)
    
    def create_source_text(self, product):
        """Concatenate product information into a text string

        Args:
            product (dict): dictionary containing product information

        Returns:
            str: concatenated product information
        """
        # TODO: update columns for other datasets
        description = "*" if product["description"] == "" else f"Description: {product['description']}"
        features = "*" if product["features"] == "" else f"Features: {product['features']}"
        details = "*" if product["details"] == "" else f"Details: {product['details']}"
        store = "*" if product["store"] == "" else f"Store: {product['store']}"
        categories = "*" if product["categories"] == "" else f"Categories: {product['categories']}"
        price = "*" if product["price"] == "" else f"Price: {product['price']}"
        author = "*" if product["author"] == "" else f"Author: {product['author']}"

        concatenated_text = f"Parent ASIN: {product['parent_asin']}; Title: {product['title']}; Author: {author}; Description: {description}; Features: {features} - {details}; Store: {store}; Categories: {categories}; Price: {price};"
        return concatenated_text
    
    def clean_text_columns(self,df, columns):
        """clean the text columns that might contain lists or dictionaries. 
        Concatenate the text columns into a single column called "source_text"
        Args:
            df (pd.DataFrame): The dataframe to clean
            columns (list): The list of columns to clean
        Returns:
            pd.DataFrame: The cleaned dataframe with the new "source_text" column
        """
        for column in columns:
            if column in df.columns:
                df[column] = self.clean_column(df, column)

        df['source_text'] = df.apply(self.create_source_text, axis=1)
        return df


class Build_KG:

    def __init__(self):
        pass
    def filter_entities(train_df, test_df, taxonomy_df):
        unique_train_entities = set(train_df['parent_asin'].unique())
        unique_test_entities = set(test_df['parent_asin'].unique())
        unique_entities = unique_train_entities.union(unique_test_entities)
        unique_entities_tax = {entity for entity in taxonomy_df["head"] if isinstance(entity, str) and (entity.startswith('B0') or entity.startswith('A'))}
        print(f"ASIN-like entities found in KG triples: {len(unique_entities_tax)}")
        # filter out entities which are not in intersection of unique_entities and unique_entities_tax
        no_intersection_asin = unique_entities_tax - unique_entities
        print(f"ASIN-like entities with no intersection with unique_entities: {len(no_intersection_asin)}")
        taxonomy_df = taxonomy_df[~taxonomy_df["head"].isin(no_intersection_asin)]
        print(f"Number of KG triples after removing ASIN-like entities with no intersection: {len(taxonomy_df)}")
        return taxonomy_df

    def user_item_triples(self, df: pd.DataFrame):
        """Converts user-item interaction df to triples format ('head', 'relation', 'tail')."""
        triples = []
        # TODO: for other datasets, update columns
        if "user_id" not in df.columns or "parent_asin" not in df.columns:
            print("Warning: Input DataFrame to user_item_triples is missing 'user_id' or 'parent_asin'.")
            return pd.DataFrame(columns=["head", "relation", "tail"])

        for _, row in df.iterrows():
            # Ensure consistent string types for mapping later
            triples.append((str(row["user_id"]), "rated", str(row["parent_asin"])))

        user_item_df = pd.DataFrame(triples, columns=["head", "relation", "tail"])
        return user_item_df
    def build_kg_without_taxonomy(self, train_df: pd.DataFrame, test_df: pd.DataFrame, out_path: str):
        """
        Builds the KG without taxonomy.
        """
        print(f"Starting build_kg. Output path: {out_path}")
        os.makedirs(out_path, exist_ok=True) # Ensure output directory exists early

        product_topic_df = pd.DataFrame(columns=['head', 'relation', 'tail'])

        for df, name in [(product_topic_df, 'product_topic_df'), (train_df, 'train_df'), (test_df, 'test_df')]:
            if not all(col in df.columns for col in ['head', 'relation', 'tail']):
                print(f"Warning: {name} is missing required columns ('head', 'relation', 'tail'). Check input data.")
                # Handle appropriately, e.g., skip concatenation or raise error if critical
                if name == 'product_topic_df': # Allow product topics to be missing
                     product_topic_df = pd.DataFrame(columns=['head', 'relation', 'tail'])
                else: # Train/Test data are critical
                     raise ValueError(f"{name} must have 'head', 'relation', 'tail' columns.")
                
        kg_all_triples = pd.concat([train_df, test_df, product_topic_df], ignore_index=True)
        kg_all_triples['head'] = kg_all_triples['head'].astype(str)
        kg_all_triples['tail'] = kg_all_triples['tail'].astype(str)
        kg_all_triples['relation'] = kg_all_triples['relation'].astype(str)
        kg_all_triples = kg_all_triples.drop_duplicates().reset_index(drop=True)
        print(f"Combined KG has {len(kg_all_triples)} unique triples.")

        # --- 2. Create Separate Mappings ---
        all_entities = pd.unique(kg_all_triples[['head', 'tail']].values.ravel('K'))
        all_relations = pd.unique(kg_all_triples['relation'])

        # --- 3. Map Train/Test/KG DataFrames using .map() ---
        entity_to_id_map = {name: i for i, name in enumerate(all_entities)}
        relation_to_id_map = {name: i for i, name in enumerate(all_relations)}
        print(f"Found {len(entity_to_id_map)} unique entities.")
        print(f"Found {len(relation_to_id_map)} unique relations.")

        # --- 3. Map Train/Test/KG DataFrames using .map() ---
        # Map original train/test dfs (ensure they are also string type first)
        train_df['head'] = train_df['head'].astype(str)
        train_df['tail'] = train_df['tail'].astype(str)
        test_df['head'] = test_df['head'].astype(str)
        test_df['tail'] = test_df['tail'].astype(str)

        train_head_ids = train_df['head'].map(entity_to_id_map)
        train_tail_ids = train_df['tail'].map(entity_to_id_map)
        test_head_ids = test_df['head'].map(entity_to_id_map)
        test_tail_ids = test_df['tail'].map(entity_to_id_map)

        # Create mapped dataframes, drop rows where mapping failed (NaNs), convert to int
        train_df_mapped = pd.DataFrame({'head_id': train_head_ids, 'tail_id': train_tail_ids}).dropna().astype(int)
        test_df_mapped = pd.DataFrame({'head_id': test_head_ids, 'tail_id': test_tail_ids}).dropna().astype(int)
        print(f"Mapped {len(train_df_mapped)} train interactions.")
        print(f"Mapped {len(test_df_mapped)} test interactions.")

         # Map the combined KG dataframe
        kg_df_mapped = kg_all_triples.copy()
        kg_df_mapped['head'] = kg_df_mapped['head'].map(entity_to_id_map)
        kg_df_mapped['tail'] = kg_df_mapped['tail'].map(entity_to_id_map)
        kg_df_mapped['relation'] = kg_df_mapped['relation'].map(relation_to_id_map)
        kg_df_mapped.dropna(inplace=True)
        kg_df_mapped = kg_df_mapped.astype(int)
        print(f"Mapped KG has {len(kg_df_mapped)} triples.")

        #exclude t

        
        # filter out relevant columns
        kg_df_mapped = kg_df_mapped[['head', 'relation', 'tail']]

        # --- 4. Write Train/Test Interaction Files ---
        train_adj_path = os.path.join(out_path, "train_adj.txt")
        test_adj_path = os.path.join(out_path, "test_adj.txt")

        print(f"Writing training interactions to: {train_adj_path}")
        train_adj_lines = [] # Store lines for return value if needed
        with open(train_adj_path, "w") as f_train:
            # Group by mapped user ID ('head_id')
            for user_id, group in train_df_mapped.groupby('head_id'):
                items = sorted(group["tail_id"].tolist()) # Sort items for consistency
                items_str = ' '.join(map(str, items))
                line = f"{user_id} {items_str}\n"
                f_train.write(line)
                train_adj_lines.append(line.strip()) # Store line without newline

        print(f"Writing testing interactions to: {test_adj_path}")
        test_adj_lines = [] # Store lines for return value if needed
        with open(test_adj_path, "w") as f_test:
            # Group by mapped user ID ('head_id')
            for user_id, group in test_df_mapped.groupby('head_id'):
                items = sorted(group["tail_id"].tolist()) # Sort items for consistency
                items_str = ' '.join(map(str, items))
                line = f"{user_id} {items_str}\n"
                f_test.write(line)
                test_adj_lines.append(line.strip()) # Store line without newline

        # --- 5. Write Mapped KG File ---
        kg_final_path = os.path.join(out_path, "kg_final.txt")
        print(f"Writing mapped KG triples to: {kg_final_path}")
        kg_df_mapped.to_csv(kg_final_path, sep='\t', index=False, header=False)

        # --- 6. Save Mappings (Optional but Recommended) ---
        pd.DataFrame(entity_to_id_map.items(), columns=['entity', 'id']).sort_values('id').to_csv(
            os.path.join(out_path, 'entity_map.csv'), index=False
        )
        pd.DataFrame(relation_to_id_map.items(), columns=['relation', 'id']).sort_values('id').to_csv(
            os.path.join(out_path, 'relation_map.csv'), index=False
        )
        print(f"Saved mapping files to {out_path}")

        # Return the mapped KG df, the interaction lines (as lists), and the entity map
        # Note: Returning train_adj_lines/test_adj_lines might not be necessary if files are written correctly.
        # Returning the entity_map is useful.
        return kg_df_mapped, train_adj_lines, test_adj_lines, entity_to_id_map

    def build_kg(self, product_topic_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame, out_path: str):
        """
        Builds the full KG, creates entity/relation maps, and writes
        train/test interaction files in 'user item1 item2...' format.
        Also saves the mapped KG triples.
        """
        print(f"Starting build_kg. Output path: {out_path}")
        os.makedirs(out_path, exist_ok=True) # Ensure output directory exists early

        # --- 1. Combine all data & ensure string types ---
        # Rename product_topic columns if necessary (example)
        product_topic_df = product_topic_df.rename(columns={'hypernym': 'head', 'hyponym': 'tail'}, errors='ignore')
        
        # Ensure required columns exist, provide defaults if not critical
        for df, name in [(product_topic_df, 'product_topic_df'), (train_df, 'train_df'), (test_df, 'test_df')]:
            if not all(col in df.columns for col in ['head', 'relation', 'tail']):
                print(f"Warning: {name} is missing required columns ('head', 'relation', 'tail'). Check input data.")
                # Handle appropriately, e.g., skip concatenation or raise error if critical
                if name == 'product_topic_df': # Allow product topics to be missing
                     product_topic_df = pd.DataFrame(columns=['head', 'relation', 'tail'])
                else: # Train/Test data are critical
                     raise ValueError(f"{name} must have 'head', 'relation', 'tail' columns.")

        kg_all_triples = pd.concat([train_df, test_df, product_topic_df], ignore_index=True)
        

        # Convert all identifiers to strings BEFORE finding unique values
        kg_all_triples['head'] = kg_all_triples['head'].astype(str)
        kg_all_triples['tail'] = kg_all_triples['tail'].astype(str)
        kg_all_triples['relation'] = kg_all_triples['relation'].astype(str)
        kg_all_triples = kg_all_triples.drop_duplicates().reset_index(drop=True)
        print(f"Combined KG has {len(kg_all_triples)} unique triples.")

        # --- 2. Create Separate Mappings ---
        all_entities = pd.unique(kg_all_triples[['head', 'tail']].values.ravel('K'))
        all_relations = pd.unique(kg_all_triples['relation'])

        # Filter out potential None or NaN representations if necessary before creating map
        all_entities = [e for e in all_entities if pd.notna(e)]
        all_relations = [r for r in all_relations if pd.notna(r)]

        entity_to_id_map = {name: i for i, name in enumerate(all_entities)}
        relation_to_id_map = {name: i for i, name in enumerate(all_relations)}
        print(f"Found {len(entity_to_id_map)} unique entities.")
        print(f"Found {len(relation_to_id_map)} unique relations.")

        # --- 3. Map Train/Test/KG DataFrames using .map() ---
        # Map original train/test dfs (ensure they are also string type first)
        train_df['head'] = train_df['head'].astype(str)
        train_df['tail'] = train_df['tail'].astype(str)
        test_df['head'] = test_df['head'].astype(str)
        test_df['tail'] = test_df['tail'].astype(str)

        train_head_ids = train_df['head'].map(entity_to_id_map)
        train_tail_ids = train_df['tail'].map(entity_to_id_map)
        test_head_ids = test_df['head'].map(entity_to_id_map)
        test_tail_ids = test_df['tail'].map(entity_to_id_map)

        # Create mapped dataframes, drop rows where mapping failed (NaNs), convert to int
        train_df_mapped = pd.DataFrame({'head_id': train_head_ids, 'tail_id': train_tail_ids}).dropna().astype(int)
        test_df_mapped = pd.DataFrame({'head_id': test_head_ids, 'tail_id': test_tail_ids}).dropna().astype(int)
        print(f"Mapped {len(train_df_mapped)} train interactions.")
        print(f"Mapped {len(test_df_mapped)} test interactions.")

        # Map the combined KG dataframe
        kg_df_mapped = kg_all_triples.copy()
        kg_df_mapped['head'] = kg_df_mapped['head'].map(entity_to_id_map)
        kg_df_mapped['tail'] = kg_df_mapped['tail'].map(entity_to_id_map)
        kg_df_mapped['relation'] = kg_df_mapped['relation'].map(relation_to_id_map)
        kg_df_mapped.dropna(inplace=True)
        kg_df_mapped = kg_df_mapped.astype(int)
        print(f"Mapped KG has {len(kg_df_mapped)} triples.")

        
        # filter out relevant columns
        kg_df_mapped = kg_df_mapped[['head', 'relation', 'tail']]

        # --- 4. Write Train/Test Interaction Files ---
        train_adj_path = os.path.join(out_path, "train_adj.txt")
        test_adj_path = os.path.join(out_path, "test_adj.txt")

        print(f"Writing training interactions to: {train_adj_path}")
        train_adj_lines = [] # Store lines for return value if needed
        with open(train_adj_path, "w") as f_train:
            # Group by mapped user ID ('head_id')
            for user_id, group in train_df_mapped.groupby('head_id'):
                items = sorted(group["tail_id"].tolist()) # Sort items for consistency
                items_str = ' '.join(map(str, items))
                line = f"{user_id} {items_str}\n"
                f_train.write(line)
                train_adj_lines.append(line.strip()) # Store line without newline

        print(f"Writing testing interactions to: {test_adj_path}")
        test_adj_lines = [] # Store lines for return value if needed
        with open(test_adj_path, "w") as f_test:
            # Group by mapped user ID ('head_id')
            for user_id, group in test_df_mapped.groupby('head_id'):
                items = sorted(group["tail_id"].tolist()) # Sort items for consistency
                items_str = ' '.join(map(str, items))
                line = f"{user_id} {items_str}\n"
                f_test.write(line)
                test_adj_lines.append(line.strip()) # Store line without newline

        # --- 5. Write Mapped KG File ---
        kg_final_path = os.path.join(out_path, "kg_final.txt")
        print(f"Writing mapped KG triples to: {kg_final_path}")
        kg_df_mapped.to_csv(kg_final_path, sep='\t', index=False, header=False)

        # --- 6. Save Mappings (Optional but Recommended) ---
        pd.DataFrame(entity_to_id_map.items(), columns=['entity', 'id']).sort_values('id').to_csv(
            os.path.join(out_path, 'entity_map.csv'), index=False
        )
        pd.DataFrame(relation_to_id_map.items(), columns=['relation', 'id']).sort_values('id').to_csv(
            os.path.join(out_path, 'relation_map.csv'), index=False
        )
        print(f"Saved mapping files to {out_path}")

        return kg_df_mapped, train_adj_lines, test_adj_lines, entity_to_id_map

class KG_Filter:
    def __init__(self):
        pass

    def is_item_node(self,node_id: str, item_prefixes=("B0", "A0")) -> bool:
        """
        Determines if a node ID represents an item.
        Adapt this function based on how items are identified in your dataset.
        For example, Amazon Standard Identification Numbers (ASINs) often start with "B0" or "A0".

        Args:
            node_id (str): The node identifier.
            item_prefixes (tuple): A tuple of string prefixes that identify item nodes.

        Returns:
            bool: True if the node_id is considered an item, False otherwise.
        """
        if not isinstance(node_id, str):
            return False
        for prefix in item_prefixes:
            if node_id.startswith(prefix):
                return True
        return False

    def filter_self_loops(self,relations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters out self-loops from the relations DataFrame.
        
        Args:
            relations_df (pd.DataFrame): DataFrame with 'head' and 'tail' columns.
            
        Returns:
            pd.DataFrame: DataFrame with self-loops removed.
        """
        self_loops = relations_df[relations_df['head'] == relations_df['tail']]
        if not self_loops.empty:
            print(f"Found {len(self_loops)} self-loops in the taxonomy.")
            return relations_df[relations_df['head'] != relations_df['tail']].reset_index(drop=True)
        print("No self-loops found in the taxonomy.")
        return relations_df

    def filter_item_to_grandchild_redundancy(self,relations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters out 'item -> grandchild_category' relationships from a DataFrame
        if both 'item -> child_category' and 'child_category -> grandchild_category'
        relationships also exist within the same DataFrame.

    Args:
        relations_df (pd.DataFrame): DataFrame with 'head', 'relation', 'tail' columns.
                                     This DataFrame should contain all relevant links
                                     (item-to-category and category-to-category).

    Returns:
        pd.DataFrame: A new DataFrame with the identified redundant
                      item-to-grandchild_category relationships removed.
        """
        if not isinstance(relations_df, pd.DataFrame):
            raise TypeError("relations_df must be a pandas DataFrame.")
        if not all(col in relations_df.columns for col in ['head', 'tail']):
                raise ValueError("relations_df must contain 'head' and 'tail' columns.")

        # Remove self-loops
        relations_df = self.filter_self_loops(relations_df)

        # Create a directed graph for efficient path finding
        G = nx.DiGraph()
        for _, row in relations_df.iterrows():
            G.add_edge(str(row['head']), str(row['tail']))

        # For quick lookups
        item_to_direct_categories = defaultdict(set)
        category_to_direct_children = defaultdict(set)

        # Identify all unique nodes and classify them
        all_nodes = set(relations_df['head']).union(set(relations_df['tail']))
        potential_category_nodes = {node for node in all_nodes if not self.is_item_node(str(node))}

        # Populate lookup dictionaries
        for _, row in relations_df.iterrows():
            h, t = str(row['head']), str(row['tail'])
            if self.is_item_node(h) and t in potential_category_nodes:
                item_to_direct_categories[h].add(t)
            elif h in potential_category_nodes and t in potential_category_nodes:
                category_to_direct_children[h].add(t)

        indices_to_drop = set()

        # Check for redundant relationships
        for idx, row in relations_df.iterrows():
            item = str(row['head'])
            grandchild_cat = str(row['tail'])

            if not (self.is_item_node(item) and grandchild_cat in potential_category_nodes):
                continue

            # Check for paths of length 2 from item to grandchild_cat
            if item in item_to_direct_categories:
                for child_cat in item_to_direct_categories[item]:
                    if child_cat == grandchild_cat:
                        continue
                    
                    # Check if there's a path: item -> child_cat -> grandchild_cat
                    if nx.has_path(G, child_cat, grandchild_cat):
                        indices_to_drop.add(idx)
                        break

        if indices_to_drop:
            print(f"Identified {len(indices_to_drop)} redundant item-to-grandchild relationships to remove.")
            filtered_df = relations_df.drop(index=list(indices_to_drop)).reset_index(drop=True)
        else:
            print("No item-to-grandchild redundant relationships (item -> child -> grandchild pattern) found.")
            filtered_df = relations_df.copy()

        print(f"Original relations DataFrame size: {len(relations_df)}")
        print(f"Filtered relations DataFrame size: {len(filtered_df)}")
        
        return filtered_df

    def filter_redundant_direct_relations(self, df: pd.DataFrame, head_col: str = 'head', relation_col: str = 'relation', tail_col: str = 'tail') -> pd.DataFrame:
        """
        Remove redundant links from a DataFrame based on transitive relationships.
        
        If there exist links a->b, a->c, and b->c, then the direct link a->b is considered redundant
        and will be removed.
        
        Args:
            df: DataFrame with columns for head, relation, and tail
            head_col: Name of the head column (default: 'head')
            relation_col: Name of the relation column (default: 'relation')
            tail_col: Name of the tail column (default: 'tail')
        
        Returns:
            DataFrame with redundant links removed
        """
        if df.empty:
            return df
        
        # Create a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        # Get all unique nodes
        all_nodes = set(result_df[head_col].unique()) | set(result_df[tail_col].unique())
        
        # Create adjacency list for each node
        adjacency = {}
        for _, row in result_df.iterrows():
            head = row[head_col]
            tail = row[tail_col]
            relation = row[relation_col]
            
            if head not in adjacency:
                adjacency[head] = []
            adjacency[head].append((tail, relation))
        
        # Find redundant links
        redundant_indices = []
        
        for idx, row in result_df.iterrows():
            print(idx)
            head = row[head_col]
            tail = row[tail_col]
            relation = row[relation_col]
            
            # Check if there's a path from head to tail through another node
            if head in adjacency:
                for intermediate_node, intermediate_relation in adjacency[head]:
                    # Skip if intermediate node is the same as tail
                    if intermediate_node == tail:
                        continue
                    
                    # Check if there's a path from intermediate_node to tail
                    if intermediate_node in adjacency:
                        for final_node, final_relation in adjacency[intermediate_node]:
                            if final_node == tail:
                                # Found a transitive path: head -> intermediate_node -> tail
                                # The direct link head -> tail is redundant
                                redundant_indices.append(idx)
                                break
                    
                    if idx in redundant_indices:
                        break
        
        # Remove redundant links
        if redundant_indices:
            result_df = result_df.drop(redundant_indices).reset_index(drop=True)
            print(f"Removed {len(redundant_indices)} redundant links")
        else:
            print("No redundant links found")
        
        return result_df

class Visualize_KG:
    def __init__(self):
        pass
    @staticmethod
    def draw_ego_hops(G, center_node, hops=2,):
        """
        Draw the subgraph of G containing center_node plus all nodes
        at distance ≤ hops (first- and second-degree neighbors).
        
        Args:
            G (nx.Graph): The input graph
            center_node: The node to center the visualization around
            hops (int): Number of hops to include in visualization
        """
        # 1) compute distances up to `hops`
        lengths = nx.single_source_shortest_path_length(G, center_node, cutoff=hops)
        # `lengths` maps node → distance (0 for center, 1 for neighbors, 2 for 2-hop)
        
        # 2) induce the subgraph
        nodes_within = list(lengths.keys())
        H = G.subgraph(nodes_within).copy()
        
        # 3) draw
        pos = nx.spring_layout(H)  # force-directed layout
        plt.figure(figsize=(8,8))
        
        # draw nodes, coloring by distance
        node_colors = [lengths[n] for n in H.nodes()]
        nx.draw_networkx_nodes(
            H, pos,
            node_size=300,
            cmap=plt.cm.viridis,
            node_color=node_colors,
            vmin=0, vmax=hops
        )
        # highlight edges connected to center node
        center_edges = list(H.edges(center_node))
        nx.draw_networkx_edges(
            H, pos,
            edgelist=center_edges,
            width=2.5,
            alpha=0.8,
            edge_color='red'
        )
        # draw edges
        nx.draw_networkx_edges(H, pos, alpha=0.5)
        
        # highlight the center node with a red border
        nx.draw_networkx_nodes(
            H, pos,
            nodelist=[center_node],
            node_size=500,
            edgecolors='red',
            linewidths=2,
        )
    
        # labels - adjust font size and add bbox for better readability
        labels = nx.draw_networkx_labels(
            H, pos,
            font_size=8,
            font_weight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
            horizontalalignment='center'
        )
        
        plt.title(f"Ego-graph of {center_node} (hops≤{hops})")
        plt.axis('off')
        plt.show()

class Process_KG:
    def __init__(self):
        pass

        # test use sigmoid function
    def process_prediction(self,prediction_df):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        prediction_df["prediction_sigmoid"] = sigmoid(prediction_df['prediction'])
        prediction_df["prediction_value"] = (prediction_df["prediction_sigmoid"] > 0.5).astype(int) # 1 if sigmoid > 0.5, otherwise 0
        return prediction_df

    def count_hits(self,error_df):
        """
        Count hits and calculate accuracy metrics for prediction results.
        
        Args:
            error_df: DataFrame containing prediction results with 'prediction_value' and 'actual' columns
            
        Returns:
            DataFrame: The input DataFrame with additional columns for prediction metrics
        """
        # Add prediction correctness column
        error_df["prediction_correct"] = error_df["prediction_value"] == error_df["actual"]
        
        # Add positive and negative prediction columns
        error_df["prediction_positive"] = error_df["prediction_value"] == 1
        error_df["prediction_negative"] = error_df["prediction_value"] == 0

        
        # Print accuracy
        print(f"Accuracy in test set: {error_df['prediction_correct'].sum() / len(error_df)}")
        
        return error_df

    def connect_parents(self,error_df, relation_df):
        error_df_item = error_df.copy()
        
        # Select only necessary columns from relation_df to avoid conflicts
        parent_map = relation_df[['head', 'tail']].copy()

        # merge first level parent
        error_df_item = error_df_item.merge(
            parent_map, 
            left_on="item_id", 
            right_on="head", 
            how="left"
        ).rename(columns={"tail": "first_level_parent"})
        error_df_item.drop(columns=["head"], inplace=True) # Drop the 'head' column from parent_map

        # merge second level parent
        error_df_item = error_df_item.merge(
            parent_map, 
            left_on="first_level_parent", 
            right_on="head", 
            how="left",
            suffixes=('', '_drop2') # Add suffix to avoid conflict if 'head' somehow exists
        ).rename(columns={"tail": "second_level_parent"})
        error_df_item.drop(columns=["head", "head_drop2"], errors='ignore', inplace=True) # Drop potentially duplicated 'head'

        # merge third level parent
        error_df_item = error_df_item.merge(
            parent_map, 
            left_on="second_level_parent", 
            right_on="head", 
            how="left",
            suffixes=('', '_drop3')
        ).rename(columns={"tail": "third_level_parent"})
        error_df_item.drop(columns=["head", "head_drop3"], errors='ignore', inplace=True)

        # merge fourth level parent
        error_df_item = error_df_item.merge(
            parent_map, 
            left_on="third_level_parent", 
            right_on="head", 
            how="left",
            suffixes=('', '_drop4')
        ).rename(columns={"tail": "fourth_level_parent"})
        error_df_item.drop(columns=["head", "head_drop4"], errors='ignore', inplace=True)

        return error_df_item


    def map_ids_to_entities(self,df, mapping_df, df_type:str):
        if df_type == "relation":
            df["head"] = df["head"].map(mapping_df.set_index("id")["entity"])
            df["tail"] = df["tail"].map(mapping_df.set_index("id")["entity"])
        elif df_type == "error":
            df["item_id"] = df["item_id"].map(mapping_df.set_index("id")["entity"])
            df["user_id"] = df["user_id"].map(mapping_df.set_index("id")["entity"])
        return df



#if __name__ == "__main__":
    # Example code for preprocessing Amazon data and building knowledge graph
    # This is just for illustration - actual implementation would be in separate modules
    
    # current_dir = os.getcwd()
    # DATASETS = ["Books", "All_Beauty", "Video_Games"]
    # DATASET = DATASETS[2]
    # CORE_SETTING = 20
    # METADATA_NAME = "raw_meta_" + DATASET
    # DATASET_NAME = "5core_timestamp_" + DATASET
    
    # COLUMNS = ['description', 'features', 'details', 'store', 'categories', 'price', 'author']
    
    # amazon_data_prep = Preprocess_Amazon_Data(DATASET_NAME, METADATA_NAME, CORE_SETTING)
    # train_df, test_df, metadata_filtered_df = amazon_data_prep.load_data()
    # metadata_filtered_df = amazon_data_prep.clean_text_columns(metadata_filtered_df, COLUMNS)
    
    # DIR_NAME = "amazon-"
    # preprocessed_path = os.path.join(current_dir, 'data', 'preprocessed', f'{DIR_NAME}{DATASET}')
    # os.makedirs(preprocessed_path, exist_ok=True)
    
    # MODEL_TECHNIQUES = ["all-MiniLM-L6-v2","intfloat_multilingual-e5-large-instruct",
    #                     "Alibaba-NLP_gte-Qwen2-1.5B-instruct","gemma3", "ner"]
    # SELECTED_TECHNIQUE = MODEL_TECHNIQUES[3]
    
    # out_path = os.path.join(current_dir, 'data', 'experiments', "#1", 
    #                        f'{DIR_NAME}{DATASET}', SELECTED_TECHNIQUE)
    # taxonomy_path = os.path.join(current_dir, 'data', 'taxonomy', f'{DIR_NAME}{DATASET}')
    
    # metadata_filtered_df.to_csv(f'{preprocessed_path}/metadata_filtered_df.csv', index=False)
    # train_df.to_csv(f'{preprocessed_path}/train_df.csv', index=False)
    # test_df.to_csv(f'{preprocessed_path}/test_df.csv', index=False)
    
    # try:
    #     taxonomy_filename = f'{SELECTED_TECHNIQUE}_taxonomy-triples.csv'
    #     taxonomy_file_path = os.path.join(taxonomy_path, taxonomy_filename)
    #     product_topic_df = pd.read_csv(taxonomy_file_path)
    # except FileNotFoundError:
    #     product_topic_df = pd.DataFrame(columns=['head', 'relation', 'tail'])
    # except Exception as e:
    #     print(f"An unexpected error occurred during data loading: {e}")
    #     exit()
    
    # build_kg = Build_KG()
    # train_triples_df = build_kg.user_item_triples(train_df)
    # test_triples_df = build_kg.user_item_triples(test_df)
    
    # kg_df_final, _, _, _ = build_kg.build_kg(
    #     product_topic_df, train_triples_df, test_triples_df, out_path
    # )
    #pass  # Main execution code would go here

