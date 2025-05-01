import pandas as pd
import numpy as np
# from scipy.sparse import csr_matrix # Not needed for this format
import os
# from collections import defaultdict # Not needed for this format
import json # For saving maps if needed

class PrepareData:

    def __init__(self):
        pass

    def user_item_triples(self, df: pd.DataFrame):
        """Converts user-item interaction df to triples format ('head', 'relation', 'tail')."""
        triples = []
        # Check required columns
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

        # Return the mapped KG df, the interaction lines (as lists), and the entity map
        # Note: Returning train_adj_lines/test_adj_lines might not be necessary if files are written correctly.
        # Returning the entity_map is useful.
        return kg_df_mapped, train_adj_lines, test_adj_lines, entity_to_id_map


if __name__ == "__main__":
    # --- Setup ---
    DATASETS = ["Books", "All_Beauty", "Video_Games","Last-FM"]
    DATASET = DATASETS[1] # Example: All_Beauty
    DIR_NAME = "amazon-"

    # --- Select ONE model technique ---
    MODEL_TECHNIQUES = ["all-MiniLM-L6-v2","intfloat_multilingual-e5-large-instruct","Alibaba-NLP_gte-Qwen2-1.5B-instruct","gemma3", "ner"]
    # Ensure technique name matches the filename format (replace / with _)
    SELECTED_TECHNIQUE = MODEL_TECHNIQUES[3] # Example: gemma3

    current_dir = os.getcwd()
    # Define specific output path based on dataset and technique
    # Make sure this path matches where the KGIL model expects the files
    out_path = os.path.join(current_dir, 'data', 'experiments', "#1", f'{DIR_NAME}{DATASET}', SELECTED_TECHNIQUE)

    preprocessed_path = os.path.join(current_dir, 'data', 'preprocessed', f'{DIR_NAME}{DATASET}')
    relations_path = os.path.join(current_dir, 'data', 'relations', f'{DIR_NAME}{DATASET}')

    # --- Load Data ---
    try:
        train_df_raw = pd.read_csv(os.path.join(preprocessed_path, 'train_df.csv'))
        test_df_raw = pd.read_csv(os.path.join(preprocessed_path, 'test_df.csv'))
        # Construct the correct relations filename based on the selected technique
        # Adjust the filename pattern if needed (e.g., _relations.csv, _taxonomy-triples.csv)
        relations_filename = f'{SELECTED_TECHNIQUE}_taxonomy-triples.csv' # Adjust if needed
        relations_file_path = os.path.join(relations_path, relations_filename)
        product_topic_df = pd.read_csv(relations_file_path)
        print(f"Loaded relations from: {relations_file_path}")
    except FileNotFoundError as e:
        print(f"Error loading data file: {e}. Check paths and filename pattern for technique '{SELECTED_TECHNIQUE}'.")
        # Fallback to empty dataframe if relations are optional or handle error
        product_topic_df = pd.DataFrame(columns=['head', 'relation', 'tail'])
        # exit() # Or exit if relations are required
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        exit()

    # --- Prepare Data ---
    prepare_data = PrepareData()
    train_triples_df = prepare_data.user_item_triples(train_df_raw)
    test_triples_df = prepare_data.user_item_triples(test_df_raw)

    # Build KG, Mappings, and Interaction Files
    # The function now handles writing files to the specified out_path
    kg_df_final, _, _, _ = prepare_data.build_kg(
        product_topic_df, train_triples_df, test_triples_df, out_path
    )

    print("\nData preparation finished.")
    print(f"Interaction files (train_adj.txt, test_adj.txt), KG (kg_final.txt), and maps saved in: {out_path}")

    # --- REMOVE Redundant File Writing ---
    # The build_kg function now handles writing files correctly.
    # os.makedirs(out_path, exist_ok=True) # Already done in function
    # # write it to txt file
    # with open(out_path+"/train_adj.txt", "w") as f: # Incorrect way to write
    #     f.write(str(train_adj))
    # with open(out_path+"/test_adj.txt", "w") as f: # Incorrect way to write
    #     f.write(str(test_adj))
    # with open(out_path+"/kg_final.txt", "w") as f: # Incorrect way to write
    #     f.write(str(kg_df))
