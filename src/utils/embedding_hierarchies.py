import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN
from bertopic.representation import KeyBERTInspired 
from scipy.cluster import hierarchy as sch
import os
import numpy as np
import time
import mlflow 
import torch
import gc
import psutil

class EmbeddingHierarchy:
    def __init__(self, embedding_model,random_seed):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.random_seed = random_seed

    def topic_model_hierarchy(self,df,bool_representation_model:bool,cluster_model:str):
        """
        Fit a topic model to the data and return the hierarchical topics and topics
        Args:
            df: DataFrame containing the data
            bool_representation_model: Boolean indicating whether to use a representation model to enhance topics
            cluster_model: String indicating the type of clustering model to use
            num_clusters: Integer indicating the number of clusters to use
        Returns:
            hierarchical_topics: DataFrame containing the hierarchical topics
            topics: Series containing the topics
        """
        # if representation_model is true, use KeyBERTInspired
        if bool_representation_model:
            representation_model = KeyBERTInspired()
        if cluster_model == "KMeans":
            cluster_model = KMeans(random_state=self.random_seed)
        elif cluster_model == "HDBSCAN":
            cluster_model = HDBSCAN()

        topic_model = BERTopic(embedding_model=self.embedding_model, representation_model=representation_model, hdbscan_model=cluster_model)

        topics, probs = topic_model.fit_transform(df.source_text) #TODO: update column name 
        linkage_function = lambda x: sch.linkage(x, 'centroid', optimal_ordering=True) #update linkage function before experiment
        hierarchical_topics = topic_model.hierarchical_topics(df.source_text, linkage_function=linkage_function)
        return hierarchical_topics, topic_model,topics
    
    def create_triples(self,hierarchical_topics,df, topics):
        def triples_from_hierarchy(hierarchical_topics):
            """
        Process hierarchical topics dataframe to create triples in the form of
        (head, relation, tail) where head is parent_name, relation is 'subcategory of',
        and tail is either child_left_name or child_right_name.
        
        Args:
            hierarchical_topics_df: DataFrame containing hierarchical topic information
            
        Returns:
                List of triples representing the hierarchy relationships
            """
            id_to_name_map = {}
            triples = []
            relation = "subcategory of"
            
            for _, row in hierarchical_topics.iterrows():
                parent_id = row['Parent_ID']
                child_left_id = row['Child_Left_ID']
                child_right_id = row['Child_Right_ID']

                # get names 
                parent_name = row["Parent_Name"]                
                child_left_name = row["Child_Left_Name"]
                child_right_name = row["Child_Right_Name"]

                triples.append((child_left_name, relation, parent_name))
                triples.append((child_right_name, relation, parent_name))

                # convert ids back to names 
                id_to_name_map[parent_id] = parent_name
                id_to_name_map[child_left_id] = child_left_name
                id_to_name_map[child_right_id] = child_right_name
                print(id_to_name_map)
            self.id_to_name_map = id_to_name_map
            # concat triples to df 
            triples_df = pd.DataFrame(triples, columns=['head', 'relation', 'tail'])
            return triples_df
        
        def link_product_to_topic(df,topics,triples_df):
            df['topic'] = topics
            df["topic"] = df["topic"].astype(str)
            # rename_column
            df = df.rename(columns={"parent_asin": "head", "topic": "tail"})
            # add relation column 
            df["relation"] = "related to"
            # filter columns 
            df = df.filter(items=["head", "relation", "tail"])
            # rename tail by reversing the id_to_name_map
            df["tail"] = df["tail"].map(self.id_to_name_map)

            # append triples to df 
            triples_df = pd.concat([triples_df, df], ignore_index=True)


            return triples_df
    # call function
        triples_df = triples_from_hierarchy(hierarchical_topics)
        triples_df = link_product_to_topic(df,topics,triples_df)

        return triples_df
    
    def safe_plots(self,folder_path,embedding_model_name,topic_model,topics,hierarchical_topics):
        """
        Visualize the term rank of the topics for given topic model
        Args:
            topics: Series containing the topics
            hierarchical_topics: DataFrame containing the hierarchical topics
            topic_model: Topic model used to create the topics and hierarchical topics
            embedding_model_name: Name of the embedding model used
            folder_path: Path to the folder where the plots should be saved, includes the dataset name
        Returns:
            Save plot to file
            Name convention: <folder_path>_term_rank_<embedding_model>.png
        """
        # Sanitize embedding model name for file naming by replacing slashes with underscores
        safe_model_name = embedding_model_name.replace('/', '_')
        
        # save term rank plot to file
        term_rank_fig = topic_model.visualize_term_rank(topics=topics)
        term_rank_fig.write_html(f"{folder_path}/term_rank_{safe_model_name}.html")

        # save topic info to csv
        topic_info_df = topic_model.get_topic_info()
        topic_info_df.to_csv(f"{folder_path}/topic_info_{safe_model_name}.csv")

        # save hierarchy plot to file
        hierarchy_fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
        hierarchy_fig.write_html(f"{folder_path}/hierarchy_{safe_model_name}.html")

        topic_model.visualize_topics(topics=topics)

        
if __name__ == "__main__":
    
    # TODO Update file path
    DATASETS = ["All_Beauty", "Video_Games","Last-FM"]
    DATASET = DATASETS[1]
    DIR_NAME = "amazon-" # else: Last_fm, MovieLens
    RANDOM_SEED = 42 # Assign the integer seed directly
    np.random.seed(RANDOM_SEED) # Set the numpy random seed
    
    # Get the project root directory (Masterarbeit-Playground folder)
    current_dir = os.getcwd()
    
    # Construct absolute path to the data file
    data_path = os.path.join(current_dir, 'data', 'preprocessed', f'{DIR_NAME}{DATASET}', 'metadata_filtered_df.csv')
    metadata_df = pd.read_csv(data_path)
    
    start_time = time.time()
    embeddings_models = ["all-MiniLM-L6-v2","intfloat/multilingual-e5-large-instruct","Alibaba-NLP/gte-Qwen2-1.5B-instruct", "Qwen/Qwen3-Embedding-0.6B","Qwen/Qwen3-Embedding-1.5B"]
    EMBEDDING_MODEL = embeddings_models[4]  

 
    os.environ['MLFLOW_TRACKING_USERNAME'] = "jonas.limniatis2"
    os.environ['MLFLOW_TRACKING_PASSWORD'] = "4563359ec5513f3621677b9729db8a4c617e07eb"
    os.environ['MLFLOW_TRACKING_PROJECTNAME'] = "LLM-Col"

    mlflow.set_tracking_uri(f'https://dagshub.com/' + os.environ['MLFLOW_TRACKING_USERNAME']
                          + '/' + os.environ['MLFLOW_TRACKING_PROJECTNAME'] + '.mlflow')
    
    EXPERIMENT_NAME = "Hierarchical Topic Modeling"

    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.start_run(run_name=f"{EMBEDDING_MODEL}")

    mlflow.set_tag("embedding_model", EMBEDDING_MODEL)
    mlflow.set_tag("dataset", DATASET)

    #NUM_CLUSTERS = 5

    safe_model_name = EMBEDDING_MODEL.replace('/', '_')

    embedding_hierarchy = EmbeddingHierarchy(embedding_model=EMBEDDING_MODEL, random_seed=RANDOM_SEED)
    hierarchical_topics, topic_model,topics = embedding_hierarchy.topic_model_hierarchy(df=metadata_df,bool_representation_model=True,cluster_model="HDBSCAN")
    

    embedding_triples_df = embedding_hierarchy.create_triples(hierarchical_topics,metadata_df,topics)
    # filter out na in tail column
    embedding_triples_df = embedding_triples_df[embedding_triples_df['tail'].notna()]
    # build path 
    mlflow.log_dict(embedding_triples_df.to_dict(orient="records"), f"{EMBEDDING_MODEL}_taxonomy.json")

    folder_path = os.path.join(current_dir, 'data', 'taxonomy', 'amazon-Video_Games')

    os.makedirs(folder_path, exist_ok=True)
    #embedding_hierarchy.safe_plots(folder_path,embedding_model_name=EMBEDDING_MODEL,topic_model=topic_model,topics=topics,hierarchical_topics=hierarchical_topics)
    embedding_triples_df.to_csv(f"{folder_path}/{EMBEDDING_MODEL}_taxonomy.csv")

    end_time = time.time()
    print("Runtime: ", end_time - start_time)
    print("Runetime per item: ", (end_time - start_time) / len(metadata_df))
    mlflow.log_param("embedding_model", EMBEDDING_MODEL)
    mlflow.log_param("dataset", DATASET)
    mlflow.log_metric("runtime", end_time - start_time)
    mlflow.end_run()