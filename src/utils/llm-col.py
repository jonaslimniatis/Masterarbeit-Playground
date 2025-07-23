from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import time
import numpy as np
import mlflow

class LLMHierarchyCOL:
    def __init__(self, llm_name: str):
        # initialize LLM once
        self.llm = OllamaLLM(model=llm_name, temperature=0.1)

        # --- detailed summary chain ---
        detailed_system = """
You are an expert that generates detailed product descriptions.
You receive a short compact product description. The detailed product description will be later used to extract Named Entities or Key Words to build a taxonomy

Side information: the product description includes ingredients (e.g almond oil), usage information (e. g. silky hair), package dimensions etc
Be creative. Make it about a paragraph long

Rules:
- Don't include explanations or notes
- Don't hallucinate
- Only add valuable information to the product
- Prioritize sizing or packaging dimensions less, but still include them
"""
        detailed_user = "This is the description of the product. {product_description}"
        detail_tpl = ChatPromptTemplate(
            messages=[("system", detailed_system), ("human", detailed_user)],
            input_variables=["product_description"]
        )
        self.detail_chain = detail_tpl | self.llm

        # --- generate taxonomy chain ---
        gen_system = """
You are a helpful assistant and expert in constructing a taxonomy from a given concept.
You will build iteratively a taxonomy with a root concept and from the taxonomy in the previous step and the entity list.

The format of the generated taxonomy is:
1. Parent Concept
1.1 Child Concept.
Do not change any entity names when building the taxonomy.

Critical rules
- DO NOT ADD ANY COMMENTS OR EXPLANATIONS
- THERE IS ONE AND ONLY ONE ROOT NODE OF THE TAXONOMY
- ALL ENTITIES FROM THE LIST MUST APPEAR IN THE TAXONOMY IF FEASIBLE
- YOU ARE ALLOWED TO ADD OTHER ENTITIES IF THEY MAKE SENSE, DON'T HALLUCINATE
- MERGE ENTITIES WITH THE SAME NAME
- EXCLUDE ENTITIES WITH NO INFORMATION DETAIL FOR THE PARENT CONCEPT
- YOU ARE ALLOWED TO ADD OTHER ENTITIES IF THEY MAKE SENSE, DON'T HALLUCINATE
- ONLY RETURN THE TAXONOMY
"""
        gen_user = """
The root entity is {root_concept}, the taxonomy in the current step is:
{taxonomy}

The entity list for this step contains:
{list_entities}
"""
        gen_tpl = ChatPromptTemplate(
            messages=[("system", gen_system), ("user", gen_user)],
            input_variables=["root_concept", "taxonomy", "list_entities"]
        )
        self.gen_chain = gen_tpl | self.llm

        # --- update taxonomy chain ---
        update_system = """
You are a helpful assistant and expert in updating a given hierarchical taxonomy with a root concept.
Review the given taxonomy from the previous step.

If the taxonomy contains unreasonable or wrong relations, update them to make sense.
You may relocate child nodes to new parents and add nodes if necessary.

Critical rules
- DO NOT ADD ANY COMMENTS OR EXPLANATIONS
- THERE IS ONE AND ONLY ONE ROOT NODE
- ALL ENTITIES MUST APPEAR IF FEASIBLE
- YOU ARE ALLOWED TO ADD OTHER ENTITIES IF THEY MAKE SENSE, DON'T HALLUCINATE
- MERGE ENTITIES WITH THE SAME NAME
- EXCLUDE ENTITIES WITH NO DETAIL
- KEEP THE NUMBERING FORMAT
- ONLY RETURN THE TAXONOMY
"""
        update_user = """
The root entity is {root_concept}.
The entity list contains: {list_entities}

Current taxonomy:
{taxonomy}
"""
        update_tpl = ChatPromptTemplate(
            messages=[("system", update_system), ("user", update_user)],
            input_variables=["root_concept", "list_entities", "taxonomy"]
        )
        self.update_chain = update_tpl | self.llm

        # --- review taxonomy chain ---
        review_system = """
You are an expert for validating a generated hierarchical taxonomy.
Your task is to check if the taxonomy suits the product description.

If it's unsuitable, return FALSE.
If it's suitable, return TRUE.

Critical rules:
- Do not include any explanations or notes
- ONLY RETURN THE BOOLEAN
- Do not hallucinate
- Validate carefully

Output Format:
BOOLEAN
"""
        review_user = """
The root entity is {root_concept}.
Entity list: {list_entities}

Taxonomy to review:
{taxonomy}

Product description:
{product_description}
"""
        review_tpl = ChatPromptTemplate(
            messages=[("system", review_system), ("user", review_user)],
            input_variables=["root_concept", "list_entities", "taxonomy", "product_description"]
        )
        self.review_chain = review_tpl | self.llm

        # --- embedding & keyword models ---
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embed_kw_model = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")
        self.kw_model = KeyBERT(model=self.embed_kw_model)

    def generate_details(self, short_summary: str) -> str:
        return self.detail_chain.invoke({"product_description": short_summary})

    def get_keywords(self, detailed_summary: str) -> list:
        raw = self.kw_model.extract_keywords(
            detailed_summary,
            keyphrase_ngram_range=(1, 2),
            top_n=5,
            stop_words="english",
            use_maxsum=True
        )
        # return unique keywords
        return list({kw for kw, _ in raw})

    def generate_tax(self, root_concept: str, list_keywords: list, taxonomy: str) -> str:
        return self.gen_chain.invoke({
            "root_concept": root_concept,
            "taxonomy": taxonomy,
            "list_entities": list_keywords
        })

    def update_tax(self, taxonomy: str, root_concept: str, list_keywords: list) -> str:
        return self.update_chain.invoke({
            "root_concept": root_concept,
            "list_entities": list_keywords,
            "taxonomy": taxonomy
        })

    def review(self, detailed_summary: str, taxonomy: str, root_concept: str, list_keywords: list) -> bool:
        out = self.review_chain.invoke({
            "root_concept": root_concept,
            "list_entities": list_keywords,
            "taxonomy": taxonomy,
            "product_description": detailed_summary
        })
        return out.strip().upper() == "TRUE"

    def taxonomy_to_triples(self, taxonomy_text: str) -> pd.DataFrame:
        lines = taxonomy_text.strip().split("\n")
        triples = []
        hierarchy = {}
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split(" ", 1)
            if len(parts) < 2:
                continue
            level_num, concept = parts
            level_num = level_num.rstrip(".")
            hierarchy[level_num] = concept
            if "." in level_num:
                parent_level = ".".join(level_num.split(".")[:-1])
                if parent_level in hierarchy:
                    triples.append({
                        "head": hierarchy[parent_level],
                        "relation": "is parent of",
                        "tail": concept
                    })
        return pd.DataFrame(triples, columns=["head", "relation", "tail"])

    def linkage_asin_to_taxonomy(
        self,
        triples_df: pd.DataFrame,
        dict_asin_keywords: dict,
        sim_thresh: float = 0.5
    ) -> pd.DataFrame:
        heads = triples_df["head"].tolist()
        tails = triples_df["tail"].tolist()
        H = self.embed_model.encode(heads)  # (n, d)
        T = self.embed_model.encode(tails)  # (n, d)

        new_links = []
        for asin, kws in dict_asin_keywords.items():
            K = self.embed_model.encode(kws)  # (k, d)
            sim_h = K @ H.T                  # (k, n)
            sim_t = K @ T.T                  # (k, n)
            # find all keyword→taxonomy matches over threshold
            idxs_h = np.argwhere(sim_h > sim_thresh)
            idxs_t = np.argwhere(sim_t > sim_thresh)
            for i, j in idxs_h:
                new_links.append({"head": asin, "relation": "related to", "tail": heads[j]})
            for i, j in idxs_t:
                new_links.append({"head": asin, "relation": "related to", "tail": tails[j]})

        if new_links:
            out = pd.concat([triples_df, pd.DataFrame(new_links)], ignore_index=True)
            return out.drop_duplicates().reset_index(drop=True)
        return triples_df


if __name__ == "__main__":
    # Hyperparameters
    DATASETS = ["Books", "All_Beauty", "Video_Games", "Last-FM"]
    DATASET = DATASETS[2]  # e.g. "All_Beauty"
    DIR_NAME = "amazon-"
    LLM_MODEL = "gemma3:4b"
    SENTENCE_TRANSFORMER = "all-MiniLM-L6-v2"
    ROOT_CONCEPT = "Product details"
    EXPERIMENT_NAME = "LLM-COL Batch sizes"
    # Paths
    current_dir = os.getcwd()
    data_path = os.path.join(
        current_dir,
        "data",
        "preprocessed",
        f"{DIR_NAME}{DATASET}",
        "metadata_filtered_df.csv"
    )
    os.environ['MLFLOW_TRACKING_USERNAME'] = "jonas.limniatis2"
    os.environ['MLFLOW_TRACKING_PASSWORD'] = "4563359ec5513f3621677b9729db8a4c617e07eb"
    os.environ['MLFLOW_TRACKING_PROJECTNAME'] = "LLM-Col"

    mlflow.set_tracking_uri(f'https://dagshub.com/' + os.environ['MLFLOW_TRACKING_USERNAME']
                            + '/' + os.environ['MLFLOW_TRACKING_PROJECTNAME'] + '.mlflow')

    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.start_run(run_name=f"Batch Size 10 local")
    mlflow.set_tag("dataset", DATASET)
    mlflow.set_tag("llm", LLM_MODEL)
    mlflow.set_tag("Similarity Threshhold", "0.5")
    mlflow.set_tag("Batch size", "10")
    # Enabling autolog for LangChain will enable trace logging.
    mlflow.langchain.autolog()
    metadata_filtered_df = pd.read_csv("/Users/U725801/Documents/GitHub/Masterarbeit-Playground/data/preprocessed/amazon-Video_Games/metadata_filtered_df.csv")

    llm_hierarchy = LLMHierarchyCOL(LLM_MODEL)

    start = time.time()


    # 1) Generate detailed summaries (streaming to save memory)
    #for idx, row in metadata_filtered_df.iterrows():
    #    print(f"Detailing row {idx}")
    #    metadata_filtered_df.at[idx, "detailed_summary"] = (
     #       llm_hierarchy.generate_details(row.source_text)
     #   )

    # Or directly (MLflow >= 1.11.0):
    #mlflow.log_dict(metadata_filtered_df.to_dict(orient="records"), "data/metadata_filtered_df.json")


    # 2) Batch‐wise taxonomy building

    BATCH_SIZE = 10
    dict_asin_keywords = {}
    taxonomy_text = ""

    for i in range(0, len(metadata_filtered_df), BATCH_SIZE):

        print(f"Processing batch {i} to {i + BATCH_SIZE}")
        batch = metadata_filtered_df.iloc[i : i + BATCH_SIZE]
        batch_kw_lists = []
        for _, item in batch.iterrows():
            #kws = llm_hierarchy.get_keywords(item.detailed_summary)
            kws = item.keyword
            batch_kw_lists.append(kws)
            dict_asin_keywords[item.parent_asin] = kws

        # generate → update → review
        taxonomy_text = llm_hierarchy.generate_tax(ROOT_CONCEPT, batch_kw_lists, taxonomy_text)
        taxonomy_text = llm_hierarchy.update_tax(taxonomy_text, ROOT_CONCEPT, batch_kw_lists)
        ok = llm_hierarchy.review(batch.iloc[0].detailed_summary, taxonomy_text, ROOT_CONCEPT, batch_kw_lists)
        if not ok:
            taxonomy_text = llm_hierarchy.generate_tax(ROOT_CONCEPT, batch_kw_lists, "")
            ok = llm_hierarchy.review(batch.iloc[0].detailed_summary, taxonomy_text, ROOT_CONCEPT, batch_kw_lists)

    # transform → link → save
    triples_df = llm_hierarchy.taxonomy_to_triples(taxonomy_text)


    linked_df = llm_hierarchy.linkage_asin_to_taxonomy(triples_df, dict_asin_keywords)
    mlflow.log_dict(linked_df.to_dict(orient="records"), f"data/{LLM_MODEL}-taxonomy_triples.json")
    # calculate time 
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    mlflow.log_metric("time", end - start)
    # log end time
    mlflow.log_param("time", end - start)
    mlflow.end_run()
    linked_df.to_csv(f"/Users/U725801/Documents/GitHub/Masterarbeit-Playground/data/taxonomy/amazon-Video_Games/{LLM_MODEL}_taxonomy_batchsize_15.csv") 