import time 
import tqdm
import json 
import numpy as np 
import pandas as pd 
import spacy
from spacy.matcher import PhraseMatcher,Matcher
from spacy.tokens import Span
from spacy.tokens import Token, DocBin

from collections import Counter
from nltk.tokenize import MWETokenizer
from nltk.util import Trie
import nltk
import os 


class Hearst_Patterns:
    """ Extracts hearst patterns from a pandas DataFrame containing product metadata
    """

    def __init__(self, patterns_dict=None, model_name="en_core_web_lg"):
        """ creates an instance of the class Hearst_Patterns

        Args:
            patterns_dict (dict, optional): dictionary containing the patterns. Defaults to None.
            model_name (str, optional): the spacy model to use. Defaults to "en_core_web_lg".
        """
        # load the spacy model
        self.nlp = spacy.load(model_name)
        
        # add pipes if they don't exist
        if "merge_entities" not in self.nlp.pipe_names:
            self.nlp.add_pipe("merge_entities")
        if "merge_noun_chunks" not in self.nlp.pipe_names:
            self.nlp.add_pipe('merge_noun_chunks')

        # initialize matcher
        self.matcher = Matcher(self.nlp.vocab)
        
        # load patterns
        if patterns_dict:
            self.patterns = patterns_dict
            self._load_patterns_to_matcher()
        
        # this list is used in the method get_matches
        self.continue_words = [',', 'and', 'or', ';', 'also', 'as well']

    def _load_patterns_to_matcher(self):
        """ Load patterns from dictionary into the matcher """
        for pattern_name, pattern_list in self.patterns.items():
            for i, pattern in enumerate(pattern_list):
                self.matcher.add(f"{pattern_name}-{i}", pattern)

    def extract_entity_relations_from_dataframe(self, df, text_column="generated_description", asin_column="parent_asin", size=None):
        """ Extract entity relations from a pandas DataFrame using NER
        
        Args:
            df (pd.DataFrame): DataFrame containing the text data and ASIN
            text_column (str, optional): name of the column containing text. Defaults to "generated_description".
            asin_column (str, optional): name of the column containing ASIN. Defaults to "asin".
            size (int, optional): maximum number of texts to process. Defaults to None (process all).

        Returns:
            list: list of extracted entity relations as tuples (head, relation, tail, text)
        """
        extracted_relations = []
        
        # limit the number of texts to process if size is specified
        texts_to_process = df[[text_column, asin_column]].iloc[:size] if size else df[[text_column, asin_column]]
        
        print(f"Processing {len(texts_to_process)} texts for entity relations...")
        
        for idx, row in texts_to_process.iterrows():
            text = row[text_column]
            asin = row[asin_column]
            
            if pd.isna(text) or text.strip() == "":
                continue
                
            try:
                relations = self.get_entity_relations(text, asin)
                if relations:
                    extracted_relations.extend(relations)
                    
                if idx % 100 == 0:
                    print(f"Processed {idx}/{len(texts_to_process)} texts, found {len(extracted_relations)} entity relations")
                    
            except Exception as e:
                print(f"Error processing text {idx}: {e}")
                continue

        print(f"Total entity relations extracted: {len(extracted_relations)}")


        # list to df 
        ner_relations_df = pd.DataFrame(extracted_relations, columns=['head', 'relation', 'tail', 'label'])
        # Rename columns to match the expected structure: head=parent_asin, relation='related to', tail=ner_entity
        ner_relations_df = ner_relations_df.rename(columns={'word1': 'head', 'word2': 'tail'})
        ner_relations_df['relation'] = 'related to'  # Set all relations to 'related to'
        ner_relations_df = ner_relations_df.drop_duplicates()

        return ner_relations_df

    def get_entity_relations(self, text, asin):
        """ Extract entity relations from a single text using NER

        Args:
            text (str): text to analyze
            asin (str): ASIN of the product

        Returns:
            list: list of extracted entity relations as tuples (head, relation, tail, text)
        """
        doc = self.nlp(text)
        relations = []
        
        # Extract all named entities from the text
        for ent in doc.ents:
            entity_text = ent.text.strip()
            entity_label = ent.label_
            
            # Create hypernym relation: ASIN -> entity (ASIN is parent of entity)
            hypernym_relation = (asin, "related_to", entity_text, text)
            relations.append(hypernym_relation)
            
        
        # Remove duplicates while preserving order
        seen = set()
        unique_relations = []
        for relation in relations:
            if relation not in seen:
                seen.add(relation)
                unique_relations.append(relation)
        
        return unique_relations

    def extract_patterns_from_dataframe(self, df, text_column="generated_description", size=None):
        """ Extract patterns from a pandas DataFrame

        Args:
            df (pd.DataFrame): DataFrame containing the text data
            text_column (str, optional): name of the column containing text. Defaults to "source_text".
            size (int, optional): maximum number of texts to process. Defaults to None (process all).

        Returns:
            list: list of extracted patterns as tuples (word1, word2, relation, label, text)
        """
        extracted_patterns = []
        
        # limit the number of texts to process if size is specified
        texts_to_process = df[text_column].iloc[:size] if size else df[text_column]
        
        print(f"Processing {len(texts_to_process)} texts...")
        
        for idx, text in enumerate(texts_to_process):
            if pd.isna(text) or text.strip() == "":
                continue
                
            try:
                patterns = self.get_matches(text)
                if patterns:
                    extracted_patterns.extend(patterns)
                    
                if idx % 100 == 0:
                    print(f"Processed {idx}/{len(texts_to_process)} texts, found {len(extracted_patterns)} patterns")
                    
            except Exception as e:
                print(f"Error processing text {idx}: {e}")
                continue

        print(f"Total patterns extracted: {len(extracted_patterns)}")
        return extracted_patterns

    def get_matches(self, text):
        """ Extract matches from a single text

        Args:
            text (str): text to analyze

        Returns:
            list: list of extracted relations as tuples (word1, word2, relation, label, text)
        """
        label = {
            'rhyper': -1,
            'hyper': 1,
        }
        
        # add a period at the beginning to handle patterns that don't work at sentence start
        doc = self.nlp('. ' + text)

        matches = self.matcher(doc)
        relations = []
        
        for match_id, start, end in matches:
            # get all entities indices in the doc
            ent_indices = [i for i in range(start, end) if doc[i].text in [
                ent.text for ent in doc[start:end].ents]]
            
            if not ent_indices:  # no entity found
                continue

            # extract X...Y from a match ..X...Y.., so now we know that the first and the last token are the entities
            span = doc[min(ent_indices):max(ent_indices)+1]

            # Get string representation
            match_info = self.nlp.vocab.strings[match_id]
            match_name = match_info.split('-')[0]   # hyper or rhyper
            match_type = match_info.split('-')[1]   # single or multi

            np_0 = span[0]  # left term
            np_1 = span[-1]  # right term (or first right term if multiple)

            # all the right terms (ex. for Y...X1, X2, ...Xn) X1...Xn are the right terms
            right_terms = [np_1.text]
            if match_type == "multi":  # look for other terms (X2,X3..etc)
                # we use the same model to get the noun chunks
                doc_remaining = self.nlp(doc[end:].text)
                for d in doc_remaining:
                    # look for entities inside the noun chunk
                    matching_ents = [
                        ent.text for ent in doc.ents if ent.text in d.text]
                    if matching_ents:
                        right_terms.append(matching_ents[0])
                    elif d.text not in self.continue_words:  # stop when seeing a word that's not in the list
                        break

            for term in right_terms:
                relations.append(
                    (np_0.text, term, match_name, label[match_name], text))

        relations = set(relations)
        return list(relations)

    def save_patterns_to_csv(self, patterns, save_path="/Users/U725801/Documents/GitHub/Masterarbeit-Playground/data/taxonomy/amazon-Video_Games/hybrid_hearst_patterns.csv"):
        """ Save extracted patterns to CSV file

        Args:
            patterns (list): list of extracted patterns
            save_path (str, optional): path to save the CSV file. Defaults to "hearst_patterns.csv".
        """
        df = pd.DataFrame(patterns, columns=['word1', 'word2', 'relation', 'label', 'text'])
        df.to_csv(save_path, index=False)
        print(f"Patterns saved to {save_path}")
        return df


# Example usage
if __name__ == "__main__":
    DATASETS = ["Books", "All_Beauty","Video_Games","Last-FM"]
    DATASET = DATASETS[2]
    DIR_NAME = "amazon-" # else: Last_fm, MovieLens

    # Using a valid model ID from Hugging Face
    NER_MODELS = ["dslim/bert-base-NER", "Jean-Baptiste/roberta-large-ner-english"]
    NER_MODEL = NER_MODELS[0]
    
    # Initialize with transformer model
    hearst_patterns = Hearst_Patterns()

    try:
        current_dir = os.getcwd()
        data_path = os.path.join(current_dir, 'data', 'preprocessed', f'{DIR_NAME}{DATASET}', 'metadata_filtered_df.csv')
        
        # load data if file exists
        if os.path.exists(data_path):
            metadata_filtered_df = pd.read_csv(data_path)

            # call the class
            hearst_patterns = Hearst_Patterns()
            patterns = hearst_patterns.extract_patterns_from_dataframe(metadata_filtered_df)
            hearst_patterns.save_patterns_to_csv(patterns)

            # with just ner
            ner_relations = hearst_patterns.extract_entity_relations_from_dataframe(metadata_filtered_df)
            ner_relations.to_csv("/Users/U725801/Documents/GitHub/Masterarbeit-Playground/data/taxonomy/amazon-Video_Games/ner_relations.csv", index=False)

    except Exception as e:
        print(f"Error: {e}")
