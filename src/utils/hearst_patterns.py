import re
from typing import List, Tuple, Dict, Any, Optional
import os 
import pandas as pd

class HearstPatternExtractor:
    """
    A class for extracting relationships from text using Hearst patterns
    with transformer-based NER to extract proper named entities
    """
    def __init__(self, ner_model="dslim/bert-base-NER", use_transformers=True, max_ngram=3):
        """
        Initialize the Hearst pattern extractor
        
        Args:
            use_transformers (bool): Whether to use transformer models for entity extraction
            ner_model (str): Name of the transformer model to use for NER
            max_ngram (int): Maximum number of words in extracted entities
        """
        # Define Hearst patterns for extracting relationships
        self.patterns = [
            # Format: (regex pattern, relationship type)
            (r'([\w\s]+) such as ([\w\s]+)', "such as"),
            (r'such ([\w\s]+) as ([\w\s]+)', "such as"),
            (r'([\w\s]+) including ([\w\s]+)', "including"),
            (r'([\w\s]+) especially ([\w\s]+)', "especially"),
            (r'([\w\s]+) and other ([\w\s]+)', "and other"),
            (r'([\w\s]+) or other ([\w\s]+)', "or other"),
            (r'([\w\s]+) is a ([\w\s]+)', "is a"),
            (r'([\w\s]+) is an ([\w\s]+)', "is an"),
            (r'([\w\s]+) are ([\w\s]+)', "are"),
            (r'([\w\s]+) is part of ([\w\s]+)', "is part of"),
            (r'([\w\s]+) has ([\w\s]+)', "has"),
            (r'([\w\s]+) contains ([\w\s]+)', "contains"),
        ]
        
        self.use_transformers = use_transformers
        self.ner_model = None
        self.max_ngram = max_ngram
        
        if use_transformers:
            try:
                # Try loading the transformer modules
                from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
                
                # Try loading the NER model - using a valid model ID
                self.tokenizer = AutoTokenizer.from_pretrained(ner_model)
                self.model = AutoModelForTokenClassification.from_pretrained(ner_model)
                self.ner_model = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")
                # cant remove stopwords, because it will remove hearst patterns. 
                # E. g. NP "{,}  is a {NP0, ... NPn}"" would lead to "NP {NP0, ... NPn}""
                print(f"Successfully loaded transformer NER model: {ner_model}")
            except Exception as e:
                print(f"Warning: Failed to load transformer model. Error: {str(e)}")
                print("Falling back to regex pattern matching only.")
                self.use_transformers = False
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text using transformer NER model
        
        Args:
            text (str): Input text
            
        Returns:
            List[Dict[str, Any]]: List of entity dictionaries with text, type, score
        """
        if not self.use_transformers or self.ner_model is None:
            return []
            
        try:
            # Process with the NER pipeline
            entities = self.ner_model(text)
            # Sort by confidence score (descending)
            entities = sorted(entities, key=lambda x: x.get('score', 0), reverse=True)
            return entities
        except Exception as e:
            print(f"Warning: NER extraction failed: {str(e)}")
            return []
    
    def limit_to_ngram(self, text: str) -> str:
        """
        Limit text to a maximum number of words (n-gram)
        
        Args:
            text (str): Original text
            
        Returns:
            str: Text limited to max_ngram words
        """
        words = text.split()
        if len(words) <= self.max_ngram:
            return text
        return ' '.join(words[:self.max_ngram])
    
    def get_best_entity(self, text: str) -> Optional[str]:
        """
        Extract the best named entity from text
        
        Args:
            text (str): Text to extract entity from
            
        Returns:
            Optional[str]: Best entity or None if no entities found
        """
        if not self.use_transformers or self.ner_model is None:
            return self.limit_to_ngram(text)
            
        entities = self.extract_entities(text)
        
        if not entities:
            # No entities found, use ngram limited text
            return self.limit_to_ngram(text)
        
        # Return the highest scoring entity
        return entities[0]["word"]
    
    def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract semantic relations from text using pattern matching
        and NER to extract proper named entities
        
        Args:
            text (str): Input text
            
        Returns:
            List[Tuple[str, str, str]]: List of (hypernym, relation, hyponym) tuples
                where hypernym and hyponym are proper named entities when possible
        """
        relations = []
        
        # Apply each pattern to the text
        for pattern, relation in self.patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Extract the matched text parts
                    raw_hypernym = match.group(1).strip()
                    raw_hyponym = match.group(2).strip()
                    
                    # Get the best entity for each match
                    hypernym = self.get_best_entity(raw_hypernym)
                    hyponym = self.get_best_entity(raw_hyponym)
                    
                    # Only add if both entities are valid
                    if hypernym and hyponym and hypernym != hyponym:
                        relations.append((hypernym, relation, hyponym))
                except Exception as e:
                    # Just continue with the next match
                    continue
        
        return relations
    def generate_triples(self,result_df):

        triples = []
        for index, row in result_df.iterrows():
            # hearst patterns
            hypernym = row['hypernym']
            hyponym = row['hyponym']
            relation = row['relation']
            triples.append((hypernym, relation, hyponym))

            # link product to hyponym
            triples.append((row['parent_asin'], 'related to', hyponym))

        triples_df = pd.DataFrame(triples, columns=['head', 'relation', 'tail'])
        return triples_df

 

# Example usage
if __name__ == "__main__":
    DATASETS = ["Books", "All_Beauty","Video_Games","Last-FM"]
    DATASET = DATASETS[2]
    DIR_NAME = "amazon-" # else: Last_fm, MovieLens

    # Using a valid model ID from Hugging Face
    NER_MODELS = ["dslim/bert-base-NER", "Jean-Baptiste/roberta-large-ner-english"]
    NER_MODEL = NER_MODELS[0]
    
    # Initialize with transformer model
    extractor = HearstPatternExtractor(ner_model=NER_MODEL, use_transformers=True, max_ngram=2)

    
    try:
        current_dir = os.getcwd()
        data_path = os.path.join(current_dir, 'data', 'preprocessed', f'{DIR_NAME}{DATASET}', 'metadata_filtered_df.csv')
        
        # load data if file exists
        if os.path.exists(data_path):
            metadata_df = pd.read_csv(data_path)
            
            # Process the samples
            results = []
            for index, row in metadata_df.iterrows():
                text = row['source_text']
                print(f"\nProcessing item {index}:")
                triples = extractor.extract_triples(text)
                for hypernym, relation, hyponym in triples:
                    results.append({
                        "index": index,
                        "parent_asin": row.get('parent_asin', ''),
                        "hypernym": hypernym,
                        "relation": relation,
                        "hyponym": hyponym
                    })
            results_df = pd.DataFrame(results)

            # create triples
            triples_df = extractor.generate_triples(results_df)

            if len(triples_df) > 0:
                out_path = os.path.join(current_dir, 'data', 'taxonomy', f'{DIR_NAME}{DATASET}/hearst_patterns_relations.csv')
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                triples_df.to_csv(out_path, index=False)
                print(f"Saved {len(triples_df)} relations to {out_path}")
            else:
                print("No relations found.")
                
    except Exception as e:
        print(f"Error processing data file: {str(e)}")
