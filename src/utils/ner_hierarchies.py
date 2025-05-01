import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
class NERRelation:
    def __init__(self,model_name,tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        

    def extract_triplets(self, text):
        triplets = []
        #TODO: change triplets list to dataframe
        relation, subject, relation, object_ = '', '', '', ''
        text = text.strip()
        current = 'x'
        for token in text.replace("", "").replace("<pad>", "").replace("</s>", "").split():
            if token == "<triplet>":
                current = 't'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                    relation = ''
                subject = ''
            elif token == "<subj>":
                current = 's'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                object_ = ''
            elif token == "<obj>":
                current = 'o'
                relation = ''
            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    object_ += ' ' + token
                elif current == 'o':
                    relation += ' ' + token
        if subject != '' and relation != '' and object_ != '':
            triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
        return triplets

    def generate_triples(self,row, triples_df):
        texts = [row.source_text]
        parent_asin = row.parent_asin
        # Tokenizer text
        model_inputs = self.tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        generated_tokens = self.model.generate(
            model_inputs["input_ids"].to(self.model.device),
            attention_mask=model_inputs["attention_mask"].to(self.model.device),
            max_length=256,
            length_penalty=0,
            num_beams=3,
            num_return_sequences=3
        )
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        for idx, sentence in enumerate(decoded_preds):
            et = self.extract_triplets(sentence)
            for t in et:
                # link product to head / hyponym
                triples_df = pd.concat([triples_df,pd.DataFrame([{"head":parent_asin,"relation": t['type'], "tail": t['head']}])], ignore_index=True)
                # link head / hyponym to tail / hypernym
                triples_df = pd.concat([triples_df,pd.DataFrame([{"head":t['head'],"relation": t['type'], "tail": t['tail']}])], ignore_index=True)
        return triples_df
if __name__ == "__main__":
    DATASETS = ["Books", "All_Beauty", "Video_Games","Last-FM"]
    DATASET = DATASETS[2]
    DIR_NAME = "amazon-" # else: Last_fm, MovieLens

    start_time = time.time()
    NER_MODEL = "Babelscape/rebel-large"
    
    # get data
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, 'data', 'preprocessed', f'{DIR_NAME}{DATASET}', 'metadata_filtered_df.csv')
    metadata_filtered_df = pd.read_csv(data_path)


    # build triples dataframe
    rebel_triples_df = pd.DataFrame(columns=["head","relation", "tail"])

    ner_pattern_extractor = NERRelation(model_name=NER_MODEL,tokenizer_name=NER_MODEL)
    for i in tqdm(range(0, len(metadata_filtered_df))):
        rebel_triples_df = ner_pattern_extractor.generate_triples(metadata_filtered_df.iloc[i],rebel_triples_df)

    # drop duplicates
    rebel_triples_df = rebel_triples_df.drop_duplicates()

    # save triples to csv
    out_path = os.path.join(current_dir, 'data', 'taxonomy', f'{DIR_NAME}{DATASET}/ner_relations.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rebel_triples_df.to_csv(out_path, index=False)
    end_time = time.time()
    print("Runtime: ", end_time - start_time)
    print("Runetime per item: ", (end_time - start_time) / len(metadata_filtered_df))
