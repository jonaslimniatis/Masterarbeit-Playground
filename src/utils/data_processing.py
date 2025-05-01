from datasets import load_dataset
import re
import pandas as pd

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
    
if __name__ == "__main__":

    # select dataset to preprocess
    DATASETS = ["Books", "All_Beauty", "Video_Games"]
    DATASET = DATASETS[0]
    CORE_SETTING = 10
    METADATA_NAME = "raw_meta_" + DATASET
    DATASET_NAME = "5core_timestamp_" + DATASET #keep 5core, because McAuley datasets only have 5core
    
    # select columns to preprocess
    COLUMNS = ['description', 'features', 'details', 'store', 'categories', 'price', 'author']

    # load data
    amazon_data_prep = Preprocess_Amazon_Data(DATASET_NAME, METADATA_NAME, CORE_SETTING)
    train_df, test_df, metadata_filtered_df = amazon_data_prep.load_data()

    # clean data
    metadata_filtered_df = amazon_data_prep.clean_text_columns(metadata_filtered_df, COLUMNS)

    # Create directory if it doesn't exist
    import os
    os.makedirs(f'../../data/preprocessed/amazon-{DATASET}', exist_ok=True)
    
    # Save data to the created directory
    metadata_filtered_df.to_csv(f'../../data/preprocessed/amazon-{DATASET}/metadata_filtered_df.csv', index=False)
    train_df.to_csv(f'../../data/preprocessed/amazon-{DATASET}/train_df.csv', index=False)
    test_df.to_csv(f'../../data/preprocessed/amazon-{DATASET}/test_df.csv', index=False)
