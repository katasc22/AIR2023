import ir_datasets
import os
import pandas as pd

class DataHandler:
    def __init__(self):
        self.dataset = ir_datasets.load("vaswani")
        self.docs = pd.DataFrame(self.dataset.docs_iter())
        self.queries = pd.DataFrame(self.dataset.queries_iter())
        self.qrels = pd.DataFrame(self.dataset.qrels_iter())

        self.path = os.path.dirname(os.path.abspath(__file__))
        self.cached_files_dir = os.path.join(self.path, "files_with_translations")

        self.write_raw_data_to_disk()


    # checks if translations are already cached on disk
    def does_cached_translated_dataset_exist(self):
        if not os.path.exists(self.cached_files_dir):
            return False
        
        return True
    
    
    def cache_translated_dataset_on_disk(self, queries: pd.DataFrame, docs: pd.DataFrame):
        if not os.path.exists(self.cached_files_dir):
            os.makedirs(self.cached_files_dir)

        queries_with_translations_path = os.path.join(self.path, "files_with_translations/queries_with_translations.csv")
        docs_with_translations_path = os.path.join(self.path, "files_with_translations/docs_with_translations.csv")
        
        queries.to_csv(queries_with_translations_path, index=False)
        docs.to_csv(docs_with_translations_path, index=False)


    def load_translated_dataset_from_disk(self):
        queries = pd.read_csv(os.path.join(self.cached_files_dir, "queries_with_translations.csv"))
        documents = pd.read_csv(os.path.join(self.cached_files_dir, "docs_with_translations.csv"))

        return queries, documents


    def get_raw_queries_and_docs(self):
        return self.docs, self.queries
    

    def get_qrels(self):
        return self.qrels
    

    # Write initial docs, queries and qrels to disk
    def write_raw_data_to_disk(self):
        file_dir = os.path.join(self.path, "files")
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        docs_path = os.path.join(self.path, "files/docs.csv")
        queries_path = os.path.join(self.path, "files/queries.csv")
        qrels_path = os.path.join(self.path, "files/qrels.csv")

        if (
             os.path.isfile(docs_path) or 
             os.path.isfile(queries_path) or 
             os.path.isfile(qrels_path)
        ):
            return

        self.docs.to_csv(docs_path, index=False)
        self.queries.to_csv(queries_path, index=False)
        self.qrels.to_csv(qrels_path, index=False)