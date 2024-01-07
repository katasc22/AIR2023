from ..data.PreprocessedData import PreprocessedData
from ..data.DataHandler import DataHandler
from .TranslationHandler import TranslationHandler
from random import seed, choice 

class Preprocessor:
	def __init__(self, experiment_mode, dataHandler: DataHandler, translationHandler: TranslationHandler, 
			  translation_target: str, translation_languages: list[str]):
		self.experiment_mode = experiment_mode
		self.dataHandler = dataHandler
		self.translationHandler = translationHandler

		self.target = translation_target
		self.translation_languages = translation_languages

		self.queries, self.docs = self.dataHandler.get_raw_queries_and_docs()
		self.convert_id_column_to_int()


	def get_possible_text_columns(self):
		possible_columns = ["text"]
		for language in self.translation_languages:
			possible_columns.append(f"text_{language}")

		return possible_columns


	def prepare_translated_data(self):
		seed(22) #TODO: Add parameter for seed
		possible_columns = self.get_possible_text_columns()
		if self.target == "docs":
			preprocessed_queries = self.queries[["query_id", "text"]].copy()
			preprocessed_docs = self.docs[["doc_id"]].copy()
			preprocessed_docs["text"] = self.docs.apply(lambda row: row[choice(possible_columns)], axis=1)
		elif self.target == "queries":
			preprocessed_docs = self.docs[["doc_id", "text"]].copy()
			preprocessed_queries = self.queries[["query_id"]].copy()
			preprocessed_queries["text"] = self.queries.apply(lambda row: row[choice(possible_columns)], axis=1)

		return PreprocessedData(preprocessed_queries, preprocessed_docs)
	
	def convert_id_column_to_int(self):
		self.queries["query_id"] = self.queries["query_id"].astype(int)
		self.docs["doc_id"] = self.docs["doc_id"].astype(int)

	def lower_untranslated_text(self):
		self.queries["text"] = self.queries["text"].str.lower()
		self.docs["text"] = self.docs["text"].str.lower()


	def lower_all_translations(self):
		for languages in self.translation_languages:
			column_name = f"text_{languages}"
			self.docs[column_name] = self.docs[column_name].str.lower()
			self.queries[column_name] = self.queries[column_name].str.lower()


	def translate_dataset_if_not_cached(self):
		if not self.dataHandler.does_cached_translated_dataset_exist():
			print("[Preprocessor] No cached files found starting translation of raw data ...")
			translated_queries, translated_docs = self.translationHandler.translate_raw_data(self.queries, self.documents)
			self.dataHandler.cache_translated_dataset_on_disk(translated_queries, translated_docs)


	def preprocess(self):
		print("[Preprocessor] Start preprocessing ...")
		# set all text lowercase
		self.lower_untranslated_text()

		if self.experiment_mode == "monolingual":
			print(self.queries["query_id"].dtype)
			print(self.docs["doc_id"].dtype)
			preprocessed_data = PreprocessedData(self.queries, self.docs)

		elif self.experiment_mode == "multilingual":
			self.translate_dataset_if_not_cached()

			self.queries, self.docs = self.dataHandler.load_translated_dataset_from_disk()
			self.lower_all_translations()

			preprocessed_data = self.prepare_translated_data()

		print("[Preprocessor] Finished preprocessing.")
		
		return preprocessed_data