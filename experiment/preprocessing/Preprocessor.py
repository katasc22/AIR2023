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


	def lower_all_translations(self):
		for languages in self.translation_languages:
			column_name = f"text_{languages}"
			self.docs[column_name] = self.docs[column_name].str.lower()
			self.queries[column_name] = self.queries[column_name].str.lower()


	def get_possible_text_columns(self):
		possible_columns = ["text"]
		for language in self.translation_languages:
			possible_columns.append(f"text_{language}")

		return possible_columns


	def prepare_translated_data(self):
		seed(22) #TODO: Add parameter for seed
		possible_columns = self.get_possible_text_columns()
		if self.target == "docs":
			preprocessed_queries = self.queries[["query_id", "text"]]
			preprocessed_docs = self.docs[["doc_id"]]
			preprocessed_docs["text"] = self.docs.apply(lambda row: row[choice(possible_columns)], axis=1)
		elif self.target == "queries":
			preprocessed_docs = self.docs[["doc_id", "text"]]
			preprocessed_queries = self.queries[["query_id"]]
			preprocessed_queries["text"] = self.queries.apply(lambda row: row[choice(possible_columns)], axis=1)

		return PreprocessedData(preprocessed_queries, preprocessed_docs)
	
	def lower_untranslated_text(self):
		self.queries["text"] = self.queries["text"].str.lower()
		self.docs["text"] = self.docs["text"].str.lower()


	def preprocess(self):
		# set all text lowercase
		self.lower_untranslated_text()

		if self.experiment_mode == "monolingual":
			preprocessed_data = PreprocessedData(self.queries, self.docs)

		elif self.experiment_mode == "multilingual":
			if not self.dataHandler.does_cached_translated_dataset_exist():
				translated_queries, translated_docs = self.translationHandler.translate_raw_data(self.queries, self.documents)
				self.dataHandler.cache_translated_dataset_on_disk(translated_queries, translated_docs)

			self.queries, self.docs = self.dataHandler.load_translated_dataset_from_disk()
			self.lower_all_translations()

			preprocessed_data = self.prepare_translated_data()
		
		return preprocessed_data