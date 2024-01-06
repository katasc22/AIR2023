import pandas as pd
from .TranslationHandler import TranslationHandler
from ..data.PreprocessedData import PreprocessedData

class Preprocessor:
	def __init__(self, docs: pd.DataFrame, queries: pd.DataFrame, translation_target: str, translation_languages: list[str]):
		self.docs = docs
		self.queries = queries

		self.target = translation_target
		self.translation_languages = translation_languages


	def preprocess(self):
		pass