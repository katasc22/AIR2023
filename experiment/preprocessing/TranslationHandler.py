import pandas as pd
from deep_translator import GoogleTranslator
from transformers import pipeline

class TranslationHandler:
	def __init__(self, translation_mode):
		self.translation_mode = translation_mode

	def translate(self, src_lang, target_lang, text):
		return GoogleTranslator(source=src_lang, target=target_lang).translate(text)

	def translate_raw_data(self, target_languages, queries, docs):
		# lowercase for better translations
		queries["text"] = queries["text"].str.lower()
		docs["text"] = docs["text"].str.lower()

		# queries
		for language in target_languages:
			print(f"[TranslationHandler]: Starting with translating raw query data to \"{language}\" ...")
			target_column = f"text_{language}"
			if self.translation_mode == "transformer":
				model_checkpoint = f"Helsinki-NLP/opus-mt-en-{language}"
				translator = pipeline("translation", model=model_checkpoint)
				queries[target_column] = queries["text"].apply(lambda text: translator(text)[0]["translation_text"])

			elif self.translation_mode == "api":
				translator = GoogleTranslator("en", language)
				queries[target_column] = queries["text"].apply(lambda text: translator.translate(text))

			print(f"[TranslationHandler]: Finished translating raw query data to \"{language}\".")

		# docs
		for language in target_languages:
			print(f"[TranslationHandler]: Starting with translating raw document data to \"{language}\" ...")
			target_column = f"text_{language}"
			if self.translation_mode == "transformer":
				model_checkpoint = f"Helsinki-NLP/opus-mt-en-{language}"
				translator = pipeline("translation", model=model_checkpoint)
				docs[target_column] = docs["text"].apply(lambda text: translator(text)[0]["translation_text"])

			elif self.translation_mode == "api":	
				translator = GoogleTranslator("en", language)
				docs[target_column] = docs["text"].apply(lambda text: translator.translate(text))

			print(f"[TranslationHandler]: Finished translating raw document data to \"{language}\".")

		return queries, docs
