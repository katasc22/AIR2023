from deep_translator import GoogleTranslator
from transformers import pipeline

class TranslationHandler:
	def __init__(self, translation_mode, translation_target, possible_languages, device):
		self.translation_mode = translation_mode
		self.translation_target = translation_target
		self.possible_languages = possible_languages
		self.device = device


	def translate(self, src_lang, target_lang, text):
		return GoogleTranslator(source=src_lang, target=target_lang).translate(text)


	def translate_raw_data(self, queries, docs):
		# queries
		for language in self.possible_languages:
			print(f"[TranslationHandler]: Starting with translating raw query data to \"{language}\" ...")
			target_column = f"text_{language}"
			if self.translation_mode == "transformer":
				model_checkpoint = f"Helsinki-NLP/opus-mt-en-{language}"
				translator = pipeline("translation", model=model_checkpoint, device=self.device)
				queries[target_column] = queries["text"].apply(lambda text: translator(text)[0]["translation_text"])

			elif self.translation_mode == "api":
				translator = GoogleTranslator("en", language)
				queries[target_column] = queries["text"].apply(lambda text: translator.translate(text))

			print(f"[TranslationHandler]: Finished translating raw query data to \"{language}\".")

		# docs
		for language in self.possible_languages:
			print(f"[TranslationHandler]: Starting with translating raw document data to \"{language}\" ...")
			target_column = f"text_{language}"
			if self.translation_mode == "transformer":
				model_checkpoint = f"Helsinki-NLP/opus-mt-en-{language}"
				translator = pipeline("translation", model=model_checkpoint, device=self.device)
				docs[target_column] = docs["text"].apply(lambda text: translator(text)[0]["translation_text"])

			elif self.translation_mode == "api":	
				translator = GoogleTranslator("en", language)
				docs[target_column] = docs["text"].apply(lambda text: translator.translate(text))

			print(f"[TranslationHandler]: Finished translating raw document data to \"{language}\".")

		return queries, docs
