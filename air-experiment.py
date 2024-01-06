from experiment.utils import utils
from experiment.data.DataHandler import DataHandler
from experiment.preprocessing.Preprocessor import Preprocessor
from experiment.preprocessing.TranslationHandler import TranslationHandler
from experiment.evaluation import Evaluator
from experiment.visualization import Visualizer
import time

POSSIBLE_LANGUAGES = ["de", "it", "fr"]

def experiment_pipeline(translation_target: str, translation_langs: list[str]):
	dataHandler = DataHandler()
	documents, queries = dataHandler.get_raw_queries_and_docs()
	qrels = dataHandler.get_qrels()

	# pretranslate dataset into all possible languages and cache on disk if translated dataset does not exist yet
	if not dataHandler.does_cached_translated_dataset_exist():
		translationHandler = TranslationHandler(translation_mode="api")
		translated_queries, translated_docs = translationHandler.translate_raw_data(POSSIBLE_LANGUAGES, queries, documents)
		dataHandler.cache_translated_dataset_on_disk(translated_queries, translated_docs)

	queries_with_translations, docs_with_translations = dataHandler.load_translated_dataset_from_disk()

	# preprocessor = Preprocessor(documents, queries, translation_target, translation_langs)
	# preprocessor.preprocess()
	'''
	
	preprocessed_data = TranslationHander.translate([it, de], query)

	exRunner = ExperimentRunner(preprocessed_data, mode) # mode e.g distiluse
	ex_results = exRunner.run()

	evaluator = Evaluator(results, true, true, false) # e.g Recall = true, Precision = true, F1Score = false
	eval_results = evaluator.evaluate()

	visualizer = Visualizer(eval_results, options for what to plot)
	visualizer.visualize()
	'''

def run():
	args = utils.parse_arguments()
	experiment_pipeline(args.translation_target[0], args.translation_languages)

if __name__ == "__main__":
	run()