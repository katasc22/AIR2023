from experiment.utils import utils
from experiment.data.DataHandler import DataHandler
from experiment.preprocessing.Preprocessor import Preprocessor
from experiment.preprocessing.TranslationHandler import TranslationHandler
# from experiment.evaluation import Evaluator
# from experiment.visualization import Visualizer

POSSIBLE_LANGUAGES = ["de", "it", "fr"] #TODO: Add it to some type of config

def experiment_pipeline(experiment_mode: str, translation_target: str, translation_langs: list[str]):
	print("[Main] Starting experiment ...")
	dataHandler = DataHandler()
	translationHandler = TranslationHandler("api", POSSIBLE_LANGUAGES)

	preprocessor = Preprocessor(experiment_mode, dataHandler, translationHandler, translation_target, translation_langs)
	preprocessed_data = preprocessor.preprocess()
	
	print(preprocessed_data.docs)

	qrels = dataHandler.get_qrels()
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
	experiment_pipeline(args.experiment_mode[0], args.translation_target[0], args.translation_languages)

if __name__ == "__main__":
	run()