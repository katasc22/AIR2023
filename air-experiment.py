from torch import device, cuda

from experiment.utils import utils
from experiment.data.DataHandler import DataHandler
from experiment.preprocessing.Preprocessor import Preprocessor
from experiment.preprocessing.TranslationHandler import TranslationHandler
from experiment.evaluation.Evaluator import Evaluator
from experiment.visualization.Visualizer import Visualizer
from experiment.ExperimentRunner import ExperimentRunner

POSSIBLE_LANGUAGES = ["de", "it", "fr"] #TODO: Add it to some type of config
PT_DEVICE = device('cuda' if cuda.is_available() else 'cpu')

def experiment_pipeline(experiment_mode: str, experiment_approach: str, translation_target: str, translation_langs: list[str], translation_mode: str):
	print("[Main] Starting experiments ...")
	dataHandler = DataHandler()
	translationHandler = TranslationHandler(translation_mode, translation_target, POSSIBLE_LANGUAGES, PT_DEVICE)

	preprocessor = Preprocessor(experiment_mode, dataHandler, translationHandler, translation_langs)
	preprocessed_data = preprocessor.preprocess()
	
	experimentRunner = ExperimentRunner(experiment_approach, experiment_mode, preprocessed_data, translationHandler, PT_DEVICE)
	ex_results = experimentRunner.runExperiment()

	evaluator = Evaluator(ex_results, dataHandler)
	val_results = evaluator.evaluate()

	visualizer = Visualizer(val_results, dataHandler)
	visualizer.visualize()
	print("[Main] Finished experiments ...")

def run():
	args = utils.parse_arguments()
	experiment_pipeline(args.experiment_mode[0], args.experiment_approach[0], args.translation_target[0], 
					 args.translation_languages, args.translation_mode[0])

if __name__ == "__main__":
	run()