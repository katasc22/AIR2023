import sys
from preprocessing import TranslationHandler
from data import DataHandler
from experiment import ExperimentRunner

def experiment_pipeline():
	pass
	'''
	query, docs = DataHandler.loadData()

	preprocessed_data = TranslationHander.translate([it, de], query)

	exRunner = ExperimentRunner(preprocessed_data, mode) # mode e.g distiluse
	ex_results = exRunner.run()

	evaluator = Evaluator(results, true, true, false) # e.g Recall = true, Precision = true, F1Score = false
	eval_results = evaluator.evaluate()

	visualizer = Visualizer(eval_results, options for what to plot)
	visualizer.visualize()
	'''

def main(argv):
	# TODO Max: parse arguments
	print(argv)

if __name__ == "__main__":
	main(sys.argv)