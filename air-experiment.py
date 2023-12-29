import sys
from experiment.data import data_handling
from experiment.preprocessing import TranslationHandler
from experiment.evaluation import Evaluator
from experiment.visualization import Visualizer

def experiment_pipeline():
	pass
	'''
	documents, queries, qrels = data_handling.loadData()    
	
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