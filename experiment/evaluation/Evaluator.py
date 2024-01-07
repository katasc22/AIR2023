from ..data.ExperimentData import ExperimentValidationData
from ..data.DataHandler import DataHandler

class Evaluator:
	def __init__(self, experiment_results, dataHandler: DataHandler):
		self.experiment_results = experiment_results
		self.dataHandler = dataHandler

		self.qrels = dataHandler.get_qrels()

	def evaluate_experiment(self, ex_result):
		return ExperimentValidationData(ex_result.experiment_approach)

	def evaluate(self):
		validation_results = []
		for ex_result in self.experiment_results:
			ex_validation_data = self.evaluate_experiment(ex_result)
			validation_results.append(ex_validation_data)

		return tuple(validation_results)