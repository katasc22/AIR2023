from ..data.ExperimentData import ExperimentValidationData
from ..data.DataHandler import DataHandler # for getting the path to save plots on disk

class Visualizer:
	def __init__(self, validation_results: ExperimentValidationData, dataHandler: DataHandler):
		self.validation_results = validation_results

	def visualize(self):
		pass