from .data.PreprocessedData import PreprocessedData

class ExperimentRunner:
	def __init__(self, experiment_approach: str, experiment_mode: str, preprocessed_data: PreprocessedData):
		self.experiment_approach = experiment_approach
		self.experiment_mode = experiment_mode
		self.preprocessed_data = preprocessed_data

	def runExperiment(self):
		print("[ExperimentRunner] Start experiment ...")
		if self.experiment_approach == "all":
			pass
		elif self.experiment_approach == "translation-based":
			pass
		elif self.experiment_approach == "ml-mbert":
			pass
		elif self.experiment_approach == "ml_knowledge_distillation":
			pass