from .data.PreprocessedData import PreprocessedData
from .data.ExperimentData import ExperimentResultData
from .approaches.translation_based import retrieve_k_documents_per_query_tb_monolingual

class ExperimentRunner:
	def __init__(self, experiment_approach: str, experiment_mode: str, preprocessed_data: PreprocessedData, device):
		self.experiment_approach = experiment_approach
		self.experiment_mode = experiment_mode
		self.preprocessed_data = preprocessed_data
		self.device = device

	def runExperiment(self):
		print("[ExperimentRunner] Start experiment ...")
		if self.experiment_approach == "all":
			print(self.device)
		elif self.experiment_approach == "translation-based":
			retrieved_docs_per_query = retrieve_k_documents_per_query_tb_monolingual(self.preprocessed_data.queries, self.preprocessed_data.docs, 
																			15, device=self.device)
			
			return (ExperimentResultData("translation-based", retrieved_docs_per_query, 15), )
		elif self.experiment_approach == "ml-mbert":
			pass
		elif self.experiment_approach == "ml_knowledge_distillation":
			pass