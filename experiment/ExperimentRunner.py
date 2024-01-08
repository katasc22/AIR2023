from .data.PreprocessedData import PreprocessedData
from .data.ExperimentData import ExperimentResultData
from .approaches.translation_based import retrieve_k_documents_per_query_tb_monolingual, retrieve_k_documents_per_query_tb_multilingual
from .approaches.multilingual_bert import retrieve_k_documents_per_query_mbert
from .approaches.knowledge_distillation import retrieve_k_documents_per_query_distiluse


class ExperimentRunner:
	def __init__(self, experiment_approach: str, experiment_mode: str, preprocessed_data: PreprocessedData,
				translationHandler, device):
		self.experiment_approach = experiment_approach
		self.experiment_mode = experiment_mode
		self.preprocessed_data = preprocessed_data
		self.translationHandler = translationHandler
		self.device = device


	def runExperiment(self):
		print("[ExperimentRunner] Start experiment ...")
		if self.experiment_approach == "translation-based":
			# retrieved_docs_per_query = retrieve_k_documents_per_query_tb_monolingual(self.preprocessed_data.queries, self.preprocessed_data.docs, 
			# 																	15, device=self.device)
			
			# return ExperimentResultData("translation-based", retrieved_docs_per_query, 15)
			if self.experiment_mode == "monolingual":
				retrieved_docs_per_query = retrieve_k_documents_per_query_tb_monolingual(self.preprocessed_data.queries, self.preprocessed_data.docs, 
																				15, device=self.device)
			
				return ExperimentResultData("translation-based", retrieved_docs_per_query, 15)
			
			elif self.experiment_mode == "multilingual":
				retrieved_docs_per_query = retrieve_k_documents_per_query_tb_multilingual(self.translationHandler, self.preprocessed_data.queries, self.preprocessed_data.docs, 
																				15, device=self.device)
				
				return ExperimentResultData("translation-based", retrieved_docs_per_query, 15)
		
		elif self.experiment_approach == "ml-mbert":
			retrieved_docs_per_query  = retrieve_k_documents_per_query_mbert(self.preprocessed_data.queries, self.preprocessed_data.docs, 
																			10, device=self.device)
			return ExperimentResultData("ml-mbert", retrieved_docs_per_query, 15)
		
		elif self.experiment_approach == "ml_knowledge_distillation":
			retrieved_docs_per_query = retrieve_k_documents_per_query_distiluse(self.preprocessed_data.queries, self.preprocessed_data.docs, 
                                                                       10, device=self.device)
                  
			return ExperimentResultData("ml_knowledge_distillation", retrieved_docs_per_query, 15)
