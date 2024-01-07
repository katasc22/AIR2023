from ..data.ExperimentData import ExperimentValidationData
from ..data.DataHandler import DataHandler

from sklearn.metrics import precision_score

class Evaluator:
	def __init__(self, experiment_results, dataHandler: DataHandler):
		self.experiment_results = experiment_results
		self.dataHandler = dataHandler

		self.qrels = dataHandler.get_qrels()
		self.convert_ground_truth_ids_to_int()


	def evaluate_experiment(self, ex_result):
		retrieved_docs_per_query = ex_result.experiment_result

		precision_scores = []
		recall_scores = []
		f1_scores = []
		for q_id, docs in retrieved_docs_per_query.items():
			precision = self.calc_precision(q_id, docs)
			precision_scores.append(precision)
			#print(f"Precision: {precision}")
			recall = self.calc_recall(q_id, docs)
			recall_scores.append(recall)
			#print(f"Recall: {recall}")
			f1_score = self.calc_f1_score(precision, recall)
			f1_scores.append(f1_score)

		avg_precision_score = sum(precision_scores) / len(precision_scores)
		avg_recall_score = sum(recall_scores) / len(recall_scores)
		avg_f1_score = sum(f1_scores) / len(f1_scores)

		print(f"Precision: {avg_precision_score} - Recall: {avg_recall_score} - F1score: {avg_f1_score}")

		return ExperimentValidationData(ex_result.experiment_approach)


	def evaluate(self):
		validation_results = []

		for ex_result in self.experiment_results:
			ex_validation_data = self.evaluate_experiment(ex_result)
			validation_results.append(ex_validation_data)

		return tuple(validation_results)
	

	def calc_f1_score(self, precision, recall):
		if (precision + recall) == 0:
			return 0 
		else:
			return (2 * precision * recall) / (precision + recall)


	def calc_precision(self, q_id, retrieved_docs):
		ground_truth_docs = self.get_ground_truth_docs_per_query(q_id)
		#print(ground_truth_docs)
		#print("---------------------")
		#print(retrieved_docs)
		true_positives = 0
		false_positives = 0
		for doc in retrieved_docs:
			if doc in ground_truth_docs:
				true_positives += 1
			else:
				false_positives += 1
                
		return true_positives / (true_positives + false_positives)


	def calc_precision_at_k(self, q_id, retrieved_docs, k: int):
		ground_truth_docs = self.get_ground_truth_docs_per_query(q_id)
		print(ground_truth_docs)
		true_positives_at_k = 0
		false_positives_at_k = 0
		for doc in retrieved_docs[:k]:
			if doc in ground_truth_docs:
				true_positives_at_k += 1
			else:
				false_positives_at_k += 1
                
		return true_positives_at_k / (true_positives_at_k + false_positives_at_k)


	def calc_recall(self, q_id, retrieved_docs):
		ground_truth_docs = self.get_ground_truth_docs_per_query(q_id)
		true_positives = 0
	
		for doc in retrieved_docs:
			if doc in ground_truth_docs:
				true_positives += 1

		return true_positives / len(ground_truth_docs)
	

	def calc_recall_at_k(self, q_id, retrieved_docs, k: int):
		ground_truth_docs = self.get_ground_truth_docs_per_query(q_id)
		true_positives_at_k = 0
		false_negatives_at_k = 0
		for doc in retrieved_docs[:k]:
			if doc in ground_truth_docs:
				true_positives_at_k += 1

		for doc in retrieved_docs[k:]:
			if doc in ground_truth_docs:
				false_negatives_at_k += 1

		return true_positives_at_k / (true_positives_at_k + false_negatives_at_k)
	

	def convert_ground_truth_ids_to_int(self):
		self.qrels["query_id"] = self.qrels["query_id"].astype(int)
		self.qrels["doc_id"] = self.qrels["doc_id"].astype(int)


	def get_ground_truth_docs_per_query(self, q_id):
		return self.qrels.loc[self.qrels["query_id"] == q_id]["doc_id"].tolist()