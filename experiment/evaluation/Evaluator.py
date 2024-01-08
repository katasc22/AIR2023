from ..data.ExperimentData import ExperimentValidationData
from ..data.DataHandler import DataHandler

from sklearn.metrics import precision_score

class Evaluator:
	def __init__(self, experiment_result, dataHandler: DataHandler):
		self.experiment_result = experiment_result
		self.dataHandler = dataHandler

		self.qrels = dataHandler.get_qrels()
		self.convert_ground_truth_ids_to_int()


	def evaluate_experiment(self, ex_result):
		retrieved_docs_per_query = ex_result.experiment_result

		precision_scores = []
		average_precision_scores = []
		recall_scores = []
		f1_scores = []
		for q_id, docs in retrieved_docs_per_query.items():
			precision = self.calc_precision(q_id, docs)
			precision_scores.append(precision)

			average_precision = self.calc_average_precision(q_id, docs)
			average_precision_scores.append(average_precision)

			recall = self.calc_recall(q_id, docs)
			recall_scores.append(recall)

			f1_score = self.calc_f1_score(precision, recall)
			f1_scores.append(f1_score)

		precision_score_mean = sum(precision_scores) / len(precision_scores)
		recall_score_mean = sum(recall_scores) / len(recall_scores)
		f1_score_mean = sum(f1_scores) / len(f1_scores)
		mean_average_precision = sum(average_precision_scores) / len(average_precision_scores)

		print(f"Precision: {precision_score_mean} - Recall: {recall_score_mean} - F1score: {f1_score_mean} - MAP: {mean_average_precision}")

		return ExperimentValidationData(ex_result.experiment_approach)


	def evaluate(self):
		validation_result = self.evaluate_experiment(self.experiment_result)

		return validation_result
	

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
	
	
	def calc_average_precision(self, q_id, retrieved_docs):
		ground_truth_docs = self.get_ground_truth_docs_per_query(q_id)
		precisions_at_k = []
		number_of_retrieved_relevant_items = 0
		for index, doc in enumerate(retrieved_docs):
			k = index + 1
			if doc in ground_truth_docs:
				number_of_retrieved_relevant_items += 1
				precision_at_k = self.calc_precision_at_k(ground_truth_docs, retrieved_docs, k)
				precisions_at_k.append(precision_at_k)

		if number_of_retrieved_relevant_items == 0:
			return 0 
		else:
			return sum(precisions_at_k) / number_of_retrieved_relevant_items
			

	def calc_precision_at_k(self, ground_truth_docs, retrieved_docs, k: int):
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