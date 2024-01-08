import torch
import time
from sentence_transformers import SentenceTransformer, util
from memory_profiler import profile

@profile
def retrieve_k_documents_per_query_distiluse(queries, documents, k, device):
	start_time = time.time()
	# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', device=device)
	model = SentenceTransformer('sentence-transformers/sentence-transformers/distiluse-base-multilingual-cased-v2', device=device)

	print("[Approach: Distiluse] Computing embeddings..")
	doc_embs = model.encode(documents["text"], convert_to_tensor=True)
	
	#print(doc_embs.shape)
	retrieved_docs_per_query = {}
	print("[Approach: Distiluse] Retrieve documents per query ...")
	for _, query in queries.iterrows():
		query_embed = model.encode(query["text"], convert_to_tensor=True)

		similarity_scores = util.cos_sim(query_embed, doc_embs)

		retrieved_results = torch.topk(similarity_scores, k)

		doc_indices = retrieved_results[1].squeeze().tolist()

		retrieved_docs = documents.iloc[doc_indices]

		retrieved_docs_per_query[query["query_id"]] = retrieved_docs["doc_id"].tolist()

	print("[Approach: Distiluse] Finished.")
	end_time = time.time()
	print(f"[Approach: Distiluse] Execution Time: {end_time - start_time} seconds")
	print(torch.cuda.max_memory_allocated())
	return retrieved_docs_per_query
