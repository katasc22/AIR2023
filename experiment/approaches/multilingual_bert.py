from transformers import BertTokenizer, BertModel
from sentence_transformers import util
import torch
import time
from memory_profiler import profile

"""
Mean pooling code is taken from the SBERT documentation.
(https://www.sbert.net/examples/applications/computing-embeddings/README.html)
"""
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

@profile
def retrieve_k_documents_per_query_mbert(queries, documents, k, device):
	start_time = time.time()
	model = BertModel.from_pretrained("bert-base-multilingual-cased")
	model.to(device)
	tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
	print("[Approach: mBert] Computing document embeddings ...")
	batch_size = 10
	num_docs = len(documents["text"])

	batch_embeddings = []
	for i in range(0, num_docs, batch_size):
		batch = documents["text"].iloc[i:i + batch_size].tolist()
    
		encoded_docs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
		encoded_docs.to(device)
		output_docs = model(**encoded_docs)
		
		doc_embeds_batch = mean_pooling(output_docs, encoded_docs['attention_mask']).detach().cpu()
		batch_embeddings.append(doc_embeds_batch)
    
	doc_embeds = torch.cat(batch_embeddings, dim=0)

	retrieved_docs_per_query = {}
	print("[Approach: mBert] Retrieve documents per query ...")
	for _, query in queries.iterrows():
		encoded_query = tokenizer(query["text"], padding=True, truncation=True, return_tensors='pt')
		encoded_query.to(device)
		output_query = model(**encoded_query)

		query_embed = mean_pooling(output_query, encoded_query['attention_mask']).detach().cpu()

		similarity_scores = util.cos_sim(query_embed, doc_embeds)

		retrieved_results = torch.topk(similarity_scores, k)

		doc_indices = retrieved_results[1].squeeze().tolist()

		retrieved_docs = documents.iloc[doc_indices]

		retrieved_docs_per_query[query["query_id"]] = retrieved_docs["doc_id"].tolist()

	print("[Approach: mBert] Finished.")
	end_time = time.time()
	print(f"[Approach: mBert] Execution Time: {end_time - start_time} seconds")
	print(torch.cuda.max_memory_allocated())
	return retrieved_docs_per_query