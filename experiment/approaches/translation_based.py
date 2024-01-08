from ..preprocessing.TranslationHandler import TranslationHandler

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from torch import topk, cuda
import time
from memory_profiler import profile


def classify_source_lang_and_translate(translationHandler: TranslationHandler, lang_classifier, text):
	output = lang_classifier(text, top_k=1, truncation=True) 
	source_lang = output[0]["label"]
	print(source_lang)
    
	if source_lang == "en":
		return text
	else:
		translated_text = translationHandler.translate(source_lang, "en", text)
		return translated_text

@profile
def retrieve_k_documents_per_query_tb_multilingual(translationHandler: TranslationHandler, queries, documents, k, device):
	start_time = time.time()
	model_checkpoint = "papluca/xlm-roberta-base-language-detection"
	lang_classifier = pipeline("text-classification", model=model_checkpoint, device=device)

	if translationHandler.translation_target == "queries":
		queries["text"] = queries["text"].apply(lambda txt: classify_source_lang_and_translate(translationHandler, lang_classifier, txt))
	elif translationHandler.translation_target == "docs":
		documents["text"] = documents["text"].apply(lambda txt: classify_source_lang_and_translate(translationHandler, lang_classifier, txt))
		
	model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2', device=device)

	print("[Approach: TranslationBased] Computing document embeddings ...")
	doc_embeds = model.encode(documents["text"], convert_to_tensor=True)

	retrieved_docs_per_query = {}
	print("[Approach: TranslationBased] Retrieve documents per query ...")
	for _, query in queries.iterrows():
		query_embed = model.encode(query["text"], convert_to_tensor=True)

		similarity_scores = util.cos_sim(query_embed, doc_embeds)

		retrieved_results = topk(similarity_scores, k)

		doc_indices = retrieved_results[1].squeeze().tolist()
		
		retrieved_docs = documents.iloc[doc_indices]
		
		retrieved_docs_per_query[query["query_id"]] = retrieved_docs["doc_id"].tolist()

	
	print("[Approach: TranslationBased] Finished.")
	end_time = time.time()
	print(f"[Approach: TranslationBased] Execution Time: {end_time - start_time} seconds")
	print(cuda.max_memory_allocated())

	return retrieved_docs_per_query

@profile
def retrieve_k_documents_per_query_tb_monolingual(queries, documents, k, device):
	start_time = time.time()
	model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2', device=device)
    
	print("[Approach: TranslationBased] Computing document embeddings ...")
	doc_embeds = model.encode(documents["text"], convert_to_tensor=True)

	retrieved_docs_per_query = {}
	print("[Approach: TranslationBased] Retrieve documents per query ...")
	for _, query in queries.iterrows():
		query_embed = model.encode(query["text"], convert_to_tensor=True)

		similarity_scores = util.cos_sim(query_embed, doc_embeds)

		retrieved_results = topk(similarity_scores, k)

		doc_indices = retrieved_results[1].squeeze().tolist()
		
		retrieved_docs = documents.iloc[doc_indices]
		
		retrieved_docs_per_query[query["query_id"]] = retrieved_docs["doc_id"].tolist()

	print("[Approach: TranslationBased] Finished.")
	end_time = time.time()
	print(f"[Approach: TranslationBased] Execution Time: {end_time - start_time} seconds")
	print(cuda.max_memory_allocated())

	return retrieved_docs_per_query