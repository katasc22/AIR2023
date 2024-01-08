import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
from experiment.data import DataHandler
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def retrieve_k_documents_per_query_distiluse(queries, documents, k, device):
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
    model.to(device)
    print("[Approach: Distiluse] Computing embeddings..")
    doc_embs = model.encode(documents["text"])
    queries["emb"] = model.encode(queries["text"])
    print("[Approach: Distiluse] Computing similarities...")

    retrieved_docs_per_query = {}
    print("[Approach: Distiluse] Retrieve documents per query ...")
    for _, query in queries.iterrows():
        similarity_scores = util.cos_sim(query["emb"], doc_embs)

        retrieved_results = torch.topk(similarity_scores, k)

        doc_indices = retrieved_results[1].squeeze().tolist()

        retrieved_docs = documents.iloc[doc_indices]

        retrieved_docs_per_query[query["query_id"]] = retrieved_docs["doc_id"].tolist()

    print("[Approach: Distiluse] Finished.")
    return retrieved_docs_per_query
