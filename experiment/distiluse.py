from sentence_transformers import SentenceTransformer
import numpy as np
from data import data_handling
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def get_similarities(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))


def retrieve_k_documents(k, documents, queries):
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
    print("Computing embeddings..")
    doc_embs = model.encode(documents["text"])
    query_embs = model.encode(queries["text"])
    query_emb = query_embs[0]                   #TODO: query(ies) as param?
    print("Computing similarities...")

    # print("Doc: ",doc_embs.shape)
    # print("Query: ",query_emb.reshape(1,-1).shape)
    similarities = cosine_similarity(query_emb.reshape(1, -1), doc_embs)

    #get the indices and scores of the top 5 documents
    top_similar_docs_indices = np.argsort(similarities[0])[::-1][:k]  # [::-1] -> sorts in reverse order
    top_similar_scores = similarities[0][top_similar_docs_indices]

    # Retrieve the top 5 documents based on similarities
    top_k_docs = documents.iloc[top_similar_docs_indices]

    print("Top ", k, " matching documents for query: ", queries["text"].iloc[0])
    print(top_k_docs)

    # Create a DataFrame with documents and their similarity scores
    result_df = pd.DataFrame({
        'doc_id': top_k_docs['doc_id'].values,
        'text': top_k_docs['text'].values,
        'Similarity Score': top_similar_scores
    })
    print(result_df)
    return result_df


documents, queries, qrels = data_handling.loadData()     #TODO: remove (is already in air-experiment)
retrieve_k_documents(5, documents,queries)
