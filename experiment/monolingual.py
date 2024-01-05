from sentence_transformers import SentenceTransformer
import numpy as np
from data import data_handling
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def retrieve_k_documents(k, documents, query):

    ### Delete below (only for testing purposes) ###
    #documents = ["Helly my name is John", "I have a dog", "I do not take my dog for a walk", "DummyQuery", "DummyQuery also", "Dummy query like this"]
    #query = ["I take my dog for a walk"]
    ### Until here ###
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

    print("Computing embeddings..")
    embDocs = model.encode(documents["text"])
    embQuers = model.encode(query["text"])
    embQuer = embQuers[0]

    print("Computing similarities...")
    similarity = cosine_similarity(embQuer.reshape(1, -1), embDocs)

    # Get indices and score of top k documents
    top_simil_indices = np.argsort(similarity[0])[::-1][:k]
    top_simil_score = similarity[0][top_simil_indices]

    # Retrieve top k documents
    top_k_docs = documents.iloc[top_simil_indices]

    print("Top ", k, "matching documents for query: ", query["text"].iloc[0])
    print(top_k_docs)

    # Creating a dataframe
    result_df = pd.DataFrame({
        'doc_id': top_k_docs['doc_id'].values,
        'text': top_k_docs['text'].values,
        'Similarity Score': top_simil_score
    })
    print(result_df)
    return result_df


documents, queries, qrels = data_handling.loadData()

### For testing purposes using k = 2 here later change to k = 5 ###
retrieve_k_documents(5, documents, queries)
