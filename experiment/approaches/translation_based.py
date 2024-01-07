from sentence_transformers import SentenceTransformer, util
from torch import topk

# @profile
def retrieve_k_documents_per_query_tb_monolingual(queries, documents, k, device):

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

    return retrieved_docs_per_query