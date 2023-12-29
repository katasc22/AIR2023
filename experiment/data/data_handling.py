import ir_datasets
import pandas as pd

#size: num of dataentries to retrieve
def loadData():
    dataset = ir_datasets.load("vaswani")
    docs = pd.DataFrame(dataset.docs_iter())
    queries = pd.DataFrame(dataset.queries_iter())
    qrels = pd.DataFrame(dataset.qrels_iter())
    return docs, queries, qrels

