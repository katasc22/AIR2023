from pandas import DataFrame
from dataclasses import dataclass

@dataclass
class PreprocessedData():
	queries: DataFrame
	docs: DataFrame