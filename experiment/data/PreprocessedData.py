import pandas as pd
from dataclasses import dataclass

@dataclass
class PreprocessedData():
	docs: pd.DataFrame
	queries: pd.DataFrame