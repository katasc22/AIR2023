from dataclasses import dataclass

@dataclass
class ExperimentResultData:
	experiment_approach: str
	experiment_result: dict[int, int]
	k: int
	
@dataclass
class ExperimentValidationData:
	experiment_approach: str
	#TODO add metrics