from llmsat.llmsat import *
from llmsat.utils.aws import update_algorithm_result

def parse_response(prompt: str, response: str) -> AlgorithmResult:
    # parse the llm response and return the algorithm result
    algorithm = ""

    # 
    algorithm_id = get_id(algorithm)
    AlgorithmResult(
        algorithm_id=algorithm_id,
        prompt=prompt,
        par2=0,
        error_rate=0,
        other_metrics={}
    )

    return algorithm_result

def store_result(result: AlgorithmResult) -> None:
    update_algorithm_result(result)