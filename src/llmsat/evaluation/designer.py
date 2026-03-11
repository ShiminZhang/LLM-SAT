from llmsat.llmsat import *
from llmsat.utils.aws import update_algorithm_result

def parse_response(prompt: str, response: str) -> AlgorithmResult:
    # parse the llm response and return the algorithm result
    algorithm = ""

    #
    algorithm_id = get_id(algorithm)
    AlgorithmResult(
        id=algorithm_id,
        function_name="",
        description="",
        role=Role.LEADER,
        status=AlgorithmStatus.Generated,
        last_updated="",
        code_id_list=[],
        prompt=prompt,
        other_metrics={}
    )

    return algorithm_result

def store_result(result: AlgorithmResult) -> None:
    update_algorithm_result(result)