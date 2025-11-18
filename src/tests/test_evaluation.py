from llmsat.llmsat import AlgorithmResult, AlgorithmStatus, NOT_INITIALIZED, CodeResult, CodeStatus
from datetime import datetime
from llmsat.evaluation.coder import Coder, CoderConfig

from huggingface_hub import login
from llmsat.utils.aws import update_algorithm_result, update_code_result
from llmsat.pipelines.evaluation import EvaluationPipeline
def main():
    # fake algorithm result
    algorithm_result = AlgorithmResult(
        id="1",
        # whatever the algorithm is
        algorithm="kissat_restarting_policy",
        status=AlgorithmStatus.Generated,
        last_updated=datetime.now(),
        prompt="",
        par2=NOT_INITIALIZED, 
        error_rate=NOT_INITIALIZED,
        code_id_list=[],
        other_metrics={}
    )
    update_algorithm_result(algorithm_result)
    # fake code result
    code_result = CodeResult(
        id="2",
        algorithm_id="1",
        code="return false;",
        status=CodeStatus.Generated,
        last_updated=datetime.now(),
        build_success=None
    )
    update_code_result(code_result)
    evaluation_pipeline = EvaluationPipeline()
    evaluation_pipeline.run_all_solvers("1")


if __name__ == "__main__":
    main()