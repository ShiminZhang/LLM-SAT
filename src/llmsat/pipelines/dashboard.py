from llmsat.llmsat import setup_logging, AlgorithmStatus, CodeStatus, AlgorithmResult, CodeResult
from llmsat.utils.aws import get_algorithm_result_of_status, get_code_result_of_status
from llmsat.llmsat import get_logger
import argparse
from typing import List
logger = get_logger(__name__)

def maybe_print_code_ids(code_results: List[CodeResult], print_ids: bool):
    if print_ids:
        for code_result in code_results:
            logger.info(code_result.id)
def maybe_print_algorithm_ids(algorithm_results: List[AlgorithmResult], print_ids: bool):
    if print_ids:
        for algorithm_result in algorithm_results:
            logger.info(algorithm_result.id)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_ids", action="store_true", default=False)
    args = parser.parse_args()
    setup_logging()
    algorithms = get_algorithm_result_of_status(AlgorithmStatus.CodeGenerated)
    logger.info(f"Found {len(algorithms)} algorithms to evaluate")
    maybe_print_algorithm_ids(algorithms, args.show_ids)
    algorithms = get_algorithm_result_of_status(AlgorithmStatus.Evaluating)
    logger.info(f"Found {len(algorithms)} algorithms evaluating")
    maybe_print_algorithm_ids(algorithms, args.show_ids)
    codes = get_code_result_of_status(CodeStatus.Generated)
    logger.info(f"Found {len(codes)} codes generated but not evaluated")
    maybe_print_code_ids(codes, args.show_ids)
    codes = get_code_result_of_status(CodeStatus.Evaluating)
    logger.info(f"Found {len(codes)} codes evaluating")
    maybe_print_code_ids(codes, args.show_ids)
    codes = get_code_result_of_status(CodeStatus.BuildFailed)
    logger.info(f"Found {len(codes)} codes build failed")
    maybe_print_code_ids(codes, args.show_ids)

if __name__ == "__main__":
    main()