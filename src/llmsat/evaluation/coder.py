from llmsat.llmsat import CodeResult, AlgorithmResult, AlgorithmStatus, get_id, CodeStatus, setup_logging, get_logger
from datetime import datetime
from typing import List
from llmsat.utils.aws import get_all_algorithm_ids, get_algorithm_result_of_status, get_all_algorithm_results, get_algorithm_result, update_algorithm_result, update_code_result
from llmsat.utils.utils import wrap_command_to_slurm
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import argparse
from copy import deepcopy
import torch
import re


def get_algorithms() -> List[str]:
    algorithm_response_path = "data/algorithm_response.jsonl"
    algorithms = []
    with open(algorithm_response_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            algorithm = json.loads(line)
            algorithms.append(algorithm)
    # parse only the useful information
    return algorithms

class CoderConfig:
    def __init__(self, code_path: str):
        self.model_name = "gpt-oss-20b"
        self.model_path = f"models/{self.model_name}"
        self.code_path = code_path

class Coder:
    def __init__(self,config: CoderConfig):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
    def load_model(self):
        # download the model from huggingface if not already downloaded
        if not os.path.exists(self.config.model_path):
            os.makedirs(self.config.model_path, exist_ok=True)

        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise RuntimeError("transformers is required to load the model")
        self.tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", cache_dir=self.config.model_path)
        use_cuda = torch.cuda.is_available()
        model_kwargs = {}
        if use_cuda:
            model_kwargs.update({"device_map": "auto", "torch_dtype": torch.bfloat16})
        self.model = AutoModelForCausalLM.from_pretrained(
            "openai/gpt-oss-20b",
            cache_dir=self.config.model_path,
            **model_kwargs,
        )
        if use_cuda and getattr(self.model, "device", None) is None:
            self.model.to("cuda")
        self.model.eval()

    def get_algorithms_to_evaluate(self, status: AlgorithmStatus) -> List[AlgorithmResult]:
        self.logger.info(f"Getting algorithms to evaluate with status {status}")
        algorithms = get_algorithm_result_of_status(status)
        return algorithms

    def parse_response(self, response: str) -> str:
        # Prefer <code>...</code> tags first
        code_tag = re.search(r"<code[^>]*>([\s\S]*?)</code>", response, re.IGNORECASE)
        if code_tag:
            return code_tag.group(1).strip()
        # Plain JSON with {"code": "..."} or other variants
        try:
            obj = json.loads(response)
            return obj.get("code")
        except Exception:
            pass
        # Try to extract a JSON object with a "code" field from noisy text
        decoder = json.JSONDecoder()
        for start_idx, ch in enumerate(response):
            if ch != "{":
                continue
            try:
                obj, end_idx = decoder.raw_decode(response[start_idx:])
                if isinstance(obj, dict) and "code" in obj:
                    return obj.get("code")
            except Exception:
                continue
        # Fallback: try to extract content inside a fenced code block
        fence_match = re.search(r"```(?:[a-zA-Z0-9_+.-]*)?\s*([\s\S]*?)```", response)
        if fence_match:
            return fence_match.group(1).strip()
        self.logger.error("Unable to parse code from model response.")
        return None
    
    def start_evaluation(self, first_n: int = 10) -> None:
        algorithms = self.get_algorithms_to_evaluate(AlgorithmStatus.Generated)
        length = len(algorithms)
        if length == 0:
            self.logger.info("No algorithms to evaluate")
            return
        self.load_model()
        first_n = min(first_n, length)
        self.logger.info(f"Starting evaluation for {first_n} algorithms out of {length}")
        for algorithm in algorithms[:first_n]:
            # print(algorithm.algorithm)
            algorithm_result = deepcopy(algorithm)
            algorithm_result.code_id_list = []
            for i in range(10):
                code = self.generate_code(algorithm)
                if code is None:
                    continue
                code_result = CodeResult(
                    id=get_id(code),
                    algorithm_id=algorithm.id,
                    code=code,
                    status=CodeStatus.Generated,
                    last_updated=datetime.now(),
                    build_success=None
                )
                update_code_result(code_result)
                algorithm_result.code_id_list.append(code_result.id)
            algorithm_result.status = AlgorithmStatus.CodeGenerated
            update_algorithm_result(algorithm_result)
    
    def generate_code(self, algorithm: str) -> str:
        # return "return false;" # TODO: test only
        prompt = self.create_prompt(algorithm)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # Move tensors to the same device as the model (GPU if available)
        try:
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception:
            pass
        outputs = self.model.generate(**inputs, max_new_tokens=2048)
        # Decode only the newly generated tokens (exclude the prompt tokens)
        output_ids = outputs[0].tolist()
        prompt_token_count = inputs["input_ids"].shape[-1]
        new_token_ids = output_ids[prompt_token_count:]
        code_response = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        self.logger.info(f"Code response: {code_response}")
        code = self.parse_response(code_response)
        return code

    def create_prompt(self, algorithm: str) -> str:
        kissat_restarting_function = """
        bool kissat_restarting (kissat *solver) {
            assert (solver->unassigned);
            if (!GET_OPTION (restart))
            return false;
            if (!solver->level)
            return false;
            if (CONFLICTS < solver->limits.restart.conflicts)
            return false;
            if (solver->stable)
            return kissat_reluctant_triggered (&solver->reluctant);
            const double fast = AVERAGE (fast_glue);
            const double slow = AVERAGE (slow_glue);
            const double margin = (100.0 + GET_OPTION (restartmargin)) / 100.0;
            const double limit = margin * slow;
            kissat_extremely_verbose (solver,
                                        "restart glue limit %g = "
                                        "%.02f * %g (slow glue) %c %g (fast glue)",
                                        limit, margin, slow,
                                        (limit > fast    ? '>'
                                        : limit == fast ? '='
                                                        : '<'),
                                        fast);
            return (limit <= fast);
            }
        """
        prompt = f"""
        You are an expert SAT solver developer that generates compilable source code for a given algorithm in Kissat solver.
        Your final code must be the kissat_restarting function with complete signature and return type. That is the only output you should output. The code will be replacing the kissat_restarting function in the restart.c file.
        You MUST give your final code in following format (replace the content inside <code> and </code> with your code):
        \n<code>
         (your code here)
        \n</code>
        - Your output must be less than 2048 tokens.
        - You must only use attributes and functions that are defined in Kissat solver and available in restarting function.
        Algorithm specification:
        {algorithm}
        
        Reference (Kissat SAT restarting function):
        {kissat_restarting_function}
        """
        return prompt

    def save_code(self, code: str) -> None:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first_n", type=int, default=100)
    args = parser.parse_args()
    setup_logging()
    coder = Coder(CoderConfig(code_path="data/code"))
    coder.start_evaluation(first_n=args.first_n)

if __name__ == "__main__":
    main()