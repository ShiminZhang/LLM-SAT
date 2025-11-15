from llmsat.llmsat import TaskStatus, CodeResult
from datetime import datetime

from llmsat.utils.utils import wrap_command_to_slurm


class CoderConfig:
    def __init__(self, code_path: str):
        self.model_name = "gpt-oss-20b"
        self.model_path = f"models/{self.model_name}"
        self.code_path = code_path

class Coder:
    def __init__(self):
        self.code_path = code_path

    def load_model(self):
        pass

    def parse_response(self, response: str) -> str:
        pass

    def generate_code(self, algorithm: str) -> str:
        prompt = self.create_prompt(algorithm)
        code_response = self.model.generate(prompt)
        code = self.parse_response(code_response)
        code_result = CodeResult(
            task_id=task_id,
            status=TaskStatus.Completed,
            created_at=datetime.now(),
            algorithm_id=algorithm,
            code=code,
            solver_id=solver_id,
            build_success=build_success
        )

        return code

    def create_prompt(self, algorithm: str) -> str:
        pass

    def save_code(self, code: str) -> None:
        pass