#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from llmsat.code_injection import FunctionRegistry, FunctionInjector


TARGET_FUNCTION = "kissat_restarting"
DEFAULT_REGISTRY_PATH = "solvers/base/function_registry.yaml"
DEFAULT_BASE_SOLVER_PATH = "solvers/base"
DEFAULT_CODE_PROMPT_TEMPLATE = "data/prompts/kissat_mab_code.txt"


@dataclass
class CaseResult:
	name: str
	solver_dir: Path
	modified_file: Path
	build_ok: bool
	build_log: Path


def _repo_root() -> Path:
	# This file is scripts/smoke_test_insertion.py
	return Path(__file__).resolve().parents[1]


def _run_cmd(cmd: list[str], cwd: Path, log_fh) -> int:
	proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
	log_fh.write(f"$ {' '.join(cmd)}\n")
	log_fh.write("=== stdout ===\n")
	log_fh.write(proc.stdout or "")
	log_fh.write("\n=== stderr ===\n")
	log_fh.write(proc.stderr or "")
	log_fh.write("\n\n")
	return proc.returncode


def _compile_solver(solver_dir: Path, build_log: Path, jobs: int) -> bool:
	build_log.parent.mkdir(parents=True, exist_ok=True)
	with build_log.open("w", encoding="utf-8") as f:
		rc1 = _run_cmd(["./configure"], cwd=solver_dir, log_fh=f)
		if rc1 != 0:
			return False
		rc2 = _run_cmd(["make", f"-j{jobs}"], cwd=solver_dir, log_fh=f)
		if rc2 != 0:
			return False

	return (solver_dir / "build" / "kissat").exists()


def _copy_base_solver(base_solver: Path, dest: Path) -> None:
	if dest.exists():
		shutil.rmtree(dest)
	shutil.copytree(base_solver, dest)


def _safe_edit(original_code: str) -> str:
	"""Make a tiny, very safe edit that should not change behavior."""
	assert_re = re.compile(r"^\s*assert\s*\(\s*solver->unassigned\s*\)\s*;\s*$")

	lines = original_code.splitlines(keepends=True)
	for i, line in enumerate(lines):
		if assert_re.match(line):
			insertion = (
				"\n"
				"  // smoke-test no-op: unreachable\n"
				"  if (0) return false;\n"
			)
			lines.insert(i + 1, insertion)
			return "".join(lines)

	raise ValueError("Original code did not contain an assert(solver->unassigned) line")


def _wrap_llm_response_in_code_tags(text: str) -> str:
	# Some saved outputs might contain only the function body, some contain <code> tags.
	if "<code" in text and "</code>" in text:
		return text
	return f"<code>\n{text.strip()}\n</code>\n"


def _generate_llm_code(code_prompt_template_path: Path, algorithm_text: str) -> str:
	"""Generate code for TARGET_FUNCTION using the repo's OpenAI helper."""
	from llmsat.utils.chatgpt_helper import get_response_from_chatgpt

	template = code_prompt_template_path.read_text(encoding="utf-8")
	try:
		prompt = template.format(algorithm=algorithm_text)
	except Exception:
		# Fallback: append algorithm text without formatting.
		prompt = f"{template}\n\nAlgorithm:\n{algorithm_text}"
	return get_response_from_chatgpt(prompt)


def _extract_text_from_openai_batch_line(obj: dict) -> str:
	"""Extract assistant text from an OpenAI batch output JSONL line.

	This matches the shape used in outputs like:
	outputs/<tag>/batch_<id>/leaders_output.txt
	"""
	resp = obj.get("response") if isinstance(obj, dict) else None
	body = resp.get("body") if isinstance(resp, dict) else None
	output = body.get("output") if isinstance(body, dict) else None
	if isinstance(output, list) and output:
		msg0 = output[0]
		content = msg0.get("content") if isinstance(msg0, dict) else None
		if isinstance(content, list) and content:
			part0 = content[0]
			text = part0.get("text") if isinstance(part0, dict) else None
			if isinstance(text, str) and text.strip():
				return text
	# Fallback: some outputs may provide output_text
	output_text = body.get("output_text") if isinstance(body, dict) else None
	if isinstance(output_text, str) and output_text.strip():
		return output_text
	raise ValueError("Could not extract assistant text from batch output line")


def _extract_json_object_string(text: str) -> str:
	"""Pull the first JSON object string from a response.

	Handles common formats like fenced blocks:
	```json\n{...}\n```
	"""
	# Prefer fenced ```json blocks
	m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
	if m:
		return m.group(1).strip()

	# Otherwise: best-effort slice between first '{' and last '}'
	start = text.find("{")
	end = text.rfind("}")
	if start != -1 and end != -1 and end > start:
		return text[start : end + 1].strip()

	raise ValueError("No JSON object found in text")


def _load_algorithm_text_from_leaders_output(
	leaders_output_path: Path,
	leader_index: int,
) -> str:
	"""Read leaders_output.txt and return the selected leader's algorithm string."""
	from llmsat.data.algorithm_parse import parse_algorithm_spec_json
	import json

	lines = leaders_output_path.read_text(encoding="utf-8").splitlines()
	items = [ln.strip() for ln in lines if ln.strip()]
	if not items:
		raise ValueError(f"No JSONL lines found in {leaders_output_path}")
	if leader_index < 0 or leader_index >= len(items):
		raise ValueError(
			f"leader_index out of range: {leader_index} (have {len(items)} leaders)"
		)

	obj = json.loads(items[leader_index])
	text = _extract_text_from_openai_batch_line(obj)
	json_str = _extract_json_object_string(text)
	spec, target_function = parse_algorithm_spec_json(json_str)

	# Hard lock for this sprint
	if (target_function or spec.get("target_function")) != TARGET_FUNCTION:
		raise ValueError(
			f"Leader target_function must be {TARGET_FUNCTION} but got: {target_function!r}"
		)

	algorithm_text = spec.get("algorithm")
	if not isinstance(algorithm_text, str) or not algorithm_text.strip():
		raise ValueError("Leader spec missing non-empty 'algorithm' field")
	return algorithm_text.strip()


def run_case(
	*,
	case_name: str,
	base_solver: Path,
	registry: FunctionRegistry,
	injector: FunctionInjector,
	new_code: str,
	out_dir: Path,
	jobs: int,
) -> CaseResult:
	case_dir = out_dir / case_name
	solver_dir = case_dir / "solver"
	case_dir.mkdir(parents=True, exist_ok=True)

	_copy_base_solver(base_solver, solver_dir)
	injector.replace_function(solver_dir, TARGET_FUNCTION, new_code)

	func_info = registry[TARGET_FUNCTION]
	modified_file = solver_dir / func_info.file

	build_log = case_dir / "build.log"
	build_ok = _compile_solver(solver_dir, build_log=build_log, jobs=jobs)

	print(f"[{case_name}] modified: {modified_file}")
	print(f"[{case_name}] build_ok={build_ok} log={build_log}")

	return CaseResult(
		name=case_name,
		solver_dir=solver_dir,
		modified_file=modified_file,
		build_ok=build_ok,
		build_log=build_log,
	)


def main(argv: Optional[list[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="Smoke test: inject + compile kissat_restarting")
	parser.add_argument("--registry", default=DEFAULT_REGISTRY_PATH)
	parser.add_argument("--base-solver", default=DEFAULT_BASE_SOLVER_PATH)
	parser.add_argument("--out", default="outputs/smoke_test_insertion")
	parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))

	parser.add_argument(
		"--llm",
		action="store_true",
		help="Run the LLM-generated edit case (requires OPENAI_API_KEY).",
	)
	parser.add_argument(
		"--llm-code-path",
		type=str,
		default=None,
		help="Path to a file containing saved LLM output for the code (with or without <code> tags).",
	)
	parser.add_argument(
		"--llm-algorithm-text",
		type=str,
		default=None,
		help="Algorithm description text for the coder prompt (used with --llm).",
	)
	parser.add_argument(
		"--leaders-output-path",
		type=str,
		default=None,
		help=(
			"Path to an OpenAI batch leaders output JSONL (e.g., "
			"outputs/test_teamleader/batch_<id>/leaders_output.txt). "
			"Used with --llm when --llm-algorithm-text is omitted."
		),
	)
	parser.add_argument(
		"--leader-index",
		type=int,
		default=0,
		help="Which leader line to use from --leaders-output-path (0-based).",
	)
	parser.add_argument(
		"--code-prompt-template",
		type=str,
		default=DEFAULT_CODE_PROMPT_TEMPLATE,
		help="Coder prompt template (defaults to data/prompts/kissat_mab_code.txt).",
	)

	args = parser.parse_args(argv)

	root = _repo_root()
	os.chdir(root)

	registry_path = root / args.registry
	base_solver = root / args.base_solver

	if not registry_path.exists():
		print(f"ERROR: registry not found: {registry_path}", file=sys.stderr)
		print(
			"Hint: python scripts/index_functions.py kissat_restarting --solver solvers/base "
			"--output solvers/base/function_registry.yaml --overwrite"
		)
		return 2

	if not base_solver.exists():
		print(f"ERROR: base solver not found: {base_solver}", file=sys.stderr)
		return 2

	registry = FunctionRegistry(registry_path)
	if TARGET_FUNCTION not in registry:
		print(f"ERROR: {TARGET_FUNCTION} not in registry {registry_path}", file=sys.stderr)
		print(f"Available: {registry.list_functions()}")
		return 2

	injector = FunctionInjector(registry, base_solver)

	run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
	out_dir = root / args.out / run_id
	out_dir.mkdir(parents=True, exist_ok=True)

	print(f"Target function locked to: {TARGET_FUNCTION}")
	print(f"Registry: {registry_path}")
	print(f"Base solver: {base_solver}")
	print(f"Output dir: {out_dir}")

	original = injector.extract_function(TARGET_FUNCTION)

	results: list[CaseResult] = []

	results.append(
		run_case(
			case_name="01_reinsert_original",
			base_solver=base_solver,
			registry=registry,
			injector=injector,
			new_code=original,
			out_dir=out_dir,
			jobs=args.jobs,
		)
	)

	safe_code = _safe_edit(original)
	results.append(
		run_case(
			case_name="02_safe_edit",
			base_solver=base_solver,
			registry=registry,
			injector=injector,
			new_code=safe_code,
			out_dir=out_dir,
			jobs=args.jobs,
		)
	)

	if args.llm_code_path:
		llm_text = Path(args.llm_code_path).read_text(encoding="utf-8")
		llm_text = _wrap_llm_response_in_code_tags(llm_text)
		parsed = injector.parse_llm_output(llm_text, expected_function=TARGET_FUNCTION)
		results.append(
			run_case(
				case_name="03_llm_saved",
				base_solver=base_solver,
				registry=registry,
				injector=injector,
				new_code=parsed.code,
				out_dir=out_dir,
				jobs=args.jobs,
			)
		)
	elif args.llm:
		if not os.environ.get("OPENAI_API_KEY"):
			print("ERROR: --llm requested but OPENAI_API_KEY is not set", file=sys.stderr)
			return 2
		algorithm_text = args.llm_algorithm_text
		if not algorithm_text:
			if not args.leaders_output_path:
				print(
					"ERROR: --llm requires either --llm-algorithm-text or --leaders-output-path",
					file=sys.stderr,
				)
				return 2
			leaders_output_path = root / args.leaders_output_path
			if not leaders_output_path.exists():
				print(f"ERROR: leaders output not found: {leaders_output_path}", file=sys.stderr)
				return 2
			try:
				algorithm_text = _load_algorithm_text_from_leaders_output(
					leaders_output_path=leaders_output_path,
					leader_index=args.leader_index,
				)
			except Exception as e:
				print(f"ERROR: failed to load leader algorithm text: {e}", file=sys.stderr)
				return 2
			(out_dir / "03_llm_leader_algorithm.txt").write_text(algorithm_text, encoding="utf-8")
		code_prompt_template = root / args.code_prompt_template
		if not code_prompt_template.exists():
			print(f"ERROR: code prompt template not found: {code_prompt_template}", file=sys.stderr)
			return 2
		try:
			llm_text = _generate_llm_code(code_prompt_template, algorithm_text)
		except Exception as e:
			print(f"ERROR: LLM generation failed: {e}", file=sys.stderr)
			return 2
		parsed = injector.parse_llm_output(llm_text, expected_function=TARGET_FUNCTION)
		(out_dir / "03_llm_raw.txt").write_text(llm_text, encoding="utf-8")
		results.append(
			run_case(
				case_name="03_llm_generated",
				base_solver=base_solver,
				registry=registry,
				injector=injector,
				new_code=parsed.code,
				out_dir=out_dir,
				jobs=args.jobs,
			)
		)
	else:
		print("Skipping LLM case (use --llm or --llm-code-path)")

	ok = sum(1 for r in results if r.build_ok)
	print(f"\nSummary: {ok}/{len(results)} builds succeeded")
	for r in results:
		print(f"- {r.name}: build_ok={r.build_ok} log={r.build_log}")

	expected_ok = all(
		r.build_ok for r in results if r.name in {"01_reinsert_original", "02_safe_edit"}
	)
	return 0 if expected_ok else 1


if __name__ == "__main__":
	raise SystemExit(main())

