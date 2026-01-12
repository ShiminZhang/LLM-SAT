"""
Function injector for extracting and replacing functions in Kissat solver copies.

Provides methods to:
- Extract function source code from the base solver (for coder prompts)
- Parse LLM-generated code output
- Replace functions in solver copies with generated code
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .registry import FunctionRegistry

logger = logging.getLogger(__name__)


@dataclass
class ParsedCode:
    """Result of parsing LLM-generated code output."""

    function_name: str
    """Name of the target function (from the function attribute)."""

    code: str
    """The extracted function code."""


class FunctionInjector:
    """
    Handles extraction and injection of functions in Kissat solver copies.

    Example:
        registry = FunctionRegistry("solvers/base/function_registry.yaml")
        injector = FunctionInjector(registry, "solvers/base")

        # Extract function for coder prompt
        source = injector.extract_function("kissat_restarting")

        # Parse LLM output
        parsed = injector.parse_llm_output(llm_response)

        # Inject into solver copy
        injector.replace_function("solvers/algorithm_1/code_1", parsed.function_name, parsed.code)
    """

    def __init__(self, registry: FunctionRegistry, base_solver_path: str | Path):
        """
        Initialize the injector.

        Args:
            registry: Function registry with location information.
            base_solver_path: Path to the base Kissat solver.
        """
        self.registry = registry
        self.base_path = Path(base_solver_path)

        if not self.base_path.exists():
            raise FileNotFoundError(f"Base solver not found: {self.base_path}")

    def extract_function(self, func_name: str) -> str:
        """
        Extract a function's source code from the base solver.

        Used to include reference code in coder prompts.

        Args:
            func_name: Name of the function to extract.

        Returns:
            The function's source code as a string.

        Raises:
            KeyError: If the function is not in the registry.
            FileNotFoundError: If the source file doesn't exist.
        """
        info = self.registry[func_name]  # Raises KeyError if not found

        file_path = self.base_path / info.file
        if not file_path.exists():
            raise FileNotFoundError(
                f"Source file not found: {file_path} (for function {func_name})"
            )

        lines = file_path.read_text().splitlines(keepends=True)

        # Line numbers are 1-indexed, Python lists are 0-indexed
        # end_line is inclusive, so we use it directly as the slice end
        start_idx = info.start_line - 1
        end_idx = info.end_line

        if start_idx < 0 or end_idx > len(lines):
            raise ValueError(
                f"Invalid line range for {func_name}: {info.start_line}-{info.end_line} "
                f"(file has {len(lines)} lines)"
            )

        return "".join(lines[start_idx:end_idx])

    def parse_llm_output(
        self, llm_output: str, expected_function: Optional[str] = None
    ) -> ParsedCode:
        """
        Parse LLM-generated code output.

        Handles two formats:
        1. New format: <code function="func_name">...</code>
        2. Legacy format: <code>...</code> (requires expected_function)

        Also normalizes escaped whitespace from JSON responses.

        Args:
            llm_output: Raw LLM response containing code.
            expected_function: Expected function name (required for legacy format,
                              optional for validation with new format).

        Returns:
            ParsedCode with function name and extracted code.

        Raises:
            ValueError: If parsing fails or function name mismatch.
        """
        # Normalize escaped whitespace (common in JSON responses)
        code = self._normalize_whitespace(llm_output)

        # Try to extract from <code> tags
        code_match = re.search(
            r'<code(?:\s+function=["\']([^"\']+)["\'])?\s*>(.*?)</code>',
            code,
            re.DOTALL,
        )

        if not code_match:
            # Legacy: no <code> tags, try to find raw function
            if expected_function:
                logger.warning("No <code> tags found, attempting raw extraction")
                extracted = self._extract_function_from_raw(code, expected_function)
                if extracted:
                    return ParsedCode(function_name=expected_function, code=extracted)
            raise ValueError("No <code>...</code> block found in LLM output")

        func_attr = code_match.group(1)  # May be None for legacy format
        code_content = code_match.group(2).strip()

        # Determine function name
        if func_attr:
            func_name = func_attr
            # Validate against expected if provided
            if expected_function and func_name != expected_function:
                raise ValueError(
                    f"Function name mismatch: expected '{expected_function}', "
                    f"got '{func_name}' in <code> tag"
                )
        elif expected_function:
            func_name = expected_function
        else:
            raise ValueError(
                "No function attribute in <code> tag and no expected_function provided"
            )

        # Extract just the function if the code contains more
        extracted = self._extract_function_from_raw(code_content, func_name)
        if extracted:
            code_content = extracted

        # Validate the code looks like a function definition
        if not self._looks_like_function(code_content, func_name):
            logger.warning(
                f"Extracted code may not be a valid function definition for {func_name}"
            )

        return ParsedCode(function_name=func_name, code=code_content)

    def replace_function(
        self, solver_path: str | Path, func_name: str, new_code: str
    ) -> None:
        """
        Replace a function in a solver copy with new code.

        Args:
            solver_path: Path to the solver copy (not the base solver).
            func_name: Name of the function to replace.
            new_code: New function code to inject.

        Raises:
            KeyError: If the function is not in the registry.
            FileNotFoundError: If the source file doesn't exist.
            ValueError: If the line range is invalid.
        """
        solver_path = Path(solver_path)
        info = self.registry[func_name]

        file_path = solver_path / info.file
        if not file_path.exists():
            raise FileNotFoundError(
                f"Source file not found: {file_path} (for function {func_name})"
            )

        lines = file_path.read_text().splitlines(keepends=True)

        start_idx = info.start_line - 1
        end_idx = info.end_line

        if start_idx < 0 or end_idx > len(lines):
            raise ValueError(
                f"Invalid line range for {func_name}: {info.start_line}-{info.end_line} "
                f"(file has {len(lines)} lines)"
            )

        # Ensure new_code ends with newline
        if not new_code.endswith("\n"):
            new_code += "\n"

        # Replace the function
        new_lines = lines[:start_idx] + [new_code] + lines[end_idx:]

        file_path.write_text("".join(new_lines))

        logger.info(
            f"Replaced {func_name} in {file_path} "
            f"(was lines {info.start_line}-{info.end_line})"
        )

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize escaped whitespace from JSON responses."""
        # Handle common escape sequences
        text = text.replace("\\n", "\n")
        text = text.replace("\\t", "\t")
        text = text.replace("\\r", "\r")
        text = text.replace('\\"', '"')
        text = text.replace("\\'", "'")
        text = text.replace("\\\\", "\\")
        # Normalize Windows line endings
        text = text.replace("\r\n", "\n")
        return text

    def _extract_function_from_raw(
        self, code: str, func_name: str
    ) -> Optional[str]:
        """
        Extract a function from raw code using brace matching.

        Handles cases where the LLM includes extra code before/after the function.
        """
        # Pattern to find function definition start
        # Handles: static, inline, return types, pointers
        header_pattern = rf"(?:static\s+)?(?:inline\s+)?[\w\s\*]+\b{re.escape(func_name)}\s*\([^)]*\)\s*\{{"

        match = re.search(header_pattern, code, re.DOTALL)
        if not match:
            return None

        start = match.start()
        open_brace = match.end() - 1  # Points at '{'

        # Count braces to find matching close
        brace_count = 0
        i = open_brace

        while i < len(code):
            char = code[i]
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    return code[start:end]
            i += 1

        logger.warning(f"Could not find matching brace for {func_name}")
        return None

    def _looks_like_function(self, code: str, func_name: str) -> bool:
        """Quick validation that code looks like a function definition."""
        # Should contain the function name followed by parentheses
        if not re.search(rf"\b{re.escape(func_name)}\s*\(", code):
            return False
        # Should have balanced braces
        if code.count("{") != code.count("}"):
            return False
        # Should have at least one opening brace
        if "{" not in code:
            return False
        return True
