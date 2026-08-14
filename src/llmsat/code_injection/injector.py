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
                # Raw extraction also failed (e.g. already-cleaned bare function body
                # stored in DB with unbalanced braces or non-standard signature).
                # Fall back to using the raw string as-is with a warning.
                logger.warning(
                    f"Raw extraction failed for '{expected_function}', "
                    "using full input as-is"
                )
                return ParsedCode(function_name=expected_function, code=code)
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

        # The registry's line numbers go stale whenever the base solver changes.
        # A blind splice at stale lines silently corrupts an unrelated region,
        # so verify the target actually starts in the recorded range and
        # relocate it if not.
        if not self._range_contains_definition(lines, start_idx, end_idx, func_name):
            relocated = self._relocate_function(lines, func_name)
            if relocated is None:
                raise ValueError(
                    f"Registry lines {info.start_line}-{info.end_line} for {func_name} "
                    f"do not contain its definition and it could not be relocated in "
                    f"{file_path}; re-run scripts/configure_target.py to re-index"
                )
            start_idx, end_idx = relocated
            logger.warning(
                f"Registry lines for {func_name} were stale; relocated definition to "
                f"lines {start_idx + 1}-{end_idx} in {file_path}"
            )

        # Ensure new_code ends with newline
        if not new_code.endswith("\n"):
            new_code += "\n"

        # Replace the function. Write via temp+rename: the target may be a
        # hardlink into the base solver tree, and an in-place write would
        # truncate the shared inode.
        new_lines = lines[:start_idx] + [new_code] + lines[end_idx:]

        tmp_path = file_path.with_name(file_path.name + ".inject.tmp")
        tmp_path.write_text("".join(new_lines))
        tmp_path.replace(file_path)

        logger.info(
            f"Replaced {func_name} in {file_path} "
            f"(was lines {info.start_line}-{info.end_line})"
        )

    def _range_contains_definition(
        self, lines: list[str], start_idx: int, end_idx: int, func_name: str
    ) -> bool:
        """True if func_name's definition plausibly starts within [start_idx, end_idx)."""
        window = "".join(lines[start_idx:min(end_idx, start_idx + 5)])
        return re.search(rf"\b{re.escape(func_name)}\s*\(", window) is not None

    @staticmethod
    def _blank_comments_and_strings(text: str) -> str:
        """Replace comment/string/char-literal contents with spaces, preserving
        line structure, so brace counting cannot be fooled by braces in them."""
        out = []
        i, n = 0, len(text)
        mode = None  # None | "line" | "block" | "str" | "chr"
        while i < n:
            c = text[i]
            nxt = text[i + 1] if i + 1 < n else ""
            if mode is None:
                if c == "/" and nxt == "/":
                    mode = "line"
                    out.append("  ")
                    i += 2
                    continue
                if c == "/" and nxt == "*":
                    mode = "block"
                    out.append("  ")
                    i += 2
                    continue
                if c == '"':
                    mode = "str"
                    out.append(" ")
                    i += 1
                    continue
                if c == "'":
                    mode = "chr"
                    out.append(" ")
                    i += 1
                    continue
                out.append(c)
            else:
                if c == "\n":
                    out.append("\n")
                    if mode == "line":
                        mode = None
                    i += 1
                    continue
                if mode == "block" and c == "*" and nxt == "/":
                    mode = None
                    out.append("  ")
                    i += 2
                    continue
                if mode in ("str", "chr") and c == "\\":
                    out.append("  ")
                    i += 2
                    continue
                if (mode == "str" and c == '"') or (mode == "chr" and c == "'"):
                    mode = None
                out.append(" ")
            i += 1
        return "".join(out)

    def _relocate_function(
        self, lines: list[str], func_name: str
    ) -> Optional[tuple[int, int]]:
        """Scan the file for func_name's definition; return (start_idx, end_idx)
        as a 0-based/exclusive line range, or None if not found unambiguously."""
        blanked = self._blank_comments_and_strings("".join(lines)).splitlines(keepends=True)
        name_re = re.compile(rf"\b{re.escape(func_name)}\s*\(")
        type_line_re = re.compile(r"^(?:static\s+|inline\s+)*[A-Za-z_][\w\s\*]*\**\s*$")

        candidates = []
        for i, line in enumerate(blanked):
            if not name_re.search(line):
                continue
            # Definitions in kissat start at column 0 (name-first for multiline
            # signatures, or type-first single-line); calls are indented.
            if not re.match(rf"[A-Za-z_].*\b{re.escape(func_name)}\s*\(", line):
                continue
            # Find the opening brace; a ';' before it means this is a prototype.
            offset = name_re.search(line).start()
            brace_line = None
            for j in range(i, min(i + 10, len(blanked))):
                text_j = blanked[j]
                from_pos = offset if j == i else 0
                semi = text_j.find(";", from_pos)
                brace = text_j.find("{", from_pos)
                if brace != -1 and (semi == -1 or brace < semi):
                    brace_line = j
                    break
                if semi != -1:
                    break
            if brace_line is None:
                continue
            # Include a bare return-type line directly above (multiline signature).
            start = i
            if i > 0 and type_line_re.match(blanked[i - 1].rstrip("\n")):
                start = i - 1
            # Brace-match from the opening brace to find the end of the body.
            depth = 0
            seen_open = False
            end = None
            for k in range(brace_line, len(blanked)):
                for ch in blanked[k]:
                    if ch == "{":
                        depth += 1
                        seen_open = True
                    elif ch == "}":
                        depth -= 1
                if seen_open and depth == 0:
                    end = k + 1
                    break
            if end is not None:
                candidates.append((start, end))

        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            logger.warning(
                f"Found {len(candidates)} definition candidates for {func_name}; refusing to guess"
            )
        return None

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
        code = re.sub(r"(?m)^\s*```[a-zA-Z0-9_-]*\s*$", "", code)
        header_pattern = (
            rf"^[ \t]*(?:static\s+)?(?:inline\s+)?"
            rf"[A-Za-z_][\w\s\*]*\b{re.escape(func_name)}\s*\([^)]*\)\s*\{{"
        )

        match = re.search(header_pattern, code, re.MULTILINE | re.DOTALL)
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
