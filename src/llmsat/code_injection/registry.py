"""
Function registry for tracking modifiable functions in the Kissat solver.

The registry maps function names to their source file locations and line ranges,
enabling extraction and replacement of specific functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class FunctionInfo:
    """Information about a registered function."""

    file: str
    """Relative path to the source file (e.g., 'src/restart.c')."""

    start_line: int
    """1-indexed line number where the function definition starts."""

    end_line: int
    """1-indexed line number where the function definition ends (inclusive)."""

    signature: str
    """Function signature (e.g., 'bool kissat_restarting(kissat *solver)')."""


class FunctionRegistry:
    """
    Registry of modifiable functions in the Kissat solver.

    Loads function location data from a YAML file and provides lookup by name.

    Example:
        registry = FunctionRegistry("solvers/base/function_registry.yaml")
        info = registry.get("kissat_restarting")
        if info:
            print(f"Function is in {info.file}, lines {info.start_line}-{info.end_line}")
    """

    def __init__(self, registry_path: str | Path):
        """
        Load the function registry from a YAML file.

        Args:
            registry_path: Path to the function_registry.yaml file.

        Raises:
            FileNotFoundError: If the registry file doesn't exist.
            ValueError: If the registry file is malformed.
        """
        self._path = Path(registry_path)

        if not self._path.exists():
            raise FileNotFoundError(f"Function registry not found: {self._path}")

        with open(self._path) as f:
            data = yaml.safe_load(f)

        if not data or "functions" not in data:
            raise ValueError(f"Malformed registry file: {self._path}")

        self.version: str = data.get("version", "unknown")
        self.generated: str = data.get("generated", "unknown")
        self._functions: dict[str, FunctionInfo] = {}

        for name, info in data["functions"].items():
            try:
                self._functions[name] = FunctionInfo(
                    file=info["file"],
                    start_line=info["start_line"],
                    end_line=info["end_line"],
                    signature=info["signature"],
                )
            except KeyError as e:
                raise ValueError(
                    f"Missing required field {e} for function '{name}' in {self._path}"
                )

    def get(self, func_name: str) -> Optional[FunctionInfo]:
        """
        Get information about a registered function.

        Args:
            func_name: Name of the function to look up.

        Returns:
            FunctionInfo if found, None otherwise.
        """
        return self._functions.get(func_name)

    def __getitem__(self, func_name: str) -> FunctionInfo:
        """
        Get information about a registered function (raises if not found).

        Args:
            func_name: Name of the function to look up.

        Returns:
            FunctionInfo for the function.

        Raises:
            KeyError: If the function is not in the registry.
        """
        if func_name not in self._functions:
            available = ", ".join(sorted(self._functions.keys()))
            raise KeyError(
                f"Function '{func_name}' not in registry. "
                f"Available functions: {available}"
            )
        return self._functions[func_name]

    def __contains__(self, func_name: str) -> bool:
        """Check if a function is in the registry."""
        return func_name in self._functions

    def list_functions(self) -> list[str]:
        """Get a list of all registered function names."""
        return list(self._functions.keys())

    def __len__(self) -> int:
        """Get the number of registered functions."""
        return len(self._functions)

    def __repr__(self) -> str:
        return (
            f"FunctionRegistry(path={self._path!r}, "
            f"version={self.version!r}, "
            f"functions={len(self._functions)})"
        )