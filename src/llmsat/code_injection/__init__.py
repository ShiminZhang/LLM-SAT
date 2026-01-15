"""
Code injection module for replacing functions in the Kissat solver.

This module provides tools to:
- Load a function registry mapping function names to file locations
- Extract function source code for use in coder prompts
- Parse LLM-generated code output
- Replace functions in solver copies with generated code
"""

from .registry import FunctionRegistry, FunctionInfo
from .injector import FunctionInjector

__all__ = ["FunctionRegistry", "FunctionInfo", "FunctionInjector"]