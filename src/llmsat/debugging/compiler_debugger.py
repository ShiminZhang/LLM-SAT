from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from llmsat.llmsat import get_logger
from llmsat.utils.chatgpt_helper import get_response_from_chatgpt
from llmsat.code_injection import FunctionInjector, FunctionRegistry

logger = get_logger(__name__)

DEBUGGING_PROMPT_TEMPLATE = """You are debugging a function from a SAT solver written in C that failed to compile. You will be provided with the compilatio error, the target function that caused the errors, and the entire file content that the function lives in.

## Compiler Error
```
{compiler_stderr}
```

## Target Function to Fix
The following function code caused the compilation error:
```c
{failing_code}
```

## Current File Content
Here is the current source file after your code was injected. Only use types, macros, and functions that exist in this file or its includes:
```c
{current_file_content}
```

## Instructions
1. Analyze the compiler error carefully
3. Fix the target function code while still following the same overall logic
4. Keep the same function signature: {function_signature}
5. Return the fixed function code wrapped in <code function="{function_name}">...</code> tags

Provide the corrected function:
"""


@dataclass
class CompilerDebugger:
    """
    LLM-powered code debugger that generates fix suggestions for compilation errors.
    
    This class is responsible ONLY for generating fixed code suggestions.
    It does NOT handle compilation - that remains in the caller's responsibility.
    
    Example:
        debugger = CompilerDebugger(model="gpt-5.2")
        fixed_code = debugger.suggest_fix(
            failing_code="bool kissat_restarting(...) { ... }",
            compiler_stderr="error: unknown type 'restart_window_stats'",
            original_file_content="...",
            function_name="kissat_restarting"
        )
    """
    
    model: str = "gpt-5.2"
    temperature: float = 0.3
    registry_path: str = "solvers/base/function_registry.yaml"
    _registry: Optional[FunctionRegistry] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Initialize the function registry for parsing LLM output."""
        if self._registry is None:
            try:
                self._registry = FunctionRegistry(self.registry_path)
            except FileNotFoundError:
                logger.warning(f"Registry not found at {self.registry_path}, code parsing may be limited")
                self._registry = None
    
    def suggest_fix(
        self,
        failing_code: str,
        compiler_stderr: str,
        current_file_content: str,
        function_name: str,
        function_signature: Optional[str] = None,
    ) -> Optional[str]:
        """
        Ask LLM to fix the code based on compiler errors.
        
        Args:
            failing_code: The C code that failed to compile
            compiler_stderr: Compiler error output from make
            current_file_content: Full content of the current source file (after injection)
            function_name: Name of the function being fixed
            function_signature: Optional function signature for context
            
        Returns:
            Fixed code string if LLM provides valid response, None otherwise.
        """
        if not function_signature:
            function_signature = f"bool {function_name}(kissat *solver)"
        
        prompt = DEBUGGING_PROMPT_TEMPLATE.format(
            compiler_stderr=compiler_stderr,
            failing_code=failing_code,
            current_file_content=current_file_content,
            function_name=function_name,
            function_signature=function_signature,
        )
        
        logger.info(f"Requesting fix from LLM for {function_name} (model: {self.model})")
        logger.debug(f"Compiler stderr length: {len(compiler_stderr)} chars")
        
        try:
            response = get_response_from_chatgpt(
                prompt=prompt,
                model=self.model,
                temperature=self.temperature,
            )
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return None
        
        fixed_code = self._extract_code_from_response(response, function_name)
        
        if fixed_code:
            logger.info(f"LLM suggested fix for {function_name}")
        else:
            logger.warning(f"Failed to extract valid code from LLM response")
            
        return fixed_code
    
    def _extract_code_from_response(
        self, 
        response: str, 
        function_name: str
    ) -> Optional[str]:
        """
        Extract fixed code from LLM response.
        
        Attempts to parse <code>...</code> tags, falling back to raw extraction.
        """
        if not response:
            return None
            
        if self._registry:
            try:
                from llmsat.code_injection import FunctionInjector
                injector = FunctionInjector(self._registry, "solvers/base")
                parsed = injector.parse_llm_output(response, expected_function=function_name)
                return parsed.code
            except (ValueError, Exception) as e:
                logger.debug(f"Injector parsing failed: {e}, trying manual extraction")
        
        return self._manual_extract_code(response, function_name)
    
    def _manual_extract_code(self, text: str, function_name: str) -> Optional[str]:
        """
        Manually extract code from <code>...</code> tags.
        """
        import re
        
        pattern = rf'<code(?:\s+function=["\']?{re.escape(function_name)}["\']?)?\s*>(.*?)</code>'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            code = match.group(1).strip()
            code = re.sub(r'^```[a-z]*\n?', '', code)
            code = re.sub(r'\n?```$', '', code)
            return code.strip()
        
        header_pattern = rf'(?:static\s+)?(?:inline\s+)?(?:bool|void|int)\s+{re.escape(function_name)}\s*\([^)]*\)\s*\{{'
        match = re.search(header_pattern, text, re.DOTALL)
        
        if match:
            start = match.start()
            brace_count = 0
            i = match.end() - 1
            while i < len(text):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return text[start:i+1].strip()
                i += 1
        
        return None
