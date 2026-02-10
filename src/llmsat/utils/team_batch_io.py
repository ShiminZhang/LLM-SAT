"""Utilities for loading strategy data from batch directories."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List


def _extract_json_from_text(text: str) -> dict | None:
    """Extract JSON object from text that may contain markdown code blocks."""
    # Try to find JSON in code blocks first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try parsing the whole text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find a bare JSON object
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _parse_gemini_response(line: str, id_prefix: str = "") -> dict | None:
    """Parse a Gemini API response line and extract the algorithm."""
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None

    # Navigate Gemini response structure
    try:
        candidates = data.get("response", {}).get("candidates", [])
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts", [])
        for part in parts:
            if "text" in part:
                alg = _extract_json_from_text(part["text"])
                if alg and "algorithm" in alg:
                    base_id = data.get("key", "unknown")
                    unique_id = f"{id_prefix}{base_id}" if id_prefix else base_id
                    return {
                        "id": unique_id,
                        "type": "leader",
                        "leader_id": None,
                        "target_function": alg.get("target_function", ""),
                        "spec": {"name": alg.get("name", "")},
                        "meta": {"model": data.get("model", "")},
                        "strategy_text": alg.get("algorithm", ""),
                    }
    except (KeyError, IndexError, TypeError):
        pass

    return None


def load_team_strategies_from_batch_dir(batch_dir: Path) -> List[dict]:
    """Load strategies from a batch directory.

    Supports:
    - Gemini API response format (leaders_output.txt with JSON lines)
    - Member output batches
    """
    strategies: List[dict] = []
    batch_dir = Path(batch_dir)

    # Load leaders
    leaders_file = batch_dir / "leaders_output.txt"
    if leaders_file.exists():
        with leaders_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                strat = _parse_gemini_response(line, id_prefix="leader-")
                if strat:
                    strat["type"] = "leader"
                    strategies.append(strat)

    # Load members from member_output_batches
    member_batches_dir = batch_dir / "member_output_batches"
    if member_batches_dir.exists():
        for member_file in sorted(member_batches_dir.glob("*.txt")):
            # Extract leader_id from filename pattern
            leader_id = member_file.stem.replace("member_output_batch_", "")
            with member_file.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    strat = _parse_gemini_response(line, id_prefix=f"member-{leader_id[:8]}-")
                    if strat:
                        strat["type"] = "member"
                        strat["leader_id"] = leader_id
                        strategies.append(strat)

    return strategies
