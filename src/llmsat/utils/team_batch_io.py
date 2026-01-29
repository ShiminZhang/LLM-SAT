from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llmsat.llmsat import get_id


@dataclass(frozen=True)
class TeamBatchMap:
    leader_batch_id: Optional[str]
    member_batch_map: Dict[str, str]  # member_batch_id -> leader_id
    code_batch_map: Dict[str, str]  # code_batch_id -> algorithm_id


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _strategy_text(spec: dict) -> str:
    parts = []
    name = spec.get("name")
    alg = spec.get("algorithm")
    reason = spec.get("reason")
    if isinstance(name, str) and name.strip():
        parts.append(name.strip())
    if isinstance(alg, str) and alg.strip():
        parts.append(alg.strip())
    if isinstance(reason, str) and reason.strip():
        parts.append(reason.strip())
    return "\n\n".join(parts).strip()


def find_latest_team_batch_map(batch_dir: Path) -> Path:
    """Find the most recent team batch map JSON in a batch output directory."""
    if not batch_dir.exists():
        raise FileNotFoundError(batch_dir)

    candidates = list(batch_dir.glob("team_batch_id_map_*.json"))
    if not candidates:
        # Older/other naming patterns
        candidates = list(batch_dir.glob("batch_id_map_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No team batch map found in {batch_dir} (expected team_batch_id_map_*.json)"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_team_batch_map(batch_dir: Path) -> TeamBatchMap:
    path = find_latest_team_batch_map(batch_dir)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid batch map JSON: {path}")

    member_batch_map = obj.get("member_batch_map") or {}
    code_batch_map = obj.get("code_batch_map") or {}

    if not isinstance(member_batch_map, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in member_batch_map.items()
    ):
        raise ValueError(f"Invalid member_batch_map in {path}")

    if not isinstance(code_batch_map, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in code_batch_map.items()
    ):
        # code_batch_map may not exist yet if code gen not run
        code_batch_map = {}

    leader_batch_id = obj.get("leader_batch_id") or obj.get("algorithm_batch_id")
    if leader_batch_id is not None and not isinstance(leader_batch_id, str):
        leader_batch_id = None

    return TeamBatchMap(
        leader_batch_id=leader_batch_id,
        member_batch_map=member_batch_map,
        code_batch_map=code_batch_map,
    )


def load_team_strategies_from_batch_dir(batch_dir: Path) -> List[dict]:
    """Load leaders+members into the same schema as generate_diverse_batch.py.

    Expects files in batch_dir:
    - leaders_output.txt
    - member_output_<batch_id>.txt (for each member batch)

    Returns records:
    - leader:  type="leader", leader_id=id
    - member:  type="member", leader_id=<leader id>
    """
    from llmsat.pipelines.chatgpt_data_generation import parse_algorithm_response

    batch_map = load_team_batch_map(batch_dir)

    leaders_path = batch_dir / "leaders_output.txt"
    if not leaders_path.exists():
        raise FileNotFoundError(f"Missing leaders_output.txt in {batch_dir}")

    leader_rows = _read_jsonl(leaders_path)
    leaders: List[dict] = []
    leader_target_functions: Dict[str, str] = {}

    for raw in leader_rows:
        algorithm_str, target_function = parse_algorithm_response(raw)
        leader_id = get_id(algorithm_str)
        try:
            spec = json.loads(algorithm_str)
        except Exception:
            spec = {"name": "", "algorithm": algorithm_str}

        tf = target_function or spec.get("target_function") or "kissat_restarting"
        leader_target_functions[leader_id] = str(tf)

        leaders.append(
            {
                "id": leader_id,
                "type": "leader",
                "leader_id": leader_id,
                "target_function": str(tf),
                "spec": spec,
                "strategy_text": _strategy_text(spec),
                "meta": {
                    "source": "team_batch",
                    "batch_dir": str(batch_dir),
                },
                "raw": raw,
            }
        )

    members: List[dict] = []
    for member_batch_id, leader_id in batch_map.member_batch_map.items():
        path = batch_dir / f"member_output_{member_batch_id}.txt"
        if not path.exists():
            # tolerate missing outputs if a batch was not downloaded
            continue
        for raw in _read_jsonl(path):
            algorithm_str, _tf = parse_algorithm_response(raw)
            member_id = get_id(algorithm_str)
            try:
                spec = json.loads(algorithm_str)
            except Exception:
                spec = {"name": "", "algorithm": algorithm_str}

            tf = leader_target_functions.get(leader_id, "kissat_restarting")

            members.append(
                {
                    "id": member_id,
                    "type": "member",
                    "leader_id": leader_id,
                    "target_function": str(tf),
                    "spec": spec,
                    "strategy_text": _strategy_text(spec),
                    "meta": {
                        "source": "team_batch",
                        "batch_dir": str(batch_dir),
                        "member_batch_id": member_batch_id,
                    },
                    "raw": raw,
                }
            )

    return leaders + members


def load_team_codes_from_batch_dir(batch_dir: Path) -> Dict[str, List[dict]]:
    """Load generated code outputs keyed by algorithm_id.

    Returns mapping algorithm_id -> list of {code_id, code, raw_response}.
    """
    from llmsat.pipelines.chatgpt_data_generation import parse_code_response

    batch_map = load_team_batch_map(batch_dir)
    if not batch_map.code_batch_map:
        return {}

    out: Dict[str, List[dict]] = {}

    for code_batch_id, algorithm_id in batch_map.code_batch_map.items():
        path = batch_dir / f"code_output_{code_batch_id}.txt"
        if not path.exists():
            # tolerate missing output file
            continue
        for raw in _read_jsonl(path):
            code_str = parse_code_response(raw)
            code_id = get_id(code_str)
            out.setdefault(algorithm_id, []).append(
                {"code_id": code_id, "code": code_str, "raw": raw, "code_batch_id": code_batch_id}
            )

    return out
