import argparse
import fcntl
import os
import json
import shutil
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote

try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.pool
except ImportError:  # Local-cache mode does not require psycopg2.
    psycopg2 = None
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()
from typing import List, Dict, Any, Mapping, Optional
from llmsat.llmsat import CodeResult, CodeStatus, AlgorithmResult, AlgorithmStatus, Role
from datetime import datetime
from llmsat.llmsat import get_logger, setup_logging
logger = get_logger(__name__)

# Storage backend switch. Local cache is the default; set
# LLMSAT_USE_LOCAL_CACHE=0 only when intentionally using the legacy RDS backend.
USE_LOCAL_CACHE = os.environ.get("LLMSAT_USE_LOCAL_CACHE", "1").lower() not in {
    "0", "false", "no",
}
LOCAL_CACHE_ROOT = Path(
    os.environ.get("LLMSAT_LOCAL_CACHE_ROOT", "data/cache/local_results")
)
_LOCAL_THREAD_LOCKS: Dict[str, threading.RLock] = {}
_LOCAL_THREAD_LOCKS_GUARD = threading.Lock()


def _local_dir(name: str) -> Path:
    path = LOCAL_CACHE_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _local_record_path(kind: str, record_id: str) -> Path:
    return _local_dir(kind) / f"{quote(str(record_id), safe='')}.json"


def _thread_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _LOCAL_THREAD_LOCKS_GUARD:
        return _LOCAL_THREAD_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _locked_local_path(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with _thread_lock(path), lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _write_json_unlocked(path: Path, value: Any) -> None:
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(value, handle, indent=2, ensure_ascii=False, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    with _locked_local_path(path):
        _write_json_unlocked(path, value)


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _algorithm_to_dict(result: AlgorithmResult) -> Dict[str, Any]:
    data = asdict(result)
    data["role"] = result.role.value if isinstance(result.role, Role) else result.role
    return data


def _algorithm_from_dict(data: Mapping[str, Any]) -> AlgorithmResult:
    values = dict(data)
    role = values.get("role", Role.LEADER.value)
    values["role"] = role if isinstance(role, Role) else Role(role)
    return AlgorithmResult(**values)


def _code_to_dict(result: CodeResult) -> Dict[str, Any]:
    return asdict(result)


def _code_from_dict(data: Mapping[str, Any]) -> CodeResult:
    return CodeResult(**dict(data))


def _iter_local_records(kind: str):
    for path in sorted(_local_dir(kind).glob("*.json")):
        yield _read_json(path)


def _router_path(name: str) -> Path:
    return _local_dir("router_tables") / f"{quote(str(name), safe='')}.json"

# DB column layout for algorithm_results (after migration):
#   id, algorithm (now stores description), code_id_list, status, last_updated,
#   prompt, par2 (now stores raw_par2_score as JSON list), error_rate (legacy),
#   other_metrics, role, function_name, parent_ids, parent_descriptions,
#   normalized_par2_scores, analysis

# --- Connection Pool ---
_pool = None

def _get_pool():
    global _pool
    if USE_LOCAL_CACHE:
        raise RuntimeError("Raw database connections are unavailable in local-cache mode")
    if psycopg2 is None:
        raise RuntimeError("psycopg2 is required when LLMSAT_USE_LOCAL_CACHE=0")
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            host="llmsat.crac0kykqrxp.us-east-2.rds.amazonaws.com",
            database="postgres",
            user="Shimin",
            password=os.environ["DB_PASS"],
            port=5432,
            sslmode="require",
        )
    return _pool

def connect_to_db():
    if USE_LOCAL_CACHE:
        raise RuntimeError("Raw database connections are unavailable in local-cache mode")
    return _get_pool().getconn()

def release_conn(conn):
    if USE_LOCAL_CACHE:
        return
    try:
        _get_pool().putconn(conn)
    except Exception:
        pass  # pool may not be initialized in edge cases


def get_code_result_of_status(status: CodeStatus) -> List[CodeResult]:
    logger.info(f"Getting code results of status {status}")
    if USE_LOCAL_CACHE:
        return [
            _code_from_dict(row)
            for row in _iter_local_records("code_results")
            if row.get("status") == status
        ]
    conn = connect_to_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM code_results WHERE status = %s;", (status,))
        rows = cur.fetchall()
        code_results = []
        for row in rows:
            if row is None:
                logger.warning(f"Row is None for status {status}")
                continue
            if len(row) != 7:
                logger.warning(f"Row has {len(row)} columns")
                logger.warning(f"Row: {row}")
                continue
            try:
                code_result = ToCodeResult(row)
            except Exception as e:
                logger.warning(f"Error converting row to CodeResult: {e}")
                logger.warning(f"Row: {row}")
                continue
            code_results.append(code_result)
        return code_results
    finally:
        release_conn(conn)

def get_algorithm_result_of_status(status: AlgorithmStatus) -> List[AlgorithmResult]:
    if USE_LOCAL_CACHE:
        return [
            _algorithm_from_dict(row)
            for row in _iter_local_records("algorithm_results")
            if row.get("status") == status
        ]
    conn = connect_to_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM algorithm_results WHERE status = %s;", (status,))
        rows = cur.fetchall()
        return [_row_to_algorithm_result(row) for row in rows]
    finally:
        release_conn(conn)

def get_all_tasks():
    if USE_LOCAL_CACHE:
        return _read_json(LOCAL_CACHE_ROOT / "tasks.json", [])
    conn = connect_to_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM tasks;")
        return cur.fetchall()
    finally:
        release_conn(conn)

def remove_code_result(code_result_id: str):
    if USE_LOCAL_CACHE:
        _local_record_path("code_results", code_result_id).unlink(missing_ok=True)
        return
    conn = connect_to_db()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM code_results WHERE id = %s;", (code_result_id,))
        conn.commit()
    finally:
        release_conn(conn)

def remove_algorithm_result(algorithm_result_id: str):
    if USE_LOCAL_CACHE:
        _local_record_path("algorithm_results", algorithm_result_id).unlink(missing_ok=True)
        return
    conn = connect_to_db()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM algorithm_results WHERE id = %s;", (algorithm_result_id,))
        conn.commit()
    finally:
        release_conn(conn)

def get_code_result(code_result_id: str) -> Optional[CodeResult]:
    if USE_LOCAL_CACHE:
        row = _read_json(_local_record_path("code_results", code_result_id))
        return _code_from_dict(row) if row is not None else None
    conn = connect_to_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM code_results WHERE id = %s;", (code_result_id,))
        assert cur.rowcount <= 1, "hash collision"
        if cur.rowcount == 1:
            result = ToCodeResult(cur.fetchone())
            return result
        else:
            return None
    finally:
        release_conn(conn)

def get_algorithms_by_prompt(prompt: str) -> List[AlgorithmResult]:
    if USE_LOCAL_CACHE:
        return [
            _algorithm_from_dict(row)
            for row in _iter_local_records("algorithm_results")
            if row.get("prompt") == prompt
        ]
    conn = connect_to_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM algorithm_results WHERE prompt = %s;", (prompt,))
        rows = cur.fetchall()
        return [_row_to_algorithm_result(row) for row in rows]
    finally:
        release_conn(conn)

def get_algorithm_result(algorithm_result_id: str) -> Optional[AlgorithmResult]:
    if USE_LOCAL_CACHE:
        row = _read_json(_local_record_path("algorithm_results", algorithm_result_id))
        return _algorithm_from_dict(row) if row is not None else None
    conn = connect_to_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM algorithm_results WHERE id = %s;", (algorithm_result_id,))
        rows = cur.fetchall()
        assert len(rows) <= 1, "hash collision"
        if len(rows) == 1:
            return _row_to_algorithm_result(rows[0])
        else:
            return None
    finally:
        release_conn(conn)

def update_code_result(code_result: CodeResult):
    if USE_LOCAL_CACHE:
        code_result.last_updated = datetime.now().isoformat()
        _write_json(
            _local_record_path("code_results", code_result.id),
            _code_to_dict(code_result),
        )
        logger.info(f"Updated code result {code_result.id}")
        return
    existing_code_result = get_code_result(code_result.id)
    conn = connect_to_db()
    try:
        cur = conn.cursor()
        build_success_text = None if code_result.build_success is None else str(code_result.build_success)
        logger.info(f"Updating code result {code_result.id}")
        code_result.last_updated = datetime.now()
        if existing_code_result is None: # add the code result
            cur.execute(
                "INSERT INTO code_results (id, code, algorithm, status, last_updated, build_success, par2) VALUES (%s, %s, %s, %s, %s, %s, %s);",
                (code_result.id, code_result.code, code_result.algorithm_id, code_result.status, code_result.last_updated, build_success_text, code_result.par2),
            )
        else: # update the code result
            cur.execute(
                "UPDATE code_results SET code = %s, algorithm = %s, status = %s, last_updated = %s, build_success = %s, par2 = %s WHERE id = %s;",
                (code_result.code, code_result.algorithm_id, code_result.status, code_result.last_updated, build_success_text, code_result.par2, code_result.id),
            )
        conn.commit()
        logger.info(f"Updated code result {code_result.id}")
    finally:
        release_conn(conn)

def update_algorithm_result(algorithm_result: AlgorithmResult):
    if USE_LOCAL_CACHE:
        algorithm_result.last_updated = datetime.now().isoformat()
        _write_json(
            _local_record_path("algorithm_results", algorithm_result.id),
            _algorithm_to_dict(algorithm_result),
        )
        logger.info(f"Updated algorithm result {algorithm_result.id}")
        return
    existing_algorithm_result = get_algorithm_result(algorithm_result.id)
    conn = connect_to_db()
    try:
        cur = conn.cursor()

        # Serialize other_metrics (no longer packing target_function/parent_id here)
        other_metrics_obj = algorithm_result.other_metrics or {}
        if isinstance(other_metrics_obj, dict):
            # Clean out legacy keys that now have their own columns
            other_metrics_obj.pop("target_function", None)
            other_metrics_obj.pop("parent_id", None)
        other_metrics_text = _serialize_json_field(other_metrics_obj)

        # Serialize new list/enum fields
        role_text = algorithm_result.role.value if isinstance(algorithm_result.role, Role) else str(algorithm_result.role)
        parent_ids_text = _serialize_json_field(algorithm_result.parent_id)
        parent_descs_text = _serialize_json_field(algorithm_result.parent_algorithm_description)
        raw_par2_text = _serialize_json_field(algorithm_result.raw_par2_score)
        norm_par2_text = _serialize_json_field(algorithm_result.normalized_par2_score)

        algorithm_result.last_updated = datetime.now()

        if existing_algorithm_result is None:
            cur.execute(
                """INSERT INTO algorithm_results
                   (id, algorithm, code_id_list, status, last_updated, prompt, par2,
                    error_rate, other_metrics, role, function_name, parent_ids,
                    parent_descriptions, normalized_par2_scores, analysis)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);""",
                (
                    algorithm_result.id,
                    algorithm_result.description,
                    _code_id_list_to_text(algorithm_result.code_id_list),
                    algorithm_result.status,
                    algorithm_result.last_updated,
                    algorithm_result.prompt,
                    raw_par2_text,
                    None,  # error_rate (legacy, always None for new records)
                    other_metrics_text,
                    role_text,
                    algorithm_result.function_name,
                    parent_ids_text,
                    parent_descs_text,
                    norm_par2_text,
                    algorithm_result.analysis,
                ),
            )
        else:
            cur.execute(
                """UPDATE algorithm_results
                   SET algorithm = %s, code_id_list = %s, status = %s, last_updated = %s,
                       prompt = %s, par2 = %s, error_rate = %s, other_metrics = %s,
                       role = %s, function_name = %s, parent_ids = %s,
                       parent_descriptions = %s, normalized_par2_scores = %s, analysis = %s
                   WHERE id = %s;""",
                (
                    algorithm_result.description,
                    _code_id_list_to_text(algorithm_result.code_id_list),
                    algorithm_result.status,
                    algorithm_result.last_updated,
                    algorithm_result.prompt,
                    raw_par2_text,
                    None,
                    other_metrics_text,
                    role_text,
                    algorithm_result.function_name,
                    parent_ids_text,
                    parent_descs_text,
                    norm_par2_text,
                    algorithm_result.analysis,
                    algorithm_result.id,
                ),
            )
        conn.commit()
        print(f"Updated algorithm result {algorithm_result.id}")
    finally:
        release_conn(conn)

def append_code_id(algorithm_id: str, code_id: str):
    """Append a code ID, avoiding duplicates in the stored list."""
    if USE_LOCAL_CACHE:
        path = _local_record_path("algorithm_results", algorithm_id)
        with _locked_local_path(path):
            row = _read_json(path)
            if row is None:
                logger.warning(f"append_code_id: algorithm {algorithm_id} not found")
                return
            code_ids = row.setdefault("code_id_list", [])
            if code_id not in code_ids:
                code_ids.append(code_id)
                row["last_updated"] = datetime.now().isoformat()
                _write_json_unlocked(path, row)
        return
    conn = connect_to_db()
    try:
        with conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT code_id_list FROM algorithm_results WHERE id = %s FOR UPDATE",
                (algorithm_id,),
            )
            row = cur.fetchone()
            if row is None:
                logger.warning(f"append_code_id: algorithm {algorithm_id} not found")
                return
            current = _text_to_code_id_list(row[0])
            if code_id not in current:
                current.append(code_id)
            cur.execute(
                "UPDATE algorithm_results SET code_id_list = %s WHERE id = %s",
                (_code_id_list_to_text(current), algorithm_id),
            )
    finally:
        release_conn(conn)

def init_tables():
    if USE_LOCAL_CACHE:
        _local_dir("algorithm_results")
        _local_dir("code_results")
        _local_dir("router_tables")
        return
    conn = connect_to_db()
    try:
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS algorithm_results (
            id TEXT PRIMARY KEY,
            algorithm TEXT,
            code_id_list TEXT,
            status TEXT,
            last_updated TEXT,
            prompt TEXT,
            par2 TEXT,
            error_rate TEXT,
            other_metrics TEXT,
            role TEXT,
            function_name TEXT,
            parent_ids TEXT,
            parent_descriptions TEXT,
            normalized_par2_scores TEXT,
            analysis TEXT
        );""")
        cur.execute("CREATE TABLE IF NOT EXISTS code_results (id TEXT PRIMARY KEY, code TEXT, algorithm TEXT, status TEXT, last_updated TEXT, solver_id TEXT, build_success TEXT, par2 TEXT);")
        conn.commit()
    finally:
        release_conn(conn)

def clear_tables():
    if USE_LOCAL_CACHE:
        for kind in ("algorithm_results", "code_results"):
            for path in _local_dir(kind).glob("*.json"):
                path.unlink()
        return
    conn = connect_to_db()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM algorithm_results;")
        cur.execute("DELETE FROM code_results;")
        conn.commit()
    finally:
        release_conn(conn)

def ToAlgorithmResult(result) -> AlgorithmResult:
    """Convert a DB row (tuple or dict) to AlgorithmResult with backwards compat."""
    if not isinstance(result, tuple):
        return _row_to_algorithm_result(result)
    # Tuple-based: convert to dict using known column order, then delegate.
    # This handles both old schema (9 cols) and new schema (15 cols).
    keys = ["id", "algorithm", "code_id_list", "status", "last_updated",
            "prompt", "par2", "error_rate", "other_metrics",
            "role", "function_name", "parent_ids", "parent_descriptions",
            "normalized_par2_scores", "analysis"]
    row = {}
    for i, key in enumerate(keys):
        if i < len(result):
            row[key] = result[i]
    return _row_to_algorithm_result(row)

def ToCodeResult(result: tuple) -> CodeResult:
    if type(result) != tuple:
        return _row_to_code_result(result)
    else:
        return CodeResult(
            id=result[0],
            algorithm_id=result[2],
            code=result[1],
            status=result[3],
            par2=_to_float(result[7]),
            last_updated=result[4],
            build_success=_to_bool(result[6]))

def get_all_algorithm_results() -> List[AlgorithmResult]:
    if USE_LOCAL_CACHE:
        return [
            _algorithm_from_dict(row)
            for row in _iter_local_records("algorithm_results")
        ]
    conn = connect_to_db()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM algorithm_results;")
        rows = cur.fetchall()
        algorithm_results = []
        for row in rows:
            algorithm_results.append(_row_to_algorithm_result(row))
        return algorithm_results
    finally:
        release_conn(conn)



def get_all_algorithm_ids():
    results = get_all_algorithm_results()
    return list(set([result.id for result in results]))

def delete_tables():
    if USE_LOCAL_CACHE:
        for kind in ("algorithm_results", "code_results"):
            shutil.rmtree(LOCAL_CACHE_ROOT / kind, ignore_errors=True)
        return
    conn = connect_to_db()
    try:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS algorithm_results;")
        cur.execute("DROP TABLE IF EXISTS code_results;")
        conn.commit()
    finally:
        release_conn(conn)

def test_utils():
    code_result = CodeResult(
        id="2",
        algorithm_id="1",
        code="return false;",
        status=CodeStatus.Generated,
        par2=None,
        last_updated=datetime.now(),
        build_success=None
    )
    update_code_result(code_result)
    algorithm_result = AlgorithmResult(
        id="1",
        function_name="kissat_restarting",
        description="Test Policy: Step 1: always restart",
        role=Role.LEADER,
        status=AlgorithmStatus.Generated,
        last_updated=datetime.now(),
        code_id_list=[],
        prompt="",
    )
    update_algorithm_result(algorithm_result)
    print(f"Tested utils")

# ------------------------
# Mapping helpers
# ------------------------

def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _to_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in ("true", "1", "t", "y", "yes"):
        return True
    if s in ("false", "0", "f", "n", "no"):
        return False
    return None

def _to_other_metrics(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value

def _code_id_list_to_text(ids: Optional[List[str]]) -> Optional[str]:
    if ids is None:
        return None
    try:
        return json.dumps(ids, ensure_ascii=False)
    except Exception:
        return ",".join(ids)

def _text_to_code_id_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    s = str(value)
    try:
        maybe = json.loads(s)
        if isinstance(maybe, list):
            return [str(x) for x in maybe]
    except Exception:
        pass
    if "," in s:
        return [part.strip() for part in s.split(",") if part.strip()]
    return [s] if s else []

def _serialize_json_field(value: Any) -> Optional[str]:
    """Serialize a value to JSON text for DB storage. Returns None for None."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)

def _to_json_list(value: Any) -> Optional[List[str]]:
    """Parse a JSON text column as a list of strings. Returns None if null/empty."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return None

def _to_json_list_float(value: Any) -> Optional[List[float]]:
    """Parse a JSON text column as a list of floats.

    Backwards compat: if value is a single float/string-number, wraps as
    [None, None, None, None, float] so the 'all' slot (index 4) is populated.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return [_to_float(x) for x in value]
    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, list):
            return [_to_float(x) for x in parsed]
    except Exception:
        pass
    # Legacy: single float value — put it in the 'all' slot (index 4)
    as_float = _to_float(value)
    if as_float is not None:
        return [None, None, None, None, as_float]
    return None

def _row_to_code_result(row: Mapping[str, Any]) -> CodeResult:
    return CodeResult(
        id=row.get("id"),
        algorithm_id=row.get("algorithm_id"),
        code=row.get("code"),
        status=row.get("status"),
        par2=_to_float(row.get("par2")),
        last_updated=row.get("last_updated"),
        build_success=_to_bool(row.get("build_success")),
    )
    
def _row_to_algorithm_result(row: Mapping[str, Any]) -> AlgorithmResult:
    """Convert a dict-like DB row to AlgorithmResult.

    Handles backwards compatibility: if new columns (role, function_name, etc.)
    are missing or NULL, falls back to extracting from other_metrics / inferring.
    """
    other_metrics = _to_other_metrics(row.get("other_metrics"))
    if other_metrics is None:
        other_metrics = {}

    # --- function_name: prefer new column, fall back to other_metrics ---
    function_name = row.get("function_name")
    if not function_name:
        function_name = "kissat_restarting"
        if isinstance(other_metrics, dict) and "target_function" in other_metrics:
            function_name = other_metrics.pop("target_function")

    # --- parent_id: prefer new parent_ids column (JSON list), fall back to other_metrics ---
    parent_id = _to_json_list(row.get("parent_ids"))
    if parent_id is None:
        # Legacy: other_metrics stored parent_id as a single string
        legacy_pid = None
        if isinstance(other_metrics, dict) and "parent_id" in other_metrics:
            legacy_pid = other_metrics.pop("parent_id")
        if legacy_pid is not None:
            parent_id = [legacy_pid] if isinstance(legacy_pid, str) else list(legacy_pid)
        # Also check for parent_a / parent_b (offspring records)
        elif isinstance(other_metrics, dict):
            pa = other_metrics.get("parent_a")
            pb = other_metrics.get("parent_b")
            if pa and pb:
                parent_id = [pa, pb]

    # --- role: prefer new column, fall back to inference ---
    role_str = row.get("role")
    if role_str:
        role = Role(role_str)
    else:
        role = Role.LEADER if parent_id is None else Role.MEMBER

    # --- description: stored in 'algorithm' column ---
    description = row.get("algorithm") or ""

    # --- raw_par2_score: prefer new JSON list in par2 column ---
    raw_par2_score = _to_json_list_float(row.get("par2"))

    # --- normalized_par2_score ---
    normalized_par2_score = _to_json_list_float(row.get("normalized_par2_scores"))

    # --- parent_algorithm_description ---
    parent_algorithm_description = _to_json_list(row.get("parent_descriptions"))

    # --- analysis ---
    analysis = row.get("analysis")

    # Clean legacy keys from other_metrics
    if isinstance(other_metrics, dict):
        other_metrics.pop("target_function", None)
        other_metrics.pop("parent_id", None)

    return AlgorithmResult(
        id=row.get("id"),
        function_name=function_name,
        description=description,
        role=role,
        status=row.get("status"),
        last_updated=row.get("last_updated"),
        code_id_list=_text_to_code_id_list(row.get("code_id_list")),
        parent_id=parent_id,
        parent_algorithm_description=parent_algorithm_description,
        raw_par2_score=raw_par2_score,
        normalized_par2_score=normalized_par2_score,
        analysis=analysis,
        prompt=row.get("prompt"),
        other_metrics=other_metrics,
    )

def add_router_table(name: str):
    # router tables has: id, type
    if USE_LOCAL_CACHE:
        path = _router_path(name)
        if not path.exists():
            _write_json(path, {})
        return
    conn = connect_to_db()
    try:
        cur = conn.cursor()
        cur.execute(f"CREATE TABLE IF NOT EXISTS {name} (id TEXT PRIMARY KEY, type TEXT);")
        conn.commit()
    finally:
        release_conn(conn)

def update_router_table(name: str, id: str, type: str):
    if USE_LOCAL_CACHE:
        path = _router_path(name)
        with _locked_local_path(path):
            rows = _read_json(path, {})
            rows[id] = type
            _write_json_unlocked(path, rows)
        return
    conn = connect_to_db()
    try:
        cur = conn.cursor()
        # if exists, update, if not insert
        cur.execute(f"SELECT * FROM {name} WHERE id = %s;", (id,))
        if cur.rowcount > 0:
            cur.execute(f"UPDATE {name} SET type = %s WHERE id = %s;", (type, id))
        else:
            cur.execute(f"INSERT INTO {name} (id, type) VALUES (%s, %s);", (id, type))
        conn.commit()
    finally:
        release_conn(conn)

def get_ids_from_router_table(name: str, type: str) -> List[str]:
    if USE_LOCAL_CACHE:
        rows = _read_json(_router_path(name), {})
        if type is None:
            return list(rows)
        return [record_id for record_id, record_type in rows.items() if record_type == type]
    conn = connect_to_db()
    try:
        cur = conn.cursor()
        if type is None:
            cur.execute(f"SELECT id FROM {name};")
        else:
            cur.execute(f"SELECT id FROM {name} WHERE type = %s;", (type,))
        rows = cur.fetchall()
        return list(set([row[0] for row in rows]))
    finally:
        release_conn(conn)

def clear_router_table(name: str):
    if USE_LOCAL_CACHE:
        _write_json(_router_path(name), {})
        return
    conn = connect_to_db()
    try:
        cur = conn.cursor()
        cur.execute(f"DELETE FROM {name};")
        conn.commit()
    finally:
        release_conn(conn)

def add_par2_to_code_results_table():
    if USE_LOCAL_CACHE:
        return
    conn = connect_to_db()
    try:
        cur = conn.cursor()
        cur.execute("ALTER TABLE code_results ADD COLUMN par2 TEXT;")
        conn.commit()
    finally:
        release_conn(conn)

def backup_db():
    if USE_LOCAL_CACHE:
        backup_root = LOCAL_CACHE_ROOT.parent / f"{LOCAL_CACHE_ROOT.name}_backup"
        shutil.copytree(LOCAL_CACHE_ROOT, backup_root, dirs_exist_ok=True)
        logger.info(f"Backed up local cache to {backup_root}")
        return
    conn = connect_to_db()
    try:
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS algorithm_results_backup (id TEXT PRIMARY KEY, algorithm TEXT, code_id_list TEXT, status TEXT, last_updated TEXT, prompt TEXT, par2 TEXT, error_rate TEXT, other_metrics TEXT);")
        cur.execute("CREATE TABLE IF NOT EXISTS code_results_backup (id TEXT PRIMARY KEY, code TEXT, algorithm TEXT, status TEXT, last_updated TEXT, solver_id TEXT, build_success TEXT);")
        cur.execute("INSERT INTO algorithm_results_backup SELECT * FROM algorithm_results;")
        cur.execute("INSERT INTO code_results_backup SELECT * FROM code_results;")
        conn.commit()
    finally:
        release_conn(conn)

def migrate_algorithm_table():
    """Add new columns to algorithm_results table and backfill existing records.

    Safe to run multiple times — uses IF NOT EXISTS / idempotent updates.
    """
    if USE_LOCAL_CACHE:
        logger.info("Local cache already uses the current AlgorithmResult schema")
        return
    conn = connect_to_db()
    try:
        cur = conn.cursor()

        # Add new columns (idempotent via try/except per column)
        new_columns = ["role", "function_name", "parent_ids",
                       "parent_descriptions", "normalized_par2_scores", "analysis"]
        for col in new_columns:
            try:
                cur.execute(f"ALTER TABLE algorithm_results ADD COLUMN {col} TEXT;")
                conn.commit()
                logger.info(f"Added column {col}")
            except Exception:
                conn.rollback()
                logger.info(f"Column {col} already exists, skipping")

        # Backfill existing records that have NULL in the new columns
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM algorithm_results WHERE role IS NULL;")
        rows = cur.fetchall()
        logger.info(f"Migrating {len(rows)} records...")

        cur2 = conn.cursor()
        for row in rows:
            other_metrics = _to_other_metrics(row.get("other_metrics")) or {}

            # Infer function_name
            function_name = "kissat_restarting"
            if isinstance(other_metrics, dict) and "target_function" in other_metrics:
                function_name = other_metrics["target_function"]

            # Infer parent_ids from other_metrics
            parent_ids = None
            if isinstance(other_metrics, dict):
                pid = other_metrics.get("parent_id")
                pa = other_metrics.get("parent_a")
                pb = other_metrics.get("parent_b")
                if pid is not None:
                    parent_ids = json.dumps([pid])
                elif pa and pb:
                    parent_ids = json.dumps([pa, pb])

            # Infer role
            role = "leader" if parent_ids is None else "member"

            # Convert algorithm JSON string to description format
            description = row.get("algorithm") or ""
            try:
                spec = json.loads(description)
                if isinstance(spec, dict) and "algorithm" in spec:
                    name = spec.get("name", "")
                    algo_text = spec.get("algorithm", "")
                    description = f"{name}: {algo_text}" if name else algo_text
            except Exception:
                pass  # already plain text or unparseable, keep as-is

            cur2.execute(
                """UPDATE algorithm_results
                   SET role = %s, function_name = %s, parent_ids = %s, algorithm = %s
                   WHERE id = %s;""",
                (role, function_name, parent_ids, description, row.get("id")),
            )

        conn.commit()
        logger.info(f"Migration complete: {len(rows)} records updated")
    finally:
        release_conn(conn)


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="AWS utils")
    parser.add_argument("--clear", action="store_true", help="Clear the tables")
    parser.add_argument("--delete", action="store_true", help="Delete the tables")
    parser.add_argument("--init", action="store_true", help="Initialize the tables")
    parser.add_argument("--test", action="store_true", help="Test the utils")
    parser.add_argument("--reset", action="store_true", help="Reset the tables")
    parser.add_argument("--show_code_results", type=str, help="Show the code results")
    parser.add_argument("--backup", action="store_true", help="Backup the tables")
    parser.add_argument("--add_router_table", type=str, default=None, help="Add a router table")
    parser.add_argument("--migrate", action="store_true", help="Add new columns and backfill existing records")
    args = parser.parse_args()
    if args.show_code_results:
        code_results = get_code_result(args.show_code_results)
        print(code_results)
    if args.clear:
        clear_tables()
    elif args.delete:
        delete_tables()
    elif args.init:
        init_tables()
    elif args.test:
        test_utils()
    elif args.reset:
        delete_tables()
        init_tables()
    elif args.backup:
        backup_db()
    elif args.migrate:
        migrate_algorithm_table()
    elif args.add_router_table:
        add_router_table(args.add_router_table)
    else:
        print("No action specified")
