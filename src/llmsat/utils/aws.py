import psycopg2
import psycopg2.extras
import argparse
import os
import json
from typing import List, Dict, Any, Mapping, Optional
from llmsat.llmsat import CodeResult, CodeStatus, AlgorithmResult, AlgorithmStatus
from datetime import datetime
from llmsat.llmsat import get_logger, setup_logging
logger = get_logger(__name__)
# for reference only
# @dataclass
# class AlgorithmResult:
#     id: str
#     algorithm: str
#     status: str
#     last_updated: str
#     prompt: str
#     par2: float
#     error_rate: float
#     code_id_list: List[str] # list of code ids that have been generated for this algorithm
#     other_metrics: Dict[str, float]

# @dataclass
# class CodeResult(TaskResult):
#     id: str
#     algorithm_id: str
#     code: str
#     status: str
#     last_updated: str
#     build_success: bool


def get_code_result_of_status(status: CodeStatus) -> List[CodeResult]:
    logger.info(f"Getting code results of status {status}")
    conn = connect_to_db()
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

def get_algorithm_result_of_status(status: AlgorithmStatus) -> List[AlgorithmResult]:
    conn = connect_to_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM algorithm_results WHERE status = %s;", (status,))
    rows = cur.fetchall()
    return [_row_to_algorithm_result(row) for row in rows]

def connect_to_db():
    # logger.info("trying to connect to db")
    conn = psycopg2.connect(
        host="llmsat.crac0kykqrxp.us-east-2.rds.amazonaws.com",
        database="postgres",
        user="Shimin",
        password=os.environ["DB_PASS"],
        port=5432,
    )
    # logger.info("connected to db")
    return conn

def get_all_tasks():
    conn = connect_to_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM tasks;")
    return cur.fetchall()

def get_code_result(code_result_id: str):
    conn = connect_to_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM code_results WHERE id = %s;", (code_result_id,))
    assert cur.rowcount <= 1, "hash collision"
    if cur.rowcount == 1:
        result = _row_to_code_result(cur.fetchone())
        return result
    else:
        return None

def get_algorithm_result(algorithm_result_id: str):
    conn = connect_to_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM algorithm_results WHERE id = %s;", (algorithm_result_id,))
    assert cur.rowcount <= 1, "hash collision"
    if cur.rowcount == 1:
        row = cur.fetchone()
        logger.info(f"Retrieved algorithm result {row}")
        return _row_to_algorithm_result(row)
    else:
        return None

def update_code_result(code_result: CodeResult):
    existing_code_result = get_code_result(code_result.id)
    conn = connect_to_db()
    cur = conn.cursor()
    build_success_text = None if code_result.build_success is None else str(code_result.build_success)
    logger.info(f"Updating code result {code_result.id}")
    code_result.last_updated = datetime.now()
    if existing_code_result is None: # add the code result
        cur.execute(
            "INSERT INTO code_results (id, code, algorithm, status, last_updated, build_success) VALUES (%s, %s, %s, %s, %s, %s);",
            (code_result.id, code_result.code, code_result.algorithm_id, code_result.status, code_result.last_updated, build_success_text),
        )
    else: # update the code result
        cur.execute(
            "UPDATE code_results SET code = %s, algorithm = %s, status = %s, last_updated = %s, build_success = %s WHERE id = %s;",
            (code_result.code, code_result.algorithm_id, code_result.status, code_result.last_updated, build_success_text, code_result.id),
        )
    conn.commit()
    logger.info(f"Updated code result {code_result.id}")

def update_algorithm_result(algorithm_result: AlgorithmResult):
    existing_algorithm_result = get_algorithm_result(algorithm_result.id)
    conn = connect_to_db()
    cur = conn.cursor()
    other_metrics_obj = algorithm_result.other_metrics
    algorithm_result.last_updated = datetime.now()
    if other_metrics_obj is None:
        other_metrics_text = None
    elif isinstance(other_metrics_obj, (dict, list)):
        other_metrics_text = json.dumps(other_metrics_obj, ensure_ascii=False)
    else:
        other_metrics_text = str(other_metrics_obj)
    if existing_algorithm_result is None: # add the algorithm result
        cur.execute(
            "INSERT INTO algorithm_results (id, algorithm, code_id_list, status, last_updated, prompt, par2, error_rate, other_metrics) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);",
            (
                algorithm_result.id,
                algorithm_result.algorithm,
                _code_id_list_to_text(algorithm_result.code_id_list),
                algorithm_result.status,
                algorithm_result.last_updated,
                algorithm_result.prompt,
                algorithm_result.par2,
                algorithm_result.error_rate,
                other_metrics_text,
            ),
        )
    else: # update the algorithm result
        cur.execute(
            "UPDATE algorithm_results SET algorithm = %s, code_id_list = %s, status = %s, last_updated = %s, prompt = %s, par2 = %s, error_rate = %s, other_metrics = %s WHERE id = %s;",
            (
                algorithm_result.algorithm,
                _code_id_list_to_text(algorithm_result.code_id_list),
                algorithm_result.status,
                algorithm_result.last_updated,
                algorithm_result.prompt,
                algorithm_result.par2,
                algorithm_result.error_rate,
                other_metrics_text,
                algorithm_result.id,
            ),
        )
    conn.commit()
    print(f"Updated algorithm result {algorithm_result.id}")

def init_tables():
    # call once only
    # algorithm table: id, algorithm, prompt, par2, error_rate, other_metrics
    # code table: id, code, algorithm, solver_id, build_success
    conn = connect_to_db()
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS algorithm_results (id TEXT PRIMARY KEY, algorithm TEXT, code_id_list TEXT, status TEXT, last_updated TEXT, prompt TEXT, par2 TEXT, error_rate TEXT, other_metrics TEXT);")
    cur.execute("CREATE TABLE IF NOT EXISTS code_results (id TEXT PRIMARY KEY, code TEXT, algorithm TEXT, status TEXT, last_updated TEXT, solver_id TEXT, build_success TEXT);")
    conn.commit()
    pass

def clear_tables():
    conn = connect_to_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM algorithm_results;")
    cur.execute("DELETE FROM code_results;")
    conn.commit()
    pass

def ToAlgorithmResult(result: tuple) -> AlgorithmResult:
    return AlgorithmResult(
        id=result[0],
        algorithm=result[1],
        status=result[3],
        last_updated=result[4],
        code_id_list=_text_to_code_id_list(result[2]),
        prompt=result[5],
        par2=_to_float(result[6]),
        error_rate=_to_float(result[7]),
        other_metrics=_to_other_metrics(result[8]))

def ToCodeResult(result: tuple) -> CodeResult:
    return _row_to_code_result(result)

def get_all_algorithm_results() -> List[AlgorithmResult]:
    conn = connect_to_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM algorithm_results;")
    rows = cur.fetchall()
    algorithm_results = []
    for row in rows:
        algorithm_results.append(_row_to_algorithm_result(row))
    return algorithm_results



def get_all_algorithm_ids():
    results = get_all_algorithm_results()
    return list(set([result.id for result in results]))

def delete_tables():
    conn = connect_to_db()
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS algorithm_results;")
    cur.execute("DROP TABLE IF EXISTS code_results;")
    conn.commit()
    pass

def test_utils():
    code_result = CodeResult(
        id="2",
        algorithm_id="1",
        code="return false;",
        status=CodeStatus.Generated,
        last_updated=datetime.now(),
        build_success=None
    )
    update_code_result(code_result)
    algorithm_result = AlgorithmResult(
        id="1",
        algorithm="kissat_restarting_policy",
        status=AlgorithmStatus.Generated,
        last_updated=datetime.now(),
        prompt="",
        par2=22,
        error_rate=0,
        code_id_list=[],
        other_metrics={}
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
def _row_to_code_result(row: Mapping[str, Any]) -> CodeResult:
    return CodeResult(
        id=row.get("id"),
        code=row.get("code"),
        algorithm_id=row.get("algorithm_id"),
        status=row.get("status"),
        last_updated=row.get("last_updated"),
        build_success=_to_bool(row.get("build_success")),
    )
def _row_to_algorithm_result(row: Mapping[str, Any]) -> AlgorithmResult:
    return AlgorithmResult(
        id=row.get("id"),
        algorithm=row.get("algorithm"),
        status=row.get("status"),
        last_updated=row.get("last_updated"),
        prompt=row.get("prompt"),
        par2=_to_float(row.get("par2")),
        error_rate=_to_float(row.get("error_rate")),
        code_id_list=_text_to_code_id_list(row.get("code_id_list")),
        other_metrics=_to_other_metrics(row.get("other_metrics")),
    )

if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="AWS utils")
    parser.add_argument("--clear", action="store_true", help="Clear the tables")
    parser.add_argument("--delete", action="store_true", help="Delete the tables")
    parser.add_argument("--init", action="store_true", help="Initialize the tables")
    parser.add_argument("--test", action="store_true", help="Test the utils")
    parser.add_argument("--reset", action="store_true", help="Reset the tables")
    parser.add_argument("--show_code_results", type=str, help="Show the code results")
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
    else:
        print("No action specified")