import psycopg2
import os
# from src.llmsat.llmsat import CodeResult, TaskResult

def connect_to_db():
    print("trying to connect to db")
    return psycopg2.connect(
        host="llmsat.crac0kykqrxp.us-east-2.rds.amazonaws.com",
        database="postgres",
        user="Shimin",
        password=os.environ["DB_PASS"],
        port=5432,
    )

def get_all_tasks():
    conn = connect_to_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM tasks;")
    return cur.fetchall()

def get_code_result(code_result_id: str):
    conn = connect_to_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM code_results WHERE id = %s;", (code_result_id,))
    assert cur.rowcount <= 1, "hash collision"
    if cur.rowcount == 1:
        return cur.fetchone()
    else:
        return None

def get_algorithm_result(algorithm_result_id: str):
    conn = connect_to_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM algorithm_results WHERE id = %s;", (algorithm_result_id,))
    assert cur.rowcount <= 1, "hash collision"
    if cur.rowcount == 1:
        return cur.fetchone()
    else:
        return None

def update_code_result(code_result: CodeResult):
    existing_code_result = get_code_result(code_result.id)
    conn = connect_to_db()
    cur = conn.cursor()
    if existing_code_result is None: # add the code result
        cur.execute("INSERT INTO code_results (id, code, algorithm, solver_id, build_success) VALUES (%s, %s, %s, %s, %s);", (code_result.id, code_result.code, code_result.algorithm, code_result.solver_id, code_result.build_success))
    else: # update the code result
        cur.execute("UPDATE code_results SET code = %s, algorithm = %s, solver_id = %s, build_success = %s WHERE id = %s;", (code_result.code, code_result.algorithm, code_result.solver_id, code_result.build_success, code_result.id))
    conn.commit()

def update_algorithm_result(algorithm_result: AlgorithmResult):
    existing_algorithm_result = get_algorithm_result(algorithm_result.id)
    conn = connect_to_db()
    cur = conn.cursor()
    if existing_algorithm_result is None: # add the algorithm result
        cur.execute("INSERT INTO algorithm_results (id, algorithm, prompt, par2, error_rate, other_metrics) VALUES (%s, %s, %s, %s, %s, %s);", (algorithm_result.id, algorithm_result.algorithm, algorithm_result.prompt, algorithm_result.par2, algorithm_result.error_rate, algorithm_result.other_metrics))
    else: # update the algorithm result
        cur.execute("UPDATE algorithm_results SET algorithm = %s, prompt = %s, par2 = %s, error_rate = %s, other_metrics = %s WHERE id = %s;", (algorithm_result.algorithm, algorithm_result.prompt, algorithm_result.par2, algorithm_result.error_rate, algorithm_result.other_metrics, algorithm_result.id))
    conn.commit()

def init_tables():
    # call once only

    pass

if __name__ == "__main__":
    conn = connect_to_db()
    print(conn)