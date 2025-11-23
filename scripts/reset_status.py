#!/usr/bin/env python
"""Reset algorithm and code result statuses to allow re-evaluation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llmsat.utils.aws import connect_to_db

def main():
    conn = connect_to_db()
    cur = conn.cursor()

    # See what statuses exist
    cur.execute("SELECT status, COUNT(*) FROM algorithm_results GROUP BY status")
    print("Current algorithm statuses:", cur.fetchall())

    cur.execute("SELECT status, COUNT(*) FROM code_results GROUP BY status")
    print("Current code statuses:", cur.fetchall())

    # Reset all evaluated/evaluating algorithms back to code_generated
    cur.execute("UPDATE algorithm_results SET status = 'code_generated' WHERE status IN ('evaluated', 'evaluating')")
    algo_count = cur.rowcount
    conn.commit()
    print(f"Reset {algo_count} algorithms to code_generated")

    # Also reset code results to generated so they get re-evaluated
    cur.execute("UPDATE code_results SET status = 'generated', build_success = NULL WHERE status IN ('evaluated', 'evaluating', 'build_failed')")
    code_count = cur.rowcount
    conn.commit()
    print(f"Reset {code_count} code results to generated")

    # Show new statuses
    cur.execute("SELECT status, COUNT(*) FROM algorithm_results GROUP BY status")
    print("New algorithm statuses:", cur.fetchall())

    cur.execute("SELECT status, COUNT(*) FROM code_results GROUP BY status")
    print("New code statuses:", cur.fetchall())

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
