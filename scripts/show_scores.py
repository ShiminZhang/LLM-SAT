"""Show PAR2 scores for a generation tag.

Usage: python scripts/show_scores.py <generation_tag>
"""
import sys
from llmsat.utils.aws import get_ids_from_router_table, get_algorithm_result
from llmsat.llmsat import CHATGPT_DATA_GENERATION_TABLE

tag = sys.argv[1]
ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, tag)
results = []
for aid in ids:
    a = get_algorithm_result(aid)
    if a and a.raw_par2_score and a.raw_par2_score[-1] is not None:
        results.append((a.raw_par2_score[-1], a.role.value, aid))
results.sort()
for par2, role, aid in results:
    print(f"{aid[:16]}  {role:<6}  PAR2={par2:.2f}")
