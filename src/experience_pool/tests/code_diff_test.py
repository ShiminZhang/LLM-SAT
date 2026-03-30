import difflib
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from llmsat.utils.aws import get_code_result


leader = "343f1d5121fdaffb198248a4ac6a4ac480c80eaa0af6ea5e5b7a8985416707ae"
leader = get_code_result(leader)
leader = leader.code

member = "c05998c9d187da2f878b542e616384fd01b54f6966765017ee4894f1b01f8e5e"

member = get_code_result(member)
member = member.code

lines1 = leader.splitlines(keepends=True)
lines2 = member.splitlines(keepends=True)

diff = difflib.unified_diff(
    lines1, 
    lines2, 
    fromfile='alg1', 
    tofile='alg2'
)

print(''.join(diff))