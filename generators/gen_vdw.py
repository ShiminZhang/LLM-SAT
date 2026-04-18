#!/usr/bin/env python3
"""
Van der Waerden (VDW) SAT benchmark generator.

Generates CNF instances encoding the Van der Waerden problem:
Can we k-color {1..n} such that no color class contains an
arithmetic progression of length l_i (for color i)?

Usage:
  python gen_vdw.py <k> <l1> [l2 ...] <n>
  python gen_vdw.py --batch   # generate a batch of instances

The DIMACS CNF file is written to stdout (or a file if --output is given).

References:
  Van der Waerden, B.L. (1927). Beweis einer Baudetschen Vermutung.
  Nieuw Arch. Wiskunde 15, 212-216.
"""

import sys
import itertools
import os

def vdw_cnf(n, lengths):
    """
    Generate CNF clauses for the VDW(k; l_1, ..., l_k) problem on {1..n}.

    Variables: var(i, c) = (i-1)*k + c + 1  (1-indexed)
      where i in 1..n (position), c in 0..k-1 (color)

    Clauses:
      1. At-least-one color for each position
      2. At-most-one color (pairwise) for each position
      3. No monochromatic AP of length lengths[c] in color c
    """
    k = len(lengths)

    def var(i, c):
        """1-indexed variable for position i (1-based) and color c (0-based)."""
        return (i - 1) * k + c + 1

    num_vars = n * k
    clauses = []

    # At-least-one color per position
    for i in range(1, n + 1):
        clauses.append([var(i, c) for c in range(k)])

    # At-most-one color per position (pairwise encoding)
    for i in range(1, n + 1):
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                clauses.append([-var(i, c1), -var(i, c2)])

    # No monochromatic AP of length l in color c
    for c in range(k):
        l = lengths[c]
        # Enumerate all APs of length l within {1..n}
        for start in range(1, n + 1):
            for step in range(1, n + 1):
                ap = [start + j * step for j in range(l)]
                if ap[-1] > n:
                    break
                # No monochromatic AP: at least one element is not color c
                clauses.append([-var(pos, c) for pos in ap])

    return num_vars, clauses


def write_dimacs(num_vars, clauses, comment_lines=None, fileobj=None):
    """Write a DIMACS CNF file."""
    out = fileobj or sys.stdout
    if comment_lines:
        for line in comment_lines:
            out.write(f"c {line}\n")
    out.write(f"p cnf {num_vars} {len(clauses)}\n")
    for clause in clauses:
        out.write(" ".join(map(str, clause)) + " 0\n")


def generate_instance(n, lengths, output_path):
    """Generate a VDW instance and write to output_path."""
    k = len(lengths)
    lengths_str = "_".join(map(str, lengths))
    num_vars, clauses = vdw_cnf(n, lengths)
    comments = [
        f"Van der Waerden benchmark instance",
        f"Colors: {k}, Lengths: {lengths}, N: {n}",
        f"Problem: Can we {k}-color {{1..{n}}} without monochromatic APs of lengths {lengths}?",
        f"Variables: {num_vars}, Clauses: {len(clauses)}",
    ]
    with open(output_path, 'w') as f:
        write_dimacs(num_vars, clauses, comments, f)
    return num_vars, len(clauses)


# Known/approximate VDW numbers for reference
# W(2; l, l) - the symmetric 2-color case
# W(3) = 9, W(4) = 35, W(5) = 178, W(6) = 1132, W(7) = 3703
# W(2; 3, k): W(3,4)=18, W(3,5)=22, W(3,6)=32, W(3,7)=46, W(3,8)=58
# W(3; 3, 3, 3) = 27

BATCH_INSTANCES = [
    # (lengths, n, sat_status) - instances near the VDW threshold
    # 2-color symmetric: W(2;k,k)
    ([3, 3], 8,  "SAT"),   # W(3,3)=9; n=8 is SAT
    ([3, 3], 9,  "UNSAT"), # W(3,3)=9; n=9 is UNSAT (= W number)
    ([4, 4], 34, "SAT"),   # W(4,4)=35; n=34 is SAT
    ([4, 4], 35, "UNSAT"), # W(4,4)=35; n=35 is UNSAT
    ([5, 5], 177, "SAT"),  # W(5,5)=178; n=177 is SAT
    ([5, 5], 178, "UNSAT"),# W(5,5)=178; n=178 is UNSAT
    # 2-color asymmetric: W(3,k)
    ([3, 7], 45, "SAT"),   # W(3,7)=46; n=45 is SAT
    ([3, 7], 46, "UNSAT"), # W(3,7)=46; n=46 is UNSAT
    ([3, 8], 57, "SAT"),   # W(3,8)=58; n=57 is SAT
    ([3, 8], 58, "UNSAT"), # W(3,8)=58; n=58 is UNSAT
]


def batch_generate(output_dir):
    """Generate a batch of VDW instances."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {len(BATCH_INSTANCES)} VDW instances in {output_dir}/")
    for lengths, n, status in BATCH_INSTANCES:
        lengths_str = "_".join(map(str, lengths))
        filename = f"vdw_{lengths_str}_n{n}.cnf"
        output_path = os.path.join(output_dir, filename)
        num_vars, num_clauses = generate_instance(n, lengths, output_path)
        print(f"  {filename}: {num_vars} vars, {num_clauses} clauses [{status}]")


if __name__ == "__main__":
    if "--batch" in sys.argv:
        out_dir = sys.argv[sys.argv.index("--batch") + 1] if "--batch" in sys.argv and sys.argv.index("--batch") + 1 < len(sys.argv) else "vdw_instances"
        batch_generate(out_dir)
    elif len(sys.argv) >= 3:
        # Usage: gen_vdw.py <l1> [l2 ...] <n>
        # or:    gen_vdw.py <l1> [l2 ...] <n> --output <file>
        args = sys.argv[1:]
        if "--output" in args:
            idx = args.index("--output")
            output_file = args[idx + 1]
            args = args[:idx]
        else:
            output_file = None

        # Parse: last arg is n, rest are lengths
        n = int(args[-1])
        lengths = list(map(int, args[:-1]))

        if output_file:
            num_vars, num_clauses = generate_instance(n, lengths, output_file)
            print(f"Written to {output_file}: {num_vars} vars, {num_clauses} clauses")
        else:
            num_vars, clauses = vdw_cnf(n, lengths)
            write_dimacs(num_vars, clauses)
    else:
        print(__doc__)
        print("\nExample: python gen_vdw.py 3 3 9   # VDW(2;3,3) on n=9 (UNSAT)")
        print("         python gen_vdw.py 4 4 35  # VDW(2;4,4) on n=35 (UNSAT)")
        print("         python gen_vdw.py --batch vdw_instances/")
