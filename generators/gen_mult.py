#!/usr/bin/env python3
"""
Integer Multiplication / Factorization SAT benchmark generator.

Generates CNF instances encoding the problem:
  "Does there exist (a, b) such that a * b = N?"

where a and b are n-bit integers (1 <= a, b <= 2^n - 1) and
N is a 2n-bit target value.

  - SAT instances: N is composite (N = p*q for some 1 < p, q < N)
  - UNSAT instances: N is prime (the only factorizations are 1*N and N*1,
    but we force a, b > 1 so UNSAT)

The encoding is a Tseitin transformation of an n-bit ripple-carry
multiplier circuit.

Usage:
  python gen_mult.py <n> <N>               # single instance, N given
  python gen_mult.py <n> --prime <p>       # UNSAT: prime N
  python gen_mult.py <n> --composite <p> <q>  # SAT: N = p*q
  python gen_mult.py --batch <output_dir>  # batch of instances

References:
  Tseitin, G.S. (1968). On the complexity of derivation in propositional calculus.
  Encoding from: Manthey, N. et al., SAT-based model checking of hardware multipliers.
"""

import sys
import os

# ---------------------------------------------------------------------------
# Variable management
# ---------------------------------------------------------------------------

class VarPool:
    def __init__(self):
        self._next = 1
        self._map = {}

    def new(self, name=None):
        v = self._next
        self._next += 1
        if name:
            self._map[name] = v
        return v

    def get(self, name):
        return self._map[name]

    def num_vars(self):
        return self._next - 1


# ---------------------------------------------------------------------------
# Gate encodings (Tseitin)
# ---------------------------------------------------------------------------

def and_gate(pool, a, b):
    """Encode z = AND(a, b); return z and clauses."""
    z = pool.new()
    clauses = [
        [-z, a],
        [-z, b],
        [z, -a, -b],
    ]
    return z, clauses


def xor_gate(pool, a, b):
    """Encode z = XOR(a, b); return z and clauses."""
    z = pool.new()
    clauses = [
        [-z, a, b],
        [-z, -a, -b],
        [z, a, -b],
        [z, -a, b],
    ]
    return z, clauses


def half_adder(pool, a, b):
    """Encode half adder: (sum, carry) = HA(a, b)."""
    s, c1 = xor_gate(pool, a, b)
    c, c2 = and_gate(pool, a, b)
    return s, c, c1 + c2


def full_adder(pool, a, b, cin):
    """Encode full adder: (sum, cout) = FA(a, b, cin)."""
    # s1 = a XOR b
    s1, cls1 = xor_gate(pool, a, b)
    # s = s1 XOR cin
    s, cls2 = xor_gate(pool, s1, cin)
    # c1 = a AND b
    c1, cls3 = and_gate(pool, a, b)
    # c2 = s1 AND cin
    c2, cls4 = and_gate(pool, s1, cin)
    # cout = c1 OR c2
    cout = pool.new()
    cls5 = [
        [cout, -c1],
        [cout, -c2],
        [-cout, c1, c2],
    ]
    return s, cout, cls1 + cls2 + cls3 + cls4 + cls5


# ---------------------------------------------------------------------------
# n-bit multiplier (schoolbook / partial-product method)
# ---------------------------------------------------------------------------

def build_multiplier(pool, n):
    """
    Build an n-bit x n-bit unsigned multiplier.

    Creates variables for:
      a[0..n-1]  : bits of first factor (LSB first)
      b[0..n-1]  : bits of second factor (LSB first)
      prod[0..2n-1] : bits of product (LSB first)

    Returns (a_vars, b_vars, prod_vars, all_clauses).
    """
    all_clauses = []

    # Input variables
    a = [pool.new(f"a{i}") for i in range(n)]
    b = [pool.new(f"b{i}") for i in range(n)]

    # Partial products: pp[i][j] = a[i] AND b[j]
    pp = []
    for i in range(n):
        row = []
        for j in range(n):
            z, cls = and_gate(pool, a[i], b[j])
            all_clauses.extend(cls)
            row.append(z)
        pp.append(row)

    # Build product bits using ripple-carry adder on partial products.
    # The partial products form an n x n grid; column k contributes to bit k.
    # We accumulate using a sequence of full/half adders.

    # columns[k] = list of bits that contribute to product bit k
    columns = [[] for _ in range(2 * n)]
    for i in range(n):
        for j in range(n):
            columns[i + j].append(pp[i][j])

    prod_vars = []

    carry_ins = []  # carries feeding into the next column

    for k in range(2 * n):
        # Gather bits for column k: column bits + incoming carries
        bits = columns[k] + carry_ins
        carry_ins = []

        if len(bits) == 0:
            # No bits -> product bit k = 0 (enforce as unit clause later)
            z = pool.new()
            all_clauses.append([-z])  # z = 0
            prod_vars.append(z)
        elif len(bits) == 1:
            prod_vars.append(bits[0])
        elif len(bits) == 2:
            s, c, cls = half_adder(pool, bits[0], bits[1])
            all_clauses.extend(cls)
            prod_vars.append(s)
            carry_ins.append(c)
        else:
            # Reduce pairs with full adders, then handle remainder
            current = bits
            while len(current) >= 3:
                s, c, cls = full_adder(pool, current[0], current[1], current[2])
                all_clauses.extend(cls)
                carry_ins.append(c)
                current = current[3:] + [s]
            if len(current) == 2:
                s, c, cls = half_adder(pool, current[0], current[1])
                all_clauses.extend(cls)
                carry_ins.append(c)
                prod_vars.append(s)
            else:
                prod_vars.append(current[0])

    # Handle any remaining carry chain after the last column
    while carry_ins:
        # These carry bits go beyond 2n bits (overflow); we just let them be
        # (they can't be 1 if a, b < 2^n, but let's add them as product bits
        #  for completeness, even though for valid inputs they should be 0).
        prod_vars.append(carry_ins.pop(0))

    return a, b, prod_vars, all_clauses


def fix_bits(var_list, value, nbits):
    """
    Return unit clauses that fix var_list[0..nbits-1] to the binary
    representation of value (LSB first).
    """
    clauses = []
    for i in range(nbits):
        bit = (value >> i) & 1
        if bit:
            clauses.append([var_list[i]])
        else:
            clauses.append([-var_list[i]])
    return clauses


def force_greater_than_one(pool, var_list, n):
    """
    Enforce that the n-bit number represented by var_list is > 1.
    (i.e., not equal to 0 and not equal to 1)
    This means: not(all zero) AND not(bit0=1, bit1..=0)

    We add:
      - at least one bit is 1 (not zero): OR(var_list)
      - not equal to 1: NOT(bit0=1 AND bit1=0 AND ... AND bit_{n-1}=0)
        = bit0=0 OR bit1=1 OR ... OR bit_{n-1}=1
        = (-var_list[0]) OR var_list[1] OR ... OR var_list[n-1]
    """
    clauses = []
    # Not zero
    clauses.append(list(var_list))
    # Not one: at least one of bit1..bit_{n-1} is 1, or bit0=0
    clauses.append([-var_list[0]] + list(var_list[1:]))
    return clauses


def gen_factorization_cnf(n, N, force_nontrivial=True):
    """
    Generate a CNF for: does there exist (a, b) s.t. a * b = N,
    where a, b are n-bit numbers.

    If force_nontrivial=True, add constraints a > 1 and b > 1
    (so trivial factorizations a=1 or b=1 are excluded).

    N must fit in 2n bits: 0 <= N < 2^(2n).
    """
    pool = VarPool()
    all_clauses = []

    a, b, prod, mult_clauses = build_multiplier(pool, n)
    all_clauses.extend(mult_clauses)

    # Fix product bits to N
    prod_bits = 2 * n
    # Make sure prod has at least 2n variables
    while len(prod) < prod_bits:
        z = pool.new()
        all_clauses.append([-z])  # forced to 0
        prod.append(z)

    for i in range(prod_bits):
        bit = (N >> i) & 1
        if bit:
            all_clauses.append([prod[i]])
        else:
            all_clauses.append([-prod[i]])

    if force_nontrivial:
        all_clauses.extend(force_greater_than_one(pool, a, n))
        all_clauses.extend(force_greater_than_one(pool, b, n))

    return pool.num_vars(), all_clauses


def write_dimacs(num_vars, clauses, comment_lines=None, fileobj=None):
    """Write a DIMACS CNF file."""
    out = fileobj or sys.stdout
    if comment_lines:
        for line in comment_lines:
            out.write(f"c {line}\n")
    out.write(f"p cnf {num_vars} {len(clauses)}\n")
    for clause in clauses:
        out.write(" ".join(map(str, clause)) + " 0\n")


def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def generate_instance(n, N, output_path, force_nontrivial=True):
    """Generate a factorization instance and write to output_path."""
    status = "UNSAT" if is_prime(N) else "SAT"
    num_vars, clauses = gen_factorization_cnf(n, N, force_nontrivial)
    comments = [
        f"Integer factorization benchmark instance",
        f"n={n} bits, Target N={N} ({'prime' if is_prime(N) else 'composite'})",
        f"Problem: Find a,b > 1 (n-bit) s.t. a*b = {N}",
        f"Status: {status}",
        f"Variables: {num_vars}, Clauses: {len(clauses)}",
    ]
    with open(output_path, 'w') as f:
        write_dimacs(num_vars, clauses, comments, f)
    return num_vars, len(clauses), status


# Batch instances: (n_bits, N)
# For n=k bits, a and b are k-bit (< 2^k), product fits in 2k bits
# UNSAT: N is prime (no non-trivial n-bit factorization exists)
# SAT:   N = p*q where both p,q in (1, 2^k)
BATCH_INSTANCES = [
    # (n_bits, N, note)
    # 8-bit factors (product fits in 16 bits, max=65025=255^2)
    (8,  65003,       "prime, UNSAT"),      # large prime < 255^2 = 65025
    (8,  58081,       "241^2, SAT"),        # 241*241, both 8-bit
    (8,  57121,       "239^2, SAT"),        # 239*239, both 8-bit
    (8,  63499,       "prime, UNSAT"),      # prime < 65025
    # 12-bit factors (product fits in 24 bits, max=16769025=4095^2)
    (12, 4194319,     "prime, UNSAT"),      # prime near 2^22
    (12, 16752649,    "4093^2, SAT"),       # 4093*4093, 4093 prime, < 4096
    # 16-bit factors (product fits in 32 bits, max=4294836225=65535^2)
    (16, 4294967291,  "prime, UNSAT"),      # prime near 2^32
    (16, 4294836225,  "65535^2, SAT"),      # 65535*65535, both 16-bit
    # 20-bit factors (product fits in 40 bits)
    (20, 549755813911, "large prime (~2^39), UNSAT"),  # 39-bit prime
    (20, 1046529,      "1023^2, SAT"),      # 1023*1023, both 20-bit
]


def batch_generate(output_dir):
    """Generate a batch of multiplication instances."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {len(BATCH_INSTANCES)} multiplication instances in {output_dir}/")
    for n, N, note in BATCH_INSTANCES:
        filename = f"mult_{n}bit_N{N}.cnf"
        output_path = os.path.join(output_dir, filename)
        try:
            num_vars, num_clauses, status = generate_instance(n, N, output_path)
            print(f"  {filename}: {num_vars} vars, {num_clauses} clauses [{status}] ({note})")
        except Exception as e:
            print(f"  ERROR generating {filename}: {e}")


if __name__ == "__main__":
    if "--batch" in sys.argv:
        idx = sys.argv.index("--batch")
        out_dir = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "mult_instances"
        batch_generate(out_dir)
    elif len(sys.argv) >= 3:
        n = int(sys.argv[1])
        N = int(sys.argv[2])
        output_file = sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "--output" else None
        if output_file:
            num_vars, num_clauses, status = generate_instance(n, N, output_file)
            print(f"Written to {output_file}: {num_vars} vars, {num_clauses} clauses [{status}]")
        else:
            num_vars, clauses = gen_factorization_cnf(n, N)
            write_dimacs(num_vars, clauses)
    else:
        print(__doc__)
       