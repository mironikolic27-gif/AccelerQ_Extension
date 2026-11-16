#!/usr/bin/env python3
"""
test a fixed set of UCCSD hyperparameter combinations on a single
8-qubit Hamiltonian to identify which parameter combinations are causing Stage 1
mining failures.

"""

import argparse
import csv
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
# kcl_util appends a relative "../" and imports utils.challenge_2024, so the repo
# root must be importable regardless of the cwd this script is launched from.
for _path in (SRC_DIR, REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# ---------------------------------------------------------------------------
# Config — paths are absolute so the script is cwd-independent
# ---------------------------------------------------------------------------
HAM_FOLDER = os.path.join(REPO_ROOT, "hamiltonian")
HAM_FILE = "08qubits_05.data"
HAM_SEED = 5          # matches filename suffix
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "uccsd_diag_8q_results.csv")

# Mirrors kcl_util_UCCSD.ITER_MAX_CAP; asserted against it at run time.
ITER_MAX_CAP = 500
DEFAULT_SEED = 42
# initial_point_seed=0 is the lower bound of the sanitised IND:12 range, so it is
# covered by several combinations rather than a single spot check.
SEED0 = 0
MIN_SEED0_RUNS = 4

CSV_FIELDS = [
    "ham",
    "label",
    "reps",
    "electron_fraction",
    "generalized",
    "initial_state",
    "optimizer",
    "iter_max",
    "initial_point_seed",
    "status",
    "energy",
    "error",
]

# ---------------------------------------------------------------------------
# Diagnostic combinations
# Axes: reps in {1,2,3}, electron_fraction in {0,1}, generalized in {0,1}
# Fixed unless overridden: initial_state=1 (HF, stable), optimizer=0 (COBYLA),
#                   iter_max=500, coeff_cutoff=1e-5, atol=1e-6,
#                   gradient_method=0 (required by COBYLA), initial_point_seed=42,
#                   max_evals_grouped=5
#
# generalized=0, vary reps x electron_fraction  (should succeed)
# generalized=1, vary reps x electron_fraction  (clamped to 0 at 8q)
# generalized=0, reps=1, vary initial_state     (stability check)
# generalized=0, reps=2, vary optimizer         (optimizer check)
# generalized=0, vary iter_max under the cap    (iter_max sensitivity)
# generalized=0, initial_point_seed=0           (start-point sensitivity)
# ---------------------------------------------------------------------------
#
# x_vec index table (14 elements):
#   0=n_qubits, 1=is_classical, 2=initial_state, 3=reps, 4=electron_fraction,
#   5=generalized, 6=optimizer_choice, 7=iter_max, 8=coeff_cutoff, 9=atol,
#   10=shots, 11=gradient_method, 12=initial_point_seed, 13=max_evals_grouped


def make_vec(n_qubits, initial_state, reps, electron_fraction, generalized,
             optimizer=0, iter_max=ITER_MAX_CAP, gradient_method=0,
             initial_point_seed=DEFAULT_SEED):
    if iter_max > ITER_MAX_CAP:
        raise ValueError(
            f"iter_max={iter_max} exceeds ITER_MAX_CAP={ITER_MAX_CAP}; "
            "the wrapper would clamp it and the CSV would record a value that never ran"
        )
    return np.array([
        float(n_qubits),      # 0
        1.0,                  # 1  is_classical
        float(initial_state), # 2
        float(reps),          # 3
        float(electron_fraction), # 4
        float(generalized),   # 5
        float(optimizer),     # 6
        float(iter_max),      # 7
        1e-5,                 # 8  coeff_cutoff
        1e-6,                 # 9  atol
        1e4,                  # 10 shots (ignored, classical)
        float(gradient_method), # 11
        float(initial_point_seed), # 12
        5.0,                  # 13 max_evals_grouped
    ], dtype=float)


N = 8

COMBINATIONS = [
    # label,                                vec
    # --- generalized=0 baseline ---
    ("g0_r1_ef0",  make_vec(N, 1, 1, 0, 0)),
    ("g0_r2_ef0",  make_vec(N, 1, 2, 0, 0)),
    ("g0_r3_ef0",  make_vec(N, 1, 3, 0, 0)),
    ("g0_r1_ef1",  make_vec(N, 1, 1, 1, 0)),
    ("g0_r2_ef1",  make_vec(N, 1, 2, 1, 0)),
    ("g0_r3_ef1",  make_vec(N, 1, 3, 1, 0)),
    # --- generalized=1 requested (sanitiser forces 0 at 8q) ---
    ("g1_r1_ef0",  make_vec(N, 1, 1, 0, 1)),
    ("g1_r2_ef0",  make_vec(N, 1, 2, 0, 1)),
    ("g1_r3_ef0",  make_vec(N, 1, 3, 0, 1)),
    ("g1_r1_ef1",  make_vec(N, 1, 1, 1, 1)),
    ("g1_r2_ef1",  make_vec(N, 1, 2, 1, 1)),
    ("g1_r3_ef1",  make_vec(N, 1, 3, 1, 1)),
    # --- initial_state variation (g0, r1, ef0) ---
    ("g0_r1_ef0_is0", make_vec(N, 0, 1, 0, 0)),  # no initial state
    ("g0_r1_ef0_is2", make_vec(N, 2, 1, 0, 0)),  # fixed Pauli rotation
    # --- optimizer variation (g0, r2, ef0, gradient_method adjusted) ---
    ("g0_r2_ef0_spsa",   make_vec(N, 1, 2, 0, 0, optimizer=1, gradient_method=0)),
    ("g0_r2_ef0_lbfgs",  make_vec(N, 1, 2, 0, 0, optimizer=2, gradient_method=0)),
    ("g0_r2_ef0_lbfgs_ps", make_vec(N, 1, 2, 0, 0, optimizer=2, gradient_method=1)),
    # --- iter_max sensitivity under the 500 cap (g0, ef0) ---
    ("g0_r1_ef0_imax10",   make_vec(N, 1, 1, 0, 0, iter_max=10)),
    ("g0_r1_ef0_imax100",  make_vec(N, 1, 1, 0, 0, iter_max=100)),
    ("g0_r2_ef0_imax250",  make_vec(N, 1, 2, 0, 0, iter_max=250)),
    # --- initial_point_seed=0 (g0, iter_max at the cap) ---
    ("g0_r1_ef0_seed0", make_vec(N, 1, 1, 0, 0, initial_point_seed=SEED0)),
    ("g0_r2_ef0_seed0", make_vec(N, 1, 2, 0, 0, initial_point_seed=SEED0)),
    ("g0_r3_ef0_seed0", make_vec(N, 1, 3, 0, 0, initial_point_seed=SEED0)),
    ("g0_r1_ef1_seed0", make_vec(N, 1, 1, 1, 0, initial_point_seed=SEED0)),
]

_labels = [label for label, _ in COMBINATIONS]
assert len(set(_labels)) == len(_labels), "duplicate combination labels"
assert all(int(round(vec[7])) <= ITER_MAX_CAP for _, vec in COMBINATIONS),\
    f"every combination must request iter_max <= {ITER_MAX_CAP}"
_n_seed0 = sum(1 for _, vec in COMBINATIONS if int(round(vec[12])) == SEED0)
assert _n_seed0 >= MIN_SEED0_RUNS, (
    f"expected at least {MIN_SEED0_RUNS} combinations with "
    f"initial_point_seed={SEED0}, got {_n_seed0}"
)


def load_sanitizer():
    """
    Return (sanitize_fn, is_effective). Falls back to identity when the quantum
    stack is unavailable, so --dry-run works on a laptop; a real run always uses
    the wrapper's own sanitiser.
    """
    try:
        from kcl_util_UCCSD import ITER_MAX_CAP as WRAPPER_ITER_MAX_CAP
        from kcl_util_UCCSD import sanitize_x_vec_params
    except Exception:
        return (lambda x: np.array(x, dtype=float, copy=True)), False
    assert WRAPPER_ITER_MAX_CAP == ITER_MAX_CAP, (
        f"iter_max cap drift: this script uses {ITER_MAX_CAP}, "
        f"kcl_util_UCCSD uses {WRAPPER_ITER_MAX_CAP}"
    )
    return sanitize_x_vec_params, True


def effective_row(ham_file, label, x_vec, sanitize):
    """Requested vector -> CSV row of the values the wrapper will actually use."""
    x_eff = sanitize(x_vec)
    return {
        "ham": ham_file,
        "label": label,
        "reps": int(round(x_eff[3])),
        "electron_fraction": int(round(x_eff[4])),
        "generalized": int(round(x_eff[5])),
        "initial_state": int(round(x_eff[2])),
        "optimizer": int(round(x_eff[6])),
        "iter_max": int(round(x_eff[7])),
        "initial_point_seed": int(round(x_eff[12])),
        "status": "",
        "energy": "",
        "error": "",
    }


def write_results(rows, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {output_csv}")


def print_table_header(col_w):
    print(f"{'label':<{col_w}} {'reps':>4} {'ef':>3} {'gen':>4} {'is':>3} "
          f"{'opt':>4} {'imax':>6} {'seed':>6}  status     energy")
    print("-" * 88)


def print_table_row(row, col_w, symbol, energy_str):
    print(f"{row['label']:<{col_w}} {row['reps']:>4} {row['electron_fraction']:>3} "
          f"{row['generalized']:>4} {row['initial_state']:>3} {row['optimizer']:>4} "
          f"{row['iter_max']:>6} {row['initial_point_seed']:>6}  "
          f"{symbol} {row['status']:<6}  {energy_str}")


def run_diagnostic(ham_dir, ham_file, ham_seed, dry_run=False):
    sanitize, is_effective = load_sanitizer()

    print(f"\n{'='*70}")
    print(f"UCCSD 8-qubit diagnostic  —  {ham_file}")
    print(f"{'='*70}\n")
    print(f"Combinations: {len(COMBINATIONS)}  |  iter_max cap: {ITER_MAX_CAP}"
          f"  |  initial_point_seed={SEED0} runs: {_n_seed0}")
    if is_effective:
        print("Columns are effective (post-sanitize) values, not requested ones.\n")
    else:
        print("Columns are REQUESTED values: kcl_util_UCCSD is unavailable here, so\n"
              "the wrapper's sanitiser could not be applied.\n")

    col_w = 26
    if dry_run:
        rows = [
            effective_row(ham_file, label, x_vec, sanitize)
            for label, x_vec in COMBINATIONS
        ]
        print(">> Dry run: no Hamiltonian loaded, no VQE executed, no CSV written.\n")
        print_table_header(col_w)
        for row in rows:
            row["status"] = "SKIP"
            print_table_row(row, col_w, "-", "—")
        return rows

    if not is_effective:
        raise RuntimeError(
            "kcl_util_UCCSD could not be imported; a real run needs the quantum "
            "stack (qiskit/quri_parts). Use --dry-run to inspect the table only."
        )

    from kcl_util import process_file
    from kcl_util_UCCSD import wrapper_uccsd

    print(">> Loading Hamiltonian...")
    n_qubits, ham = process_file(ham_dir, ham_file)
    n_qubits = int(n_qubits)
    print(f"   n_qubits={n_qubits}\n")

    rows = []
    print_table_header(col_w)

    for label, x_vec in COMBINATIONS:
        row = effective_row(ham_file, label, x_vec, sanitize)
        try:
            energy = wrapper_uccsd(x_vec, n_qubits, ham,
                                   ham_seed=ham_seed,
                                   ham_dir=ham_dir)
            row["status"] = "PASS"
            row["energy"] = energy
            energy_str = f"{energy:.6f}"
        except Exception as e:
            row["status"] = "FAIL"
            row["error"] = type(e).__name__ + ": " + str(e)[:120]
            energy_str = "—"

        symbol = "✓" if row["status"] == "PASS" else "✗"
        print_table_row(row, col_w, symbol, energy_str)
        rows.append(row)

    print()
    passes = sum(1 for r in rows if r["status"] == "PASS")
    fails = sum(1 for r in rows if r["status"] == "FAIL")
    print(f"Summary: {passes} PASS / {fails} FAIL out of {len(rows)}")

    write_results(rows, OUTPUT_CSV)

    fail_rows = [r for r in rows if r["status"] == "FAIL"]
    if fail_rows:
        print("\nFailure details:")
        for r in fail_rows:
            print(f"  {r['label']:<{col_w}} {r['error']}")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="UCCSD hyperparameter diagnostic on a single 8-qubit Hamiltonian"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the combination table and exit without running VQE",
    )
    args = parser.parse_args()

    run_diagnostic(HAM_FOLDER, HAM_FILE, HAM_SEED, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
