#!/usr/bin/env python3
"""
Diagnostic: re-run the HEA default baseline for one Hamiltonian across several
initial_point_seed values (IND:12).

Every baseline in Final_VQE.csv uses default_hea_x_vec with a single fixed
initial_point_seed, so there is exactly one COBYLA start per Hamiltonian. Varying
only IND:12 separates an unlucky start from a genuine landscape property — the
question raised by 16qubits_05, the only baseline with a positive energy.

Results go to diagnostics/baseline_seed_sweep.csv, with a column data dictionary
written alongside it as diagnostics/baseline_seed_sweep_columns.csv. Final_VQE.csv
is never written: it is keyed by (ham_prefix, implementation, is_baseline), so seed
variants would collide with the canonical baseline row and corrupt Comparison D.

Usage (from anywhere; paths are resolved against the repository root):
    ./diagnostics/baseline_seed_sweep.py
    ./diagnostics/baseline_seed_sweep.py --ham-prefix 24qubits_09 --seeds 0 1 2
"""

import argparse
import csv
import os
import re
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
# kcl_util appends a relative "../" and imports utils.challenge_2024, so the repo
# root must be importable regardless of the cwd this script is launched from.
for _path in (SRC_DIR, REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

OUTPUT_CSV = os.path.join(SCRIPT_DIR, "baseline_seed_sweep.csv")
COLUMN_DOC_CSV = os.path.join(SCRIPT_DIR, "baseline_seed_sweep_columns.csv")
FINAL_VQE_CSV = os.path.join(REPO_ROOT, "data", "Final_VQE_run", "Final_VQE.csv")

SWEEP_FIELDS = [
    "ham_prefix",
    "n_qubits",
    "initial_point_seed",
    "vqe_energy",
    "n_iterations",
    "iter_max",
    "converged",
    "run_time_s",
    "status",
    "error_message",
]

COLUMN_DOC_FIELDS = ["column", "dtype", "units", "description"]

# sanitize_x_vec_params clamps IND:12 into this range; -1 is the reserved NN_SEED
# sentinel, so out-of-range seeds would silently run as seed 0.
SEED_MIN = 0
SEED_MAX = 10 ** 4

# Data dictionary for baseline_seed_sweep.csv, keyed by column name so the doc is
# written from the same list the sweep rows are written from and cannot drift.
COLUMN_DOC = {
    "ham_prefix": (
        "str", "",
        "Hamiltonian identifier, '<n>qubits_<seed>'; matches Final_VQE.csv ham_prefix",
    ),
    "n_qubits": (
        "int", "qubits",
        "Qubit count parsed from ham_prefix; sets x_vec IND:0",
    ),
    "initial_point_seed": (
        "int", "",
        f"Swept hyperparameter x_vec IND:12, RNG seed for the COBYLA start "
        f"(valid range [{SEED_MIN}, {SEED_MAX}]); the only value that varies between rows",
    ),
    "vqe_energy": (
        "float", "Hartree",
        "Final VQE energy for this seed; empty when status=failed",
    ),
    "n_iterations": (
        "int", "iterations",
        "Optimizer iterations actually used",
    ),
    "iter_max": (
        "int", "iterations",
        "Effective iteration cap from default_hea_x_vec (IND:7), post-sanitize",
    ),
    "converged": (
        "bool", "",
        "True when the optimizer reported convergence before hitting iter_max",
    ),
    "run_time_s": (
        "float", "seconds",
        "Wall-clock time for this single VQE run",
    ),
    "status": (
        "str", "",
        "'success' or 'failed'; failed rows carry error_message and no energy",
    ),
    "error_message": (
        "str", "",
        "Exception text when status=failed, empty otherwise",
    ),
}

assert set(COLUMN_DOC) == set(SWEEP_FIELDS), (
    "COLUMN_DOC must document exactly the columns in SWEEP_FIELDS; "
    f"undocumented={sorted(set(SWEEP_FIELDS) - set(COLUMN_DOC))}, "
    f"unknown={sorted(set(COLUMN_DOC) - set(SWEEP_FIELDS))}"
)

DEFAULT_HAM_PREFIX = "16qubits_05"
DEFAULT_SEEDS = [0, 1, 2, 3]

_PREFIX_RE = re.compile(r"^(\d+)qubits_(\d+)$")


def parse_n_qubits(ham_prefix):
    match = _PREFIX_RE.match(ham_prefix.strip())
    if not match:
        raise ValueError(f"Cannot parse ham_prefix: {ham_prefix!r}")
    return int(match.group(1))


def validate_seeds(seeds):
    for seed in seeds:
        if seed < SEED_MIN or seed > SEED_MAX:
            raise ValueError(
                f"initial_point_seed must be in [{SEED_MIN}, {SEED_MAX}]; got {seed}"
            )
    return seeds


def recorded_baseline(ham_prefix):
    """Canonical HEA baseline energy from Final_VQE.csv, or None (read-only)."""
    if not os.path.exists(FINAL_VQE_CSV):
        return None
    with open(FINAL_VQE_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("ham_prefix", "").strip() != ham_prefix:
                continue
            if row.get("implementation", "").strip() != "HEA":
                continue
            if str(row.get("is_baseline", "")).strip().lower() not in ("true", "1", "yes"):
                continue
            if row.get("status") != "success":
                continue
            try:
                return float(row["vqe_energy"])
            except (KeyError, TypeError, ValueError):
                return None
    return None


def sweep_row(ham_prefix, n_qubits, seed):
    # Imported lazily so --doc-only needs no quantum stack (qiskit/quri_parts).
    from final_VQE_run import parse_ham_seed, run_hea_vqe
    from kcl_util_HEA import default_hea_x_vec

    x_vec = default_hea_x_vec(n_qubits)
    x_vec[12] = float(seed)

    row = {
        "ham_prefix": ham_prefix,
        "n_qubits": n_qubits,
        "initial_point_seed": seed,
    }
    try:
        t0 = time.perf_counter()
        metrics = run_hea_vqe(x_vec, n_qubits, parse_ham_seed(ham_prefix))
        elapsed = time.perf_counter() - t0
        row.update(
            vqe_energy=metrics["vqe_energy"],
            n_iterations=metrics["n_iterations"],
            iter_max=metrics["iter_max"],
            converged=metrics["converged"],
            run_time_s=elapsed,
            status="success",
            error_message="",
        )
    except Exception as exc:
        row.update(status="failed", error_message=str(exc))
    return row


def row_key(row):
    return (
        str(row.get("ham_prefix", "")).strip(),
        str(row.get("initial_point_seed", "")).strip(),
    )


def write_column_doc(output_path=COLUMN_DOC_CSV):
    """Write the data dictionary for baseline_seed_sweep.csv in SWEEP_FIELDS order."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMN_DOC_FIELDS)
        writer.writeheader()
        for column in SWEEP_FIELDS:
            dtype, units, description = COLUMN_DOC[column]
            writer.writerow({
                "column": column,
                "dtype": dtype,
                "units": units,
                "description": description,
            })
    return output_path


def upsert_rows(new_rows, output_path):
    """Merge new rows into the sweep CSV by (ham_prefix, initial_point_seed).

    Called after every run so a long sweep is durable if interrupted.
    """
    existing = {}
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        with open(output_path, newline="") as f:
            for row in csv.DictReader(f):
                existing[row_key(row)] = row

    for row in new_rows:
        existing[row_key(row)] = row

    merged = sorted(
        existing.values(),
        key=lambda r: (str(r.get("ham_prefix", "")), int(r.get("initial_point_seed", 0))),
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SWEEP_FIELDS, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)
    return len(merged)


def print_verdict(ham_prefix, rows):
    energies = [r["vqe_energy"] for r in rows if r["status"] == "success"]
    print()
    print("=" * 70)
    print(f"Verdict — {ham_prefix} baseline across {len(rows)} seed(s)")
    print("=" * 70)

    if not energies:
        print("  No successful runs; see error_message in the CSV.")
        return

    n_positive = sum(1 for e in energies if e > 0)
    n_negative = len(energies) - n_positive
    print(f"  successful runs:   {len(energies)}/{len(rows)}")
    print(f"  energy min/max:    {min(energies):.10f} / {max(energies):.10f}")
    print(f"  energy mean +/- std: {np.mean(energies):.10f} +/- {np.std(energies):.10f}")
    print(f"  positive/negative: {n_positive}/{n_negative}")

    reference = recorded_baseline(ham_prefix)
    if reference is not None:
        print(f"  Final_VQE.csv baseline (reference): {reference:.10f}")

    print()
    if n_positive and n_negative:
        print("  Mixed signs across seeds: the recorded baseline reflects one particular")
        print("  COBYLA start, so single-baseline-per-Hamiltonian is a real limitation.")
    elif n_negative == len(energies):
        print("  All seeds negative: the recorded positive baseline was an unlucky start,")
        print("  not a landscape property.")
    else:
        print("  All seeds positive: a genuine property of this Hamiltonian under the")
        print("  default HEA circuit, reportable as-is with that context.")


def main():
    parser = argparse.ArgumentParser(
        description="Re-run the HEA default baseline across initial_point_seed values"
    )
    parser.add_argument(
        "--ham-prefix",
        default=DEFAULT_HAM_PREFIX,
        help=f"Hamiltonian prefix to sweep (default: {DEFAULT_HAM_PREFIX})",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help=f"initial_point_seed values (default: {' '.join(map(str, DEFAULT_SEEDS))})",
    )
    parser.add_argument(
        "--doc-only",
        action="store_true",
        help="write the column data dictionary and exit without running VQE",
    )
    args = parser.parse_args()

    if args.doc_only:
        print(f"Written: {write_column_doc()} ({len(SWEEP_FIELDS)} columns documented)")
        return 0

    ham_prefix = args.ham_prefix.strip()
    n_qubits = parse_n_qubits(ham_prefix)
    seeds = validate_seeds(args.seeds)

    print(f"Written: {write_column_doc()} ({len(SWEEP_FIELDS)} columns documented)")

    print(f"Baseline seed sweep | {ham_prefix} ({n_qubits} qubits) | seeds {seeds}")
    print(f"{'seed':>6}  {'energy':>18}  {'iters':>11}  {'converged':>9}  {'time_s':>8}  status")
    print("-" * 78)

    rows = []
    for seed in seeds:
        row = sweep_row(ham_prefix, n_qubits, seed)
        if row["status"] == "success":
            print(
                f"{seed:>6}  {row['vqe_energy']:>18.10f}  "
                f"{row['n_iterations']:>5}/{row['iter_max']:<5}  "
                f"{str(row['converged']):>9}  {row['run_time_s']:>8.1f}  success"
            )
        else:
            print(f"{seed:>6}  {'—':>18}  {'—':>11}  {'—':>9}  {'—':>8}  "
                  f"failed: {row['error_message'][:60]}")
        rows.append(row)
        n_total = upsert_rows([row], OUTPUT_CSV)

    print(f"\nWritten: {OUTPUT_CSV} ({n_total} rows after upsert)")
    print_verdict(ham_prefix, rows)
    return 0 if any(r["status"] == "success" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
