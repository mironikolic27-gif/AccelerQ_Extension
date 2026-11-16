import csv
import numpy as np
from scipy.sparse.linalg import eigsh
from kcl_util import process_file
from kcl_util_hamiltonian import to_qubit_operator

PREFIXES = ["14qubits_05", "14qubits_06", "16qubits_05", "16qubits_06"]
REF_CSV = "ham_reference_energies.csv"

results = []
for prefix in PREFIXES:
    n_qubits = int(prefix.split("qubits_")[0])
    _, ham = process_file("../hamiltonian/", prefix + ".data")
    operator = to_qubit_operator(ham)

    sparse_matrix = operator.to_matrix(sparse=True)  # scipy csr_matrix, no dense materialisation
    eigval, _ = eigsh(sparse_matrix, k=1, which="SA")  # smallest algebraic eigenvalue
    exact_energy = float(eigval[0])

    print(prefix, exact_energy)
    results.append({
        "ham_prefix": prefix,
        "n_qubits": n_qubits,
        "dataset": "",  # left blank deliberately — see note below
        "reference_energy": exact_energy,
        "reference_type": "exact (sparse diagonalisation)",
        "notes": "computed directly via scipy.sparse.linalg.eigsh, not from external benchmark table",
    })

with open(REF_CSV, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["ham_prefix","n_qubits","dataset","reference_energy","reference_type","notes"])
    for row in results:
        writer.writerow(row)

print(f"\nAppended {len(results)} rows to {REF_CSV}")